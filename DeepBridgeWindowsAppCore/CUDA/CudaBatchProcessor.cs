using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace DeepBridgeWindowsApp.CUDA
{
    public class CudaBatchProcessor : IDisposable
    {
        private const int BATCH_SIZE = 100;
        private const long MAX_GPU_MEMORY = 4L * 1024L * 1024L * 1024L; // 4GB
        private readonly Context context;

        // L'accélérateur est maintenant public pour permettre l'accès depuis DicomImageProcessor
        public Accelerator Accelerator { get; private set; }

        // Dictionnaire pour stocker les tampons d'entrée DICOM persistants en VRAM
        private readonly Dictionary<string, MemoryBuffer1D<byte, Stride1D.Dense>> persistentInputBuffers = new Dictionary<string, MemoryBuffer1D<byte, Stride1D.Dense>>();
        
        // Buffer de sortie réutilisable
        private MemoryBuffer1D<byte, Stride1D.Dense> sharedOutputBuffer;

        private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, double, double> pixelKernel;

        public CudaBatchProcessor()
        {
            // Création du contexte avec support CUDA uniquement
            context = Context.Create(builder => builder.Cuda());

            // Sélection du premier accélérateur CUDA disponible
            var device = context.Devices.First(d => d.AcceleratorType == AcceleratorType.Cuda);
            Accelerator = device.CreateAccelerator(context);

            Console.WriteLine($"Using GPU: {Accelerator.Name}");
            Console.WriteLine($"Available Memory: {Accelerator.MemorySize / (1024 * 1024 * 1024)}GB");

            // Compilation du kernel
            pixelKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>,
                ArrayView<byte>, int, int, int, int, int, double, double>(ProcessPixelKernel);
        }

        /// <summary>
        /// Alloue un tampon d'entrée persistant en VRAM pour une tranche DICOM spécifique
        /// </summary>
        public void LoadDicomSliceToGPU(string sliceId, byte[] inputData)
        {
            if (persistentInputBuffers.ContainsKey(sliceId))
            {
                // Déjà chargé, on ne fait rien
                return;
            }

            // Allouer et copier les données d'entrée vers la VRAM
            var deviceBuffer = Accelerator.Allocate1D<byte>(inputData);
            deviceBuffer.CopyFromCPU(inputData);
            
            // Stocker le tampon dans notre dictionnaire
            persistentInputBuffers[sliceId] = deviceBuffer;
            
            Console.WriteLine($"Loaded slice {sliceId} to GPU memory");
        }
        
        /// <summary>
        /// Libère un tampon d'entrée persistant de la VRAM
        /// </summary>
        public void UnloadDicomSliceFromGPU(string sliceId)
        {
            if (persistentInputBuffers.TryGetValue(sliceId, out var buffer))
            {
                buffer.Dispose();
                persistentInputBuffers.Remove(sliceId);
                Console.WriteLine($"Unloaded slice {sliceId} from GPU memory");
            }
        }

        /// <summary>
        /// Traite une seule tranche DICOM en utilisant CUDA, avec préférence pour les tampons persistants
        /// </summary>
        public byte[] ProcessSlice(
            string sliceId,
            byte[] inputData,
            int windowCenter,
            int windowWidth,
            int bitsStored,
            int pixelRepresentation,
            int bitsAllocated,
            double rescaleSlope,
            double rescaleIntercept)
        {
            // Calculer la taille de sortie (RGBA = 4 bytes par pixel)
            var outputLength = (inputData.Length / 2) * 4;
            var output = new byte[outputLength];
            
            // Utiliser le tampon d'entrée persistant s'il existe, sinon en créer un temporaire
            MemoryBuffer1D<byte, Stride1D.Dense> deviceInput;
            bool usingPersistentBuffer = persistentInputBuffers.TryGetValue(sliceId, out deviceInput);
            
            try
            {
                if (!usingPersistentBuffer)
                {
                    // Créer un tampon temporaire
                    deviceInput = Accelerator.Allocate1D(inputData);
                    deviceInput.CopyFromCPU(inputData);
                }
                
                // Créer ou réutiliser le tampon de sortie
                if (sharedOutputBuffer == null || sharedOutputBuffer.Length < outputLength)
                {
                    sharedOutputBuffer?.Dispose();
                    sharedOutputBuffer = Accelerator.Allocate1D<byte>(outputLength);
                }

                // Calculer le nombre d'éléments à traiter (chaque pixel est sur 2 bytes)
                var numElements = inputData.Length / 2;

                // Exécuter le kernel CUDA
                pixelKernel(new Index1D((int)numElements), deviceInput.View, sharedOutputBuffer.View,
                    windowCenter, windowWidth, bitsStored,
                    pixelRepresentation, bitsAllocated,
                    rescaleSlope, rescaleIntercept);

                // Synchroniser pour s'assurer que le traitement est terminé
                Accelerator.Synchronize();
                
                // Copier le résultat vers le CPU
                sharedOutputBuffer.CopyToCPU(output);
            }
            finally
            {
                // Libérer le tampon temporaire si utilisé
                if (!usingPersistentBuffer)
                {
                    deviceInput?.Dispose();
                }
            }
            
            return output;
        }
        
        /// <summary>
        /// Précharge un ensemble de tranches DICOM en VRAM pour un accès plus rapide
        /// </summary>
        public void PreloadBatch(Dictionary<string, byte[]> slices)
        {
            // Libérer toute mémoire GPU inutilisée
            CleanupUnusedBuffers();
            
            foreach (var slice in slices)
            {
                LoadDicomSliceToGPU(slice.Key, slice.Value);
            }
        }
        
        /// <summary>
        /// Libère les tampons GPU qui ne sont plus nécessaires pour économiser de la mémoire
        /// </summary>
        public void CleanupUnusedBuffers(IEnumerable<string> activeSliceIds = null)
        {
            if (activeSliceIds == null)
            {
                return;
            }
            
            var activeSet = new HashSet<string>(activeSliceIds);
            var keysToRemove = persistentInputBuffers.Keys.Where(k => !activeSet.Contains(k)).ToList();
            
            foreach (var key in keysToRemove)
            {
                UnloadDicomSliceFromGPU(key);
            }
        }

        private static void ProcessPixelKernel(
            Index1D index,
            ArrayView<byte> inputData,
            ArrayView<byte> outputData,
            int windowCenter,
            int windowWidth,
            int bitsStored,
            int pixelRepresentation,
            int bitsAllocated,
            double rescaleSlope,
            double rescaleIntercept)
        {
            var i = index * 2;
            if (i >= inputData.Length - 1) return;

            // Convertir les deux bytes en valeur de pixel 16-bit
            ushort storedValue = (ushort)((inputData[i]) | (inputData[i + 1] << 8));

            // Appliquer le masque de bits
            var mask = (ushort)(ushort.MaxValue >> (bitsAllocated - bitsStored));
            var maskedValue = storedValue & mask;
            var maxValue = 1 << bitsStored;

            // Gérer les pixels signés
            double pixelValue = maskedValue;
            if (pixelRepresentation == 1 && pixelValue > (maxValue / 2))
            {
                pixelValue = pixelValue - maxValue;
            }

            // Convertir en unités réelles
            double realValue = (pixelValue * rescaleSlope) + rescaleIntercept;

            // Appliquer la transformation fenêtre/niveau
            double windowHalf = ((windowWidth - 1) / 2.0) - 0.5;
            byte intensity;
            byte alpha;

            if (realValue <= windowCenter - windowHalf)
            {
                intensity = 0;
                alpha = 0;
            }
            else if (realValue >= windowCenter + windowHalf)
            {
                intensity = 255;
                alpha = 255;
            }
            else
            {
                double normalized = (realValue - (windowCenter - windowHalf)) / windowWidth;
                intensity = (byte)(normalized * 255);
                alpha = 255;
            }

            // Écrire les valeurs RGBA
            var outIndex = (index * 4);
            outputData[outIndex] = intensity;     // B
            outputData[outIndex + 1] = intensity; // G
            outputData[outIndex + 2] = intensity; // R
            outputData[outIndex + 3] = alpha;     // A
        }

        public void Dispose()
        {
            // Nettoyer tous les tampons persistants
            foreach (var buffer in persistentInputBuffers.Values)
            {
                buffer.Dispose();
            }
            persistentInputBuffers.Clear();
            
            // Nettoyer le tampon de sortie partagé
            sharedOutputBuffer?.Dispose();
            
            Accelerator?.Dispose();
            context?.Dispose();
        }
    }
}