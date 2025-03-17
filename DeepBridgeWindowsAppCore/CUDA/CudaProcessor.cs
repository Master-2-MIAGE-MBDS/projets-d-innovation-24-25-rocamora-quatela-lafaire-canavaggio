using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Linq;

namespace DeepBridgeWindowsApp.CUDA
{
    /// <summary>
    /// Gère le traitement CUDA des images DICOM avec accélération GPU.
    /// </summary>
    public class CudaProcessor : IDisposable
    {
        // Contexte ILGPU et accélérateur
        private readonly Context context;
        public Accelerator Accelerator { get; private set; }

        // Kernel pour le traitement des pixels
        private readonly Action<Index1D,
            ArrayView<byte>,
            ArrayView<byte>,
            int, int, int, int, int,
            float, float> pixelKernel;

        public CudaProcessor()
        {
            try
            {
                // Initialisation du contexte CUDA
                context = Context.Create(builder => builder.Cuda());

                // Recherche d'un GPU CUDA
                var cudaDevice = context.Devices
                    .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

                if (cudaDevice == null)
                {
                    throw new InvalidOperationException("Aucun GPU CUDA trouvé.");
                }

                // Création de l'accélérateur
                Accelerator = cudaDevice.CreateAccelerator(context);
                Console.WriteLine($"Utilisation du GPU: {Accelerator.Name}");
                Console.WriteLine($"Mémoire disponible: {Accelerator.MemorySize / (1024 * 1024 * 1024)}GB");

                // Compilation du kernel
                pixelKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView<byte>,
                    ArrayView<byte>,
                    int, int, int, int, int,
                    float, float>(ProcessPixelKernel);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erreur d'initialisation CUDA: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Traite une image DICOM sur le GPU.
        /// </summary>
        public unsafe void ProcessImage(
            byte[] inputData,
            byte[] outputData,
            int windowCenter,
            int windowWidth,
            int bitsStored,
            int pixelRepresentation,
            int bitsAllocated,
            float rescaleSlope,
            float rescaleIntercept)
        {
            using var deviceInput = Accelerator.Allocate1D<byte>(inputData);
            using var deviceOutput = Accelerator.Allocate1D<byte>(outputData.Length);

            // Copier les données d'entrée vers le GPU
            deviceInput.CopyFromCPU(inputData);

            // Exécuter le kernel
            var numElements = inputData.Length / 2;
            pixelKernel(numElements,
                deviceInput.View,
                deviceOutput.View,
                windowCenter, windowWidth,
                bitsStored, pixelRepresentation, bitsAllocated,
                rescaleSlope, rescaleIntercept);

            // Synchroniser et récupérer les résultats
            Accelerator.Synchronize();
            deviceOutput.CopyToCPU(outputData);
        }

        /// <summary>
        /// Kernel CUDA pour le traitement des pixels DICOM.
        /// </summary>
        private static void ProcessPixelKernel(
            Index1D index,
            ArrayView<byte> inputData,
            ArrayView<byte> outputData,
            int windowCenter,
            int windowWidth,
            int bitsStored,
            int pixelRepresentation,
            int bitsAllocated,
            float rescaleSlope,
            float rescaleIntercept)
        {
            var i = index * 2;
            if (i >= inputData.Length - 1) return;

            // Combiner les bytes en une valeur 16-bit
            ushort storedValue = (ushort)((inputData[i]) | (inputData[i + 1] << 8));

            // Appliquer le masque de bits
            var mask = (ushort)(ushort.MaxValue >> (bitsAllocated - bitsStored));
            var maskedValue = storedValue & mask;

            // Gérer les pixels signés
            float pixelValue = maskedValue;
            if (pixelRepresentation == 1 && pixelValue > (1 << (bitsStored - 1)))
            {
                pixelValue -= (1 << bitsStored);
            }

            // Convertir en unités réelles
            float realValue = (pixelValue * rescaleSlope) + rescaleIntercept;

            // Appliquer la transformation fenêtre/niveau
            float windowHalf = (windowWidth - 1) / 2.0f;
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
                float normalized = (realValue - (windowCenter - windowHalf)) / windowWidth;
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
            Accelerator?.Dispose();
            context?.Dispose();
        }
    }
}