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
        private const int MAX_GPU_MEMORY = 4 * 1024 * 1024 * 1024; // 4GB
        private readonly Context context;

        // L'accélérateur est maintenant public pour permettre l'accès depuis DicomImageProcessor
        public Accelerator Accelerator { get; private set; }

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
        /// Traite une seule tranche DICOM en utilisant CUDA.
        /// </summary>
        public void ProcessSlice(
            MemoryBuffer1D<byte, Stride1D.Dense> deviceInput,
            MemoryBuffer1D<byte, Stride1D.Dense> deviceOutput,
            int windowCenter,
            int windowWidth,
            int bitsStored,
            int pixelRepresentation,
            int bitsAllocated,
            double rescaleSlope,
            double rescaleIntercept)
        {
            // Calculer le nombre d'éléments à traiter (chaque pixel est sur 2 bytes)
            var numElements = deviceInput.Length / 2;

            // Exécuter le kernel CUDA
            pixelKernel(new Index1D((int)numElements), deviceInput.View, deviceOutput.View,
                windowCenter, windowWidth, bitsStored,
                pixelRepresentation, bitsAllocated,
                rescaleSlope, rescaleIntercept);

            // Synchroniser pour s'assurer que le traitement est terminé
            Accelerator.Synchronize();
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
            Accelerator?.Dispose();
            context?.Dispose();
        }
    }
}