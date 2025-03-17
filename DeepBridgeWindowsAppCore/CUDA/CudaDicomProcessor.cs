using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Drawing;

namespace DeepBridgeWindowsApp.CUDA
{
    /// <summary>
    /// Processeur CUDA pour le traitement accéléré des images DICOM.
    /// Gère l'initialisation du GPU et l'exécution des kernels CUDA.
    /// </summary>
    public class CudaDicomProcessor : IDisposable
    {
        // Contexte ILGPU et accélérateur
        private Context context;
        private Accelerator accelerator;

        // Définition du kernel avec tous les paramètres nécessaires
        private Action<Index1D, ArrayView<byte>, ArrayView<byte>,
            int, int, int, int, int, double, double> pixelKernel;

        /// <summary>
        /// Initialise le processeur CUDA et configure l'environnement d'exécution.
        /// </summary>
        public CudaDicomProcessor()
        {
            // Initialiser le contexte ILGPU avec support CUDA et CPU
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());

            // Afficher les accélérateurs disponibles
            var availableDevices = context.Devices;
            Console.WriteLine("Available Accelerators:");
            foreach (var deviceInfo in availableDevices)
            {
                Console.WriteLine($"- {deviceInfo.Name} ({deviceInfo.AcceleratorType})");
            }

            // Sélectionner l'accélérateur préféré (CUDA si disponible)
            var selectedDevice = context.GetPreferredDevice(preferCPU: false);
            accelerator = selectedDevice.CreateAccelerator(context);

            Console.WriteLine($"\nUsing accelerator: {accelerator.Name}");
            Console.WriteLine($"Accelerator Type: {accelerator.AcceleratorType}");

            // Compiler le kernel
            pixelKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>,
                ArrayView<byte>, int, int, int, int, int, double, double>(ProcessPixelKernel);
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

            // Calculer l'intensité et la transparence
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

        /// <summary>
        /// Traite les données de pixels DICOM en utilisant CUDA.
        /// </summary>
        public byte[] ProcessPixelData(
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

            // Allouer la mémoire GPU et copier les données
            using (var deviceInput = accelerator.Allocate1D(inputData))
            using (var deviceOutput = accelerator.Allocate1D<byte>(outputLength))
            {
                // Copier les données d'entrée vers le GPU
                deviceInput.CopyFromCPU(inputData);

                // Exécuter le kernel
                var numElements = inputData.Length / 2;
                pixelKernel(numElements, deviceInput.View, deviceOutput.View,
                    windowCenter, windowWidth, bitsStored, pixelRepresentation,
                    bitsAllocated, rescaleSlope, rescaleIntercept);

                // Copier le résultat vers le CPU
                deviceOutput.CopyToCPU(output);
            }

            return output;
        }

        /// <summary>
        /// Libère les ressources CUDA.
        /// </summary>
        public void Dispose()
        {
            accelerator?.Dispose();
            context?.Dispose();
        }
    }
}