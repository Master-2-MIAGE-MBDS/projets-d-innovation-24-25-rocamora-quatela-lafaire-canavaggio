using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using DeepBridgeWindowsApp.CUDA;

namespace DeepBridgeWindowsApp.Dicom
{
    public class DicomImageProcessor
    {
        private static readonly CudaProcessor cudaProcessor = new CudaProcessor();

        public static Bitmap ConvertToBitmap(DicomMetadata metadata, int windowWidth = -1, int windowCenter = -1)
        {
            try
            {
                // Utiliser les valeurs par défaut si non spécifiées
                if (windowWidth == -1) windowWidth = metadata.WindowWidth;
                if (windowCenter == -1) windowCenter = metadata.WindowCenter;

                // Préparer les données d'entrée
                var inputData = metadata.PixelData.ToArray();
                var outputLength = (inputData.Length / 2) * 4; // Format RGBA
                var outputData = new byte[outputLength];

                // Traiter l'image avec CUDA
                cudaProcessor.ProcessImage(
                    inputData,
                    outputData,
                    windowCenter,
                    windowWidth,
                    metadata.BitsStored,
                    metadata.PixelRepresentation,
                    metadata.BitsAllocated,
                    (float)metadata.RescaleSlope,
                    (float)metadata.RescaleIntercept);

                // Créer le bitmap résultant
                var bitmap = new Bitmap(metadata.Columns, metadata.Rows, PixelFormat.Format32bppArgb);
                var bitmapData = bitmap.LockBits(
                    new Rectangle(0, 0, metadata.Columns, metadata.Rows),
                    ImageLockMode.WriteOnly,
                    PixelFormat.Format32bppArgb);

                try
                {
                    Marshal.Copy(outputData, 0, bitmapData.Scan0, outputData.Length);
                }
                finally
                {
                    bitmap.UnlockBits(bitmapData);
                }

                return bitmap;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erreur lors de la conversion DICOM: {ex.Message}");
                throw;
            }
        }

        public static void Cleanup()
        {
            cudaProcessor?.Dispose();
        }
    }
}