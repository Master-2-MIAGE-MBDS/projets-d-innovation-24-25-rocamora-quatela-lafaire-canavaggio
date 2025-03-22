using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using EvilDICOM.Core.Helpers;
using EvilDICOM.Core.Interfaces;
using EvilDICOM.Core;
using System.IO;
using System.Xml.Linq;
using DeepBridgeWindowsApp.Dicom;
using System.Diagnostics;

namespace DeepBridgeWindowsApp.DICOM
{
    public class DicomReader : IDisposable
    {
        private readonly string directoryPath;
        public DicomMetadata[] Slices { get; private set; }
        public DicomMetadata GlobalView { get; private set; }
        
        // Maximum number of slices to load at once to avoid memory overflow
        private const int BATCH_LOAD_SIZE = 10;

        public DicomReader(string directoryPath)
        {
            this.directoryPath = directoryPath;
        }

        private string[] GetValidatedDicomFiles()
        {
            var dicomFiles = Directory.GetFiles(directoryPath, "*.dcm");
            if (dicomFiles.Length == 0)
                throw new FileNotFoundException("No DICOM files found in the directory.");
            return dicomFiles;
        }

        /// <summary>
        /// Loads just the global view (first file) to get metadata
        /// </summary>
        public void LoadGlobalView()
        {
            var dicomFiles = GetValidatedDicomFiles();
            var firstDicom = DICOMObject.Read(dicomFiles[0]);
            GlobalView = new DicomMetadata(firstDicom);
        }

        /// <summary>
        /// Loads all DICOM files in batches to reduce memory usage
        /// </summary>
        public void LoadAllFiles()
        {
            var stopwatch = Stopwatch.StartNew();
            
            var dicomFiles = GetValidatedDicomFiles();
            Console.WriteLine($"Found {dicomFiles.Length} DICOM files.");

            // Load first file for global view
            var firstDicom = DICOMObject.Read(dicomFiles[0]);
            GlobalView = new DicomMetadata(firstDicom);

            var seriesNumber = firstDicom.FindFirst(TagHelper.SeriesNumber).DData.ToString();
            
            // Preprocess file list to get location information
            Console.WriteLine("Prétraitement des fichiers DICOM...");
            var fileInfo = new List<(string FilePath, double SliceLocation)>();
            
            foreach (var file in dicomFiles)
            {
                // Read the file - since we're only accessing metadata, we don't need to worry
                // about the full pixel data being loaded into memory temporarily
                var dcm = DICOMObject.Read(file);
                var sliceLocation = Convert.ToDouble(dcm.FindFirst(TagHelper.SliceLocation)?.DData ?? 0);
                fileInfo.Add((file, sliceLocation));
                
                // Force cleanup
                dcm = null;
                GC.Collect();
            }
            
            // Sort files by slice location
            var sortedFiles = fileInfo.OrderBy(f => f.SliceLocation).Select(f => f.FilePath).ToArray();
            
            // Prepare final slices array
            Slices = new DicomMetadata[sortedFiles.Length];
            
            // Load in batches to reduce memory usage
            for (int batchStart = 0; batchStart < sortedFiles.Length; batchStart += BATCH_LOAD_SIZE)
            {
                int batchSize = Math.Min(BATCH_LOAD_SIZE, sortedFiles.Length - batchStart);
                Console.WriteLine($"Chargement du lot {batchStart / BATCH_LOAD_SIZE + 1}/{(sortedFiles.Length + BATCH_LOAD_SIZE - 1) / BATCH_LOAD_SIZE}...");
                
                // Load this batch
                for (int i = 0; i < batchSize; i++)
                {
                    int currentIndex = batchStart + i;
                    var dicomObject = DICOMObject.Read(sortedFiles[currentIndex]);
                    var currentSeriesNumber = dicomObject.FindFirst(TagHelper.SeriesNumber).DData.ToString();
                    
                    if (currentSeriesNumber != seriesNumber)
                    {
                        throw new InvalidOperationException("All DICOM files must be part of the same series.");
                    }
                    
                    // Create metadata - this will cache pixel data to disk and free memory
                    Slices[currentIndex] = new DicomMetadata(dicomObject);
                    
                    // Force memory cleanup
                    dicomObject = null;
                    GC.Collect();
                }
            }
            
            stopwatch.Stop();
            Console.WriteLine($"Chargement terminé en {stopwatch.ElapsedMilliseconds/1000.0:F1} secondes");
            Console.WriteLine($"Consommation mémoire après chargement: {GC.GetTotalMemory(true) / (1024*1024)} MB");
        }
        
        public void Dispose()
        {
            // Clean up slices
            if (Slices != null)
            {
                foreach (var slice in Slices)
                {
                    slice?.Dispose();
                }
                Slices = null;
            }
            
            // Clean up global view
            GlobalView?.Dispose();
            GlobalView = null;
            
            // Force garbage collection
            GC.Collect();
        }
    }
}
