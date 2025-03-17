using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using EvilDICOM.Core.Helpers;
using EvilDICOM.Core;
using EvilDICOM.Network.DIMSE.IOD;
using System.Net.Mime;
using System.Runtime.CompilerServices;

namespace DeepBridgeWindowsApp.Dicom
{
    public class DicomMetadata : IDisposable
    {
        // Patient and study metadata
        public string PatientID { get; set; }
        public string PatientName { get; set; }
        public string PatientSex { get; set; }
        public string Modality { get; set; }
        public int Series { get; set; }
        public string SeriesTime { get; set; }
        public string ContentTime { get; set; }
        
        // Image properties
        public int Rows { get; set; }
        public int Columns { get; set; }
        public int WindowCenter { get; set; }
        public int WindowWidth { get; set; }
        public double SliceThickness { get; set; }        // (0018,0050) DS
        public double SliceLocation { get; set; }         // (0020,1041) DS
        public double PixelSpacing { get; set; }          // (0028,0030) DS[2]
        
        // Pixel data - can be loaded on demand or stored on disk
        private List<byte> pixelData;
        private string pixelDataCachePath;
        private bool isPixelDataLoaded = false;
        
        // Image type metadata
        public int BitsAllocated { get; set; }
        public int BitsStored { get; set; }
        public int HighBit { get; set; }
        public int PixelRepresentation { get; set; }
        public double RescaleIntercept { get; set; }
        public double RescaleSlope { get; set; }
        public int? PixelPaddingValue { get; set; }
        
        // Cache directory for temporary pixel data storage
        private static readonly string TempCacheDir = Path.Combine(
            Path.GetTempPath(), 
            "DeepBridgeCache",
            Guid.NewGuid().ToString());
            
        // Static constructor to create the cache directory
        static DicomMetadata()
        {
            if (!Directory.Exists(TempCacheDir))
            {
                Directory.CreateDirectory(TempCacheDir);
            }
            
            // Register cleanup on application exit
            AppDomain.CurrentDomain.ProcessExit += (s, e) => CleanupCache();
        }
        
        /// <summary>
        /// Creates a new DicomMetadata instance from a DICOM object
        /// </summary>
        public DicomMetadata(DICOMObject dicomObject)
        {
            PatientID = dicomObject.FindFirst(TagHelper.PatientID)?.DData.ToString();
            PatientName = dicomObject.FindFirst(TagHelper.PatientName)?.DData.ToString();
            PatientSex = dicomObject.FindFirst(TagHelper.PatientSex)?.DData.ToString();
            Modality = dicomObject.FindFirst(TagHelper.Modality)?.DData.ToString();
            Series = Convert.ToInt32(dicomObject.FindFirst(TagHelper.SeriesNumber)?.DData ?? 0);
            SeriesTime = dicomObject.FindFirst(TagHelper.SeriesTime)?.DData.ToString();
            ContentTime = dicomObject.FindFirst(TagHelper.ContentTime)?.DData.ToString();
            Rows = Convert.ToInt32(dicomObject.FindFirst(TagHelper.Rows)?.DData ?? 0);
            Columns = Convert.ToInt32(dicomObject.FindFirst(TagHelper.Columns)?.DData ?? 0);
            WindowCenter = Convert.ToInt32(dicomObject.FindFirst(TagHelper.WindowCenter)?.DData ?? 0);
            WindowWidth = Convert.ToInt32(dicomObject.FindFirst(TagHelper.WindowWidth)?.DData ?? 0);
            SliceThickness = Convert.ToDouble(dicomObject.FindFirst(TagHelper.SliceThickness)?.DData ?? 0);
            SliceLocation = Convert.ToDouble(dicomObject.FindFirst(TagHelper.SliceLocation)?.DData ?? 0);
            PixelSpacing = Convert.ToDouble(dicomObject.FindFirst(TagHelper.PixelSpacing)?.DData.ToString().Split('\\').Select(double.Parse).ToArray()[0]);
            BitsAllocated = Convert.ToInt32(dicomObject.FindFirst(TagHelper.BitsAllocated).DData);
            BitsStored = Convert.ToInt32(dicomObject.FindFirst(TagHelper.BitsStored).DData);
            HighBit = Convert.ToInt32(dicomObject.FindFirst(TagHelper.HighBit).DData);
            PixelRepresentation = Convert.ToInt32(dicomObject.FindFirst(TagHelper.PixelRepresentation).DData);
            RescaleIntercept = Convert.ToDouble(dicomObject.FindFirst(TagHelper.RescaleIntercept).DData);
            RescaleSlope = Convert.ToDouble(dicomObject.FindFirst(TagHelper.RescaleSlope).DData);
            PixelPaddingValue = dicomObject.FindFirst(TagHelper.PixelPaddingValue)?.DData != null
                ? Convert.ToInt32(dicomObject.FindFirst(TagHelper.PixelPaddingValue).DData)
                : null;
                
            // Store pixel data temporarily and cache it to disk
            pixelData = (List<byte>)dicomObject.FindFirst(TagHelper.PixelData).DData_;
            CachePixelData();
            
            // Clear the in-memory copy to reduce RAM usage
            UnloadPixelData();
        }

        /// <summary>
        /// Gets the pixel data for this DICOM image, loading it from cache if necessary
        /// </summary>
        public List<byte> PixelData 
        { 
            get 
            {
                if (!isPixelDataLoaded)
                {
                    LoadPixelData();
                }
                return pixelData;
            }
        }
        
        /// <summary>
        /// Loads pixel data from disk cache into memory
        /// </summary>
        public void LoadPixelData()
        {
            if (isPixelDataLoaded) return;
            
            lock (this)
            {
                if (!isPixelDataLoaded && File.Exists(pixelDataCachePath))
                {
                    pixelData = new List<byte>(File.ReadAllBytes(pixelDataCachePath));
                    isPixelDataLoaded = true;
                    Console.WriteLine($"Loaded pixel data for slice {SliceLocation} from cache");
                }
            }
        }
        
        /// <summary>
        /// Unloads pixel data from memory to free up RAM
        /// </summary>
        public void UnloadPixelData()
        {
            if (!isPixelDataLoaded) return;
            
            lock (this)
            {
                pixelData = null;
                isPixelDataLoaded = false;
                GC.Collect();
                Console.WriteLine($"Unloaded pixel data for slice {SliceLocation} from memory");
            }
        }
        
        /// <summary>
        /// Caches pixel data to disk to reduce memory usage
        /// </summary>
        private void CachePixelData()
        {
            // Generate a unique filename based on some unique attributes of this slice
            string filename = $"{PatientID ?? "unknown"}_{Series}_{SliceLocation:F2}.bin";
            pixelDataCachePath = Path.Combine(TempCacheDir, filename);
            
            try
            {
                // Write the pixel data to disk
                File.WriteAllBytes(pixelDataCachePath, pixelData.ToArray());
                Console.WriteLine($"Cached pixel data for slice {SliceLocation} to disk");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error caching pixel data: {ex.Message}");
                // If we can't cache to disk, keep in memory
                isPixelDataLoaded = true;
                return;
            }
        }
        
        /// <summary>
        /// Cleans up the temporary cache directory
        /// </summary>
        public static void CleanupCache()
        {
            try
            {
                if (Directory.Exists(TempCacheDir))
                {
                    Directory.Delete(TempCacheDir, true);
                    Console.WriteLine("Cleaned up DICOM pixel data cache");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error cleaning up cache: {ex.Message}");
            }
        }

        public void PrintInfo()
        {
            Console.WriteLine($"Series: {Series}");
            Console.WriteLine($"Series Time: {SeriesTime}");
            Console.WriteLine($"Modality: {Modality}");
            Console.WriteLine($"Rows: {Rows}");
            Console.WriteLine($"Columns: {Columns}");
            Console.WriteLine($"Window Center: {WindowCenter}");
            Console.WriteLine($"Window Width: {WindowWidth}");
            Console.WriteLine($"Slice Thickness: {SliceThickness}");
            Console.WriteLine($"Slice Location: {SliceLocation}");
            Console.WriteLine($"Pixel Spacing: {PixelSpacing} x {PixelSpacing}");
            Console.WriteLine($"Pixel Data Loaded: {isPixelDataLoaded}");
        }
        
        public void Dispose()
        {
            UnloadPixelData();
        }
    }
}
