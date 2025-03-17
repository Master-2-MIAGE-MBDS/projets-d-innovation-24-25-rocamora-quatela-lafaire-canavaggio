using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using DeepBridgeWindowsApp.CUDA;

namespace DeepBridgeWindowsApp.Dicom
{
    public class DicomImageProcessor
    {
        // Utiliser CudaBatchProcessor pour une meilleure gestion de la mémoire GPU
        private static readonly CudaBatchProcessor batchProcessor = new CudaBatchProcessor();
        
        // Cache des slices actuellement utilisées
        private static readonly HashSet<string> activeSlices = new HashSet<string>();
        
        // Cache des bitmaps pour éviter des reconversions inutiles
        private static readonly Dictionary<string, WeakReference<Bitmap>> bitmapCache = 
            new Dictionary<string, WeakReference<Bitmap>>();
            
        // Nombre maximum de slices à garder en mémoire GPU simultanément
        private const int MAX_ACTIVE_SLICES = 10;
        
        // Verrou pour synchroniser l'accès au cache
        private static readonly object cacheLock = new object();

        public static Bitmap ConvertToBitmap(DicomMetadata metadata, int windowWidth = -1, int windowCenter = -1)
        {
            try
            {
                // Utiliser les valeurs par défaut si non spécifiées
                if (windowWidth == -1) windowWidth = metadata.WindowWidth;
                if (windowCenter == -1) windowCenter = metadata.WindowCenter;

                // Générer un identifiant unique pour cette tranche
                string sliceId = GetSliceId(metadata);
                // Créer une clé de cache qui inclut les paramètres de fenêtrage
                string cacheKey = $"{sliceId}_{windowWidth}_{windowCenter}";
                
                // Vérifier si nous avons déjà cette image en cache
                lock (cacheLock)
                {
                    if (bitmapCache.TryGetValue(cacheKey, out var weakRef) && 
                        weakRef.TryGetTarget(out var cachedBitmap))
                    {
                        Console.WriteLine($"Cache hit for slice {sliceId}");
                        return cachedBitmap;
                    }
                }
                
                // Marquer cette tranche comme active et nettoyer le cache si nécessaire
                bool needCleanup = false;
                lock (activeSlices)
                {
                    if (activeSlices.Count >= MAX_ACTIVE_SLICES)
                    {
                        needCleanup = true;
                    }
                    activeSlices.Add(sliceId);
                }
                
                if (needCleanup)
                {
                    // Nettoyer les ressources si nous avons trop de slices actives
                    CleanupInactiveResources();
                }

                // Charger les données de pixels depuis le stockage
                metadata.LoadPixelData();
                
                // Obtenir les données d'entrée
                var inputData = metadata.PixelData.ToArray();
                
                // Précharger cette tranche en mémoire GPU si ce n'est pas déjà fait
                batchProcessor.LoadDicomSliceToGPU(sliceId, inputData);
                
                // Traiter l'image avec CUDA
                var outputData = batchProcessor.ProcessSlice(
                    sliceId,
                    inputData,
                    windowCenter,
                    windowWidth,
                    metadata.BitsStored,
                    metadata.PixelRepresentation,
                    metadata.BitsAllocated,
                    metadata.RescaleSlope,
                    metadata.RescaleIntercept);

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
                
                // Libérer les données de pixel de la RAM maintenant qu'elles sont en VRAM
                metadata.UnloadPixelData();
                
                // Ajouter au cache avec une référence faible pour permettre au GC de nettoyer si nécessaire
                lock (cacheLock)
                {
                    bitmapCache[cacheKey] = new WeakReference<Bitmap>(bitmap);
                }

                return bitmap;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erreur lors de la conversion DICOM: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Nettoie les ressources inactives pour libérer de la mémoire
        /// </summary>
        private static void CleanupInactiveResources()
        {
            Console.WriteLine("Nettoyage des ressources inactives...");
            
            // 1. Nettoyer le cache de bitmaps
            lock (cacheLock)
            {
                var keysToRemove = new List<string>();
                foreach (var kvp in bitmapCache)
                {
                    if (!kvp.Value.TryGetTarget(out _))
                    {
                        keysToRemove.Add(kvp.Key);
                    }
                }
                
                foreach (var key in keysToRemove)
                {
                    bitmapCache.Remove(key);
                }
                
                Console.WriteLine($"Nettoyé {keysToRemove.Count} bitmaps du cache");
            }
            
            // 2. Nettoyer les ressources GPU
            batchProcessor.CleanupUnusedBuffers(activeSlices);
            
            // 3. Forcer le ramasse-miettes pour libérer la mémoire
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        
        /// <summary>
        /// Précharge plusieurs tranches DICOM en mémoire GPU, mais charge les données depuis le disque à la demande
        /// </summary>
        public static void PreloadSlices(DicomMetadata[] slices)
        {
            if (slices == null || slices.Length == 0)
                return;
                
            // Limiter le nombre de slices à précharger pour éviter de surcharger la mémoire GPU
            int count = Math.Min(slices.Length, MAX_ACTIVE_SLICES);
            var slicesToLoad = new Dictionary<string, byte[]>();
            
            Console.WriteLine($"Préchargement de {count} tranches DICOM sur {slices.Length} disponibles");
            
            for (int i = 0; i < count; i++)
            {
                var metadata = slices[i];
                string sliceId = GetSliceId(metadata);
                
                // Charger les données depuis le disque
                metadata.LoadPixelData();
                
                // Ajouter à la liste des données à précharger en GPU
                slicesToLoad[sliceId] = metadata.PixelData.ToArray();
                
                // Marquer comme active
                lock (activeSlices)
                {
                    activeSlices.Add(sliceId);
                }
                
                // Libérer immédiatement de la RAM - maintenant en VRAM
                metadata.UnloadPixelData();
            }
            
            // Précharger toutes les tranches en une seule opération
            batchProcessor.PreloadBatch(slicesToLoad);
            Console.WriteLine($"Terminé le préchargement de {count} tranches DICOM en VRAM");
        }
        
        /// <summary>
        /// Libère une tranche de la mémoire active
        /// </summary>
        public static void ReleaseSlice(DicomMetadata metadata)
        {
            string sliceId = GetSliceId(metadata);
            
            lock (activeSlices)
            {
                activeSlices.Remove(sliceId);
            }
        }
        
        /// <summary>
        /// Génère un identifiant unique pour une tranche DICOM
        /// </summary>
        private static string GetSliceId(DicomMetadata metadata)
        {
            // Utiliser une combinaison de propriétés qui identifient de façon unique la tranche
            return $"{metadata.PatientID}_{metadata.Series}_{metadata.SliceLocation}";
        }

        /// <summary>
        /// Nettoie toutes les ressources utilisées par le processeur d'images
        /// </summary>
        public static void Cleanup()
        {
            try
            {
                lock (cacheLock)
                {
                    foreach (var weakRef in bitmapCache.Values)
                    {
                        if (weakRef.TryGetTarget(out var bitmap))
                        {
                            bitmap.Dispose();
                        }
                    }
                    bitmapCache.Clear();
                }
                
                batchProcessor?.Dispose();
                DicomMetadata.CleanupCache();
                
                // Forcer le nettoyage complet de la mémoire
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                
                Console.WriteLine("Nettoyage complet des ressources terminé");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erreur lors du nettoyage: {ex.Message}");
            }
        }
    }
}