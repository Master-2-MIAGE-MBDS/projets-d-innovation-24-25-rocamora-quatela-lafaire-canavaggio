using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using DeepBridgeWindowsApp.DICOM;

namespace DeepBridgeWindowsApp.Dicom
{
    public class DicomDisplayManager : IDisposable
    {
        // Tranches DICOM chargées
        public DicomMetadata[] slices;
        public DicomMetadata globalView { get; private set; }
        
        // État actuel de l'affichage
        private int currentSliceIndex;
        public int windowWidth { get; private set; }
        public int windowCenter { get; private set; }
        
        // Taille du cache en mémoire GPU
        private const int PRELOAD_WINDOW_SIZE = 5;
        
        // Intervalle de nettoyage automatique (en millisecondes)
        private const int CLEANUP_INTERVAL = 30000; // 30 secondes
        
        // Minuteur pour le nettoyage automatique
        private Timer cleanupTimer;
        
        // Verrou pour les opérations de chargement
        private readonly object loadLock = new object();
        
        // Indique si le gestionnaire a été disposé
        private bool isDisposed = false;

        public DicomDisplayManager(DicomReader reader)
        {
            Console.WriteLine($"Initialisation du gestionnaire d'affichage DICOM avec {reader.Slices?.Length ?? 0} tranches");
            
            globalView = reader.GlobalView;
            slices = reader.Slices ?? Array.Empty<DicomMetadata>();
            currentSliceIndex = 0;
            windowWidth = slices.Length > 0 ? slices[0].WindowWidth : 0;
            windowCenter = slices.Length > 0 ? slices[0].WindowCenter : 0;
            
            // Précharger un ensemble initial de tranches en VRAM
            if (slices.Length > 0)
            {
                PreloadSlicesAroundIndex(0);
            }
            
            // Configurer le minuteur de nettoyage automatique
            cleanupTimer = new Timer(PerformPeriodicCleanup, null, CLEANUP_INTERVAL, CLEANUP_INTERVAL);
            
            // Afficher des statistiques de mémoire
            LogMemoryStats();
        }
        
        /// <summary>
        /// Affiche des statistiques sur l'utilisation de la mémoire
        /// </summary>
        private void LogMemoryStats()
        {
            Console.WriteLine("Statistiques mémoire:");
            Console.WriteLine($"  Nombre total de tranches: {slices.Length}");
            Console.WriteLine($"  Mémoire gérée utilisée: {GC.GetTotalMemory(false) / (1024 * 1024)}MB");
        }
        
        /// <summary>
        /// Effectue un nettoyage périodique de la mémoire
        /// </summary>
        private void PerformPeriodicCleanup(object state)
        {
            if (isDisposed) return;
            
            Console.WriteLine("Exécution du nettoyage mémoire périodique");
            
            // Décharger les tranches qui ne sont pas à proximité de celle actuellement affichée
            UnloadDistantSlices();
            
            // Forcer la collecte des déchets
            GC.Collect();
            
            LogMemoryStats();
        }
        
        /// <summary>
        /// Décharge les tranches éloignées de l'index courant pour libérer de la mémoire
        /// </summary>
        private void UnloadDistantSlices()
        {
            int windowStart = Math.Max(0, currentSliceIndex - PRELOAD_WINDOW_SIZE * 2);
            int windowEnd = Math.Min(slices.Length - 1, currentSliceIndex + PRELOAD_WINDOW_SIZE * 2);
            
            int unloadedCount = 0;
            
            for (int i = 0; i < slices.Length; i++)
            {
                if (i < windowStart || i > windowEnd)
                {
                    slices[i].UnloadPixelData();
                    DicomImageProcessor.ReleaseSlice(slices[i]);
                    unloadedCount++;
                }
            }
            
            Console.WriteLine($"Déchargé {unloadedCount} tranches éloignées de la mémoire");
        }

        public DicomMetadata GetSlice(int sliceIndex)
        {
            if (sliceIndex < 0 || sliceIndex >= slices.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(sliceIndex));
            }
            
            return slices[sliceIndex];
        }

        public Bitmap GetCurrentSliceImage(int windowWidth = -1, int windowCenter = -1)
        {
            if (isDisposed) throw new ObjectDisposedException(nameof(DicomDisplayManager));
            
            if (slices.Length == 0 || currentSliceIndex < 0 || currentSliceIndex >= slices.Length)
            {
                return null;
            }
            
            return DicomImageProcessor.ConvertToBitmap(slices[currentSliceIndex], windowWidth, windowCenter);
        }

        public Bitmap GetGlobalViewImage()
        {
            if (isDisposed) throw new ObjectDisposedException(nameof(DicomDisplayManager));
            
            return DicomImageProcessor.ConvertToBitmap(globalView);
        }

        public void SetSliceIndex(int index)
        {
            if (isDisposed) return;
            
            if (index >= 0 && index < slices.Length)
            {
                lock (loadLock)
                {
                    // Si on se déplace de plus de quelques tranches, précharger de nouvelles tranches
                    if (Math.Abs(index - currentSliceIndex) > 2)
                    {
                        PreloadSlicesAroundIndex(index);
                    }
                    
                    currentSliceIndex = index;
                }
            }
        }
        
        /// <summary>
        /// Précharge les tranches autour d'un index spécifique en mémoire GPU
        /// </summary>
        private void PreloadSlicesAroundIndex(int centerIndex)
        {
            if (isDisposed) return;
            
            int startIndex = Math.Max(0, centerIndex - PRELOAD_WINDOW_SIZE / 2);
            int endIndex = Math.Min(slices.Length - 1, centerIndex + PRELOAD_WINDOW_SIZE / 2);
            int count = endIndex - startIndex + 1;
            
            Console.WriteLine($"Préchargement des tranches {startIndex} à {endIndex}");
            
            // Extraire les tranches à précharger
            var slicesToPreload = new DicomMetadata[count];
            for (int i = 0; i < count; i++)
            {
                slicesToPreload[i] = slices[startIndex + i];
            }
            
            // Précharger les tranches en mémoire GPU
            DicomImageProcessor.PreloadSlices(slicesToPreload);
        }

        public int GetCurrentSliceIndex() => currentSliceIndex;
        public int GetTotalSlices() => slices.Length;
        
        /// <summary>
        /// Libère les ressources utilisées par le gestionnaire d'affichage
        /// </summary>
        public void Dispose()
        {
            if (isDisposed) return;
            
            Console.WriteLine("Libération des ressources du gestionnaire d'affichage DICOM");
            
            isDisposed = true;
            
            // Arrêter le minuteur de nettoyage
            cleanupTimer?.Dispose();
            cleanupTimer = null;
            
            // Décharger toutes les tranches
            if (slices != null)
            {
                foreach (var slice in slices)
                {
                    slice.Dispose();
                }
            }
            
            // Forcer la collecte des déchets
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
