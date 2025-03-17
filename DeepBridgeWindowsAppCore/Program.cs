using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using DeepBridgeWindowsApp.DICOM;
using DeepBridgeWindowsApp.CUDA;
using DeepBridgeWindowsApp.Dicom;

namespace DeepBridgeWindowsApp
{
    internal static class Program
    {
        // Référence statique pour s'assurer que le processeur CUDA reste en vie
        // pendant toute la durée de l'application
        private static CudaBatchProcessor cudaBatchProcessor;
        
        [STAThread]
        static void Main()
        {
            // Créer une console pour voir les messages de débogage
            AllocConsole();

            Console.WriteLine("Initialisation du processeur CUDA...");

            // Initialiser le processeur CUDA une seule fois au démarrage
            // Cette initialisation est maintenant faite avec CudaBatchProcessor 
            // pour une meilleure gestion de la mémoire GPU
            cudaBatchProcessor = new CudaBatchProcessor();
            
            // Afficher des informations sur la mémoire système
            DisplayMemoryInfo();

            // Configuration de l'application
            Application.SetHighDpiMode(HighDpiMode.SystemAware);
            Application.SetCompatibleTextRenderingDefault(false);
            
            // Enregistrer un gestionnaire d'événement pour nettoyer les ressources GPU à la fermeture
            Application.ApplicationExit += (sender, e) => {
                Console.WriteLine("Nettoyage des ressources GPU...");
                DicomImageProcessor.Cleanup();
                cudaBatchProcessor.Dispose();
            };
            
            // Démarrer l'application
            Application.Run(new MainForm());
        }
        
        /// <summary>
        /// Affiche des informations sur la mémoire système
        /// </summary>
        private static void DisplayMemoryInfo()
        {
            Console.WriteLine("Informations mémoire système:");
            Console.WriteLine($"Mémoire totale RAM: {GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024 * 1024 * 1024.0):F2}GB");
            Console.WriteLine($"Mémoire maximum .NET: {GC.MaxGeneration + 1} générations");
        }

        // Importer la fonction Windows pour créer une console
        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
        private static extern bool AllocConsole();
    }
}