using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using DeepBridgeWindowsApp.DICOM;
using DeepBridgeWindowsApp.CUDA;

namespace DeepBridgeWindowsApp
{
    internal static class Program
    {
        [STAThread]
        static void Main()
        {
            // Créer une console pour voir les messages de débogage
            AllocConsole();

            Console.WriteLine("Initialisation du processeur CUDA...");

            // Initialiser le processeur CUDA une seule fois au démarrage
            var cudaProcessor = new CudaDicomProcessor();

            Application.SetHighDpiMode(HighDpiMode.SystemAware);
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }

        // Importer la fonction Windows pour créer une console
        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
        private static extern bool AllocConsole();
    }
}