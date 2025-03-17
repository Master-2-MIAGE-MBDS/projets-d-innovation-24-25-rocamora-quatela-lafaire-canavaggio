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
            // Cr�er une console pour voir les messages de d�bogage
            AllocConsole();

            Console.WriteLine("Initialisation du processeur CUDA...");

            // Initialiser le processeur CUDA une seule fois au d�marrage
            var cudaProcessor = new CudaDicomProcessor();

            Application.SetHighDpiMode(HighDpiMode.SystemAware);
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }

        // Importer la fonction Windows pour cr�er une console
        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
        private static extern bool AllocConsole();
    }
}