using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WinForms = System.Windows.Forms;
using DeepBridgeWindowsApp.Dicom;
using DeepBridgeWindowsApp.DICOM;

namespace DeepBridgeWindowsApp
{
    public partial class MainForm : Form
    {
        private string currentDirectory;
        private readonly string defaultDirectory = @"Z:\dataset_chu\dataset_chu_nice_2020_2021\scan\SF103E8_10.241.3.232_20210118173228207_CT_SR\SF103E8_10.241.3.232_20210118173228207";
        //private readonly string defaultDirectory = @"C:\dataset_chu_nice_2020_2021\scan\SF103E8_10.241.3.232_20210118174910223_CT\SF103E8_10.241.3.232_20210118174910223";
        private Panel rightPanel;
        private ListView contentListView;
        private Button viewDicomButton;
        private TextBox directoryTextBox;
        private Label infoLabel;
        private TableLayoutPanel mainTableLayout;
        private PictureBox globalViewPictureBox;

        public MainForm()
        {
            InitializeComponents();
            currentDirectory = defaultDirectory;
            LoadDirectory(currentDirectory);
        }

        private void InitializeComponents()
        {
            // Form settings
            Text = "DICOM Viewer - Main";
            Size = new System.Drawing.Size(1000, 600);
            MinimumSize = new System.Drawing.Size(800, 400);  // Set minimum size to prevent layout issues

            // Create main table layout
            mainTableLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 2,
                Padding = new Padding(10),
            };

            // Configure row and column styles
            mainTableLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));  // Top controls
            mainTableLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));  // Main content

            // Top panel for directory controls
            var topPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                Margin = new Padding(0)
            };
            topPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 85));
            topPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 15));

            // Directory selection controls
            directoryTextBox = new TextBox
            {
                Dock = DockStyle.Fill,
                Text = defaultDirectory
            };

            var browseButton = new Button
            {
                Dock = DockStyle.Fill,
                Text = "Browse",
                Margin = new Padding(5, 0, 0, 0)
            };
            browseButton.Click += BrowseButton_Click;

            topPanel.Controls.Add(directoryTextBox, 0, 0);
            topPanel.Controls.Add(browseButton, 1, 0);

            // Content panel
            var contentPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                Margin = new Padding(0, 10, 0, 0)
            };
            contentPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 65));
            contentPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 35));

            // Content list view
            contentListView = new ListView
            {
                Dock = DockStyle.Fill,
                View = View.Details,
                FullRowSelect = true
            };
            contentListView.Columns.Add("Name", -2);
            contentListView.Columns.Add("Type", -2);
            contentListView.Columns.Add("DICOM Files", -2);
            contentListView.SelectedIndexChanged += ContentListView_SelectedIndexChanged;

            // Right panel
            rightPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BorderStyle = BorderStyle.FixedSingle,
                Margin = new Padding(10, 0, 0, 0)
            };

            // Info label in right panel
            infoLabel = new Label
            {
                Location = new System.Drawing.Point(10, 10),
                AutoSize = false,
                Dock = DockStyle.Top,
                Height = 150
            };
            rightPanel.Controls.Add(infoLabel);

            // View DICOM button
            viewDicomButton = new Button
            {
                Dock = DockStyle.Bottom,
                Height = 30,
                Text = "View DICOM Images",
                Enabled = false,
                Margin = new Padding(10)
            };
            viewDicomButton.Click += ViewDicomButton_Click;
            rightPanel.Controls.Add(viewDicomButton);

            // Add controls to panels
            contentPanel.Controls.Add(contentListView, 0, 0);
            contentPanel.Controls.Add(rightPanel, 1, 0);

            // Add panels to main layout
            mainTableLayout.Controls.Add(topPanel, 0, 0);
            mainTableLayout.Controls.Add(contentPanel, 0, 1);

            // Add main layout to form
            Controls.Add(mainTableLayout);

            // Add resize event handler
            Resize += MainForm_Resize;
        }

        private void MainForm_Resize(object sender, EventArgs e)
        {
            // Adjust column widths in ListView
            if (contentListView.Columns.Count > 0)
            {
                int totalWidth = contentListView.ClientSize.Width;
                contentListView.Columns[0].Width = (int)(totalWidth * 0.5);  // Name: 50%
                contentListView.Columns[1].Width = (int)(totalWidth * 0.25); // Type: 25%
                contentListView.Columns[2].Width = (int)(totalWidth * 0.25); // DICOM Files: 25%
            }
        }

        private void BrowseButton_Click(object sender, EventArgs e)
        {
            using (WinForms.FolderBrowserDialog folderDialog = new FolderBrowserDialog())
            {
                folderDialog.SelectedPath = currentDirectory;
                folderDialog.ShowNewFolderButton = false;
                if (folderDialog.ShowDialog() == DialogResult.OK)
                {
                    currentDirectory = folderDialog.SelectedPath;
                    directoryTextBox.Text = currentDirectory;
                    LoadDirectory(currentDirectory);
                }
            }
        }

        private void LoadDirectory(string path)
        {
            contentListView.Items.Clear();
            infoLabel.Text = string.Empty;
            viewDicomButton.Enabled = false;

            // Get directories that directly contain .dcm files (no recursion)
            var directories = Directory.GetDirectories(path)
                .Where(dir => Directory.GetFiles(dir, "*.dcm").Length > 0);

            // Add directories with DICOM files to the list
            foreach (var dir in directories)
            {
                var dirInfo = new DirectoryInfo(dir);
                var dicomCount = Directory.GetFiles(dir, "*.dcm").Length;

                if (dicomCount <= 10)
                    continue;

                var item = new ListViewItem(new[]
                {
                    dirInfo.Name,
                    "Folder",
                    dicomCount.ToString()
                });
                item.Tag = dir;
                contentListView.Items.Add(item);
            }

            // Check if current directory has DICOM files
            var currentDirDicomFiles = Directory.GetFiles(path, "*.dcm");
            if (currentDirDicomFiles.Length > 0)
            {
                ShowDicomInfo(path);
            }
        }

        private void ContentListView_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (contentListView.SelectedItems.Count == 0)
                return;

            var selectedPath = contentListView.SelectedItems[0].Tag.ToString();
            ShowDicomInfo(selectedPath);
        }

        private DicomDisplayManager currentDisplayManager;
        
        private void ShowDicomInfo(string path)
        {
            var dicomFiles = Directory.GetFiles(path, "*.dcm", SearchOption.TopDirectoryOnly);
            if (dicomFiles.Length == 0)
                return;

            // Clean up any previous resources
            CleanupCurrentResources();
            
            // Report current memory usage
            Console.WriteLine($"Mémoire avant chargement: {GC.GetTotalMemory(true) / (1024 * 1024)} MB");
            
            // Create new reader and display manager
            var reader = new DicomReader(path);
            reader.LoadGlobalView();
            currentDisplayManager = new DicomDisplayManager(reader);

            // Basic info display - you can expand this to show more DICOM metadata
            infoLabel.Text = $"{Path.GetFileName(path)}\n" +
                            $"Number of DICOM files: {dicomFiles.Length}\n" +
                            $"Total size: {GetDirectorySize(path) / 1024.0 / 1024.0:F2} MB\n\n" +
                            $"Patient ID: {currentDisplayManager.globalView.PatientID}\n" +
                            $"Patient Name: {currentDisplayManager.globalView.PatientName}\n" +
                            $"Patient Sex: {currentDisplayManager.globalView.PatientSex}\n" +
                            $"Modality: {currentDisplayManager.globalView.Modality}\n" +
                            $"Resolution: {currentDisplayManager.globalView.Rows} x {currentDisplayManager.globalView.Columns}";

            globalViewPictureBox?.Dispose();
            globalViewPictureBox = new PictureBox
            {
                Dock = DockStyle.Fill,
                SizeMode = PictureBoxSizeMode.Zoom
            };
            globalViewPictureBox.Image = currentDisplayManager.GetGlobalViewImage();
            rightPanel.Controls.Add(globalViewPictureBox);

            viewDicomButton.Enabled = true;
            viewDicomButton.Tag = path;
            
            // Report memory usage after loading
            Console.WriteLine($"Mémoire après chargement: {GC.GetTotalMemory(false) / (1024 * 1024)} MB");
        }
        
        private void CleanupCurrentResources()
        {
            // Dispose of the current display manager
            if (currentDisplayManager != null)
            {
                currentDisplayManager.Dispose();
                currentDisplayManager = null;
            }
            
            // Force garbage collection
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        private long GetDirectorySize(string path)
        {
            return Directory.GetFiles(path, "*.dcm")
                           .Sum(file => new FileInfo(file).Length);
        }

        private void ViewDicomButton_Click(object sender, EventArgs e)
        {
            var path = viewDicomButton.Tag.ToString();
            
            // Clean up current resources before loading a large dataset
            CleanupCurrentResources();
            
            // Force a full garbage collection before loading new files
            GC.Collect();
            GC.WaitForPendingFinalizers();
            
            Console.WriteLine($"Mémoire avant chargement complet: {GC.GetTotalMemory(true) / (1024 * 1024)} MB");
            
            // Create a new reader and load all files
            var reader = new DicomReader(path);
            reader.LoadAllFiles();  // Now loads all files when viewing
            
            // Open viewer form
            using (var viewerForm = new DicomViewerForm(reader))
            {
                viewerForm.ShowDialog();
            }
            
            // Clean up and collect garbage after closing viewer
            reader.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            
            Console.WriteLine($"Mémoire après fermeture de la visionneuse: {GC.GetTotalMemory(true) / (1024 * 1024)} MB");
        }
        
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Clean up designer-generated components
                if (components != null)
                {
                    components.Dispose();
                }
                
                // Clean up resources
                CleanupCurrentResources();
                globalViewPictureBox?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
