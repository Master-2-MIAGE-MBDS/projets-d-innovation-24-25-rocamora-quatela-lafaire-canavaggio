using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Printing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using DeepBridgeWindowsApp.Dicom;
using OpenTK.GLControl;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;

namespace DeepBridgeWindowsApp
{
    public partial class RenderDicomForm : Form
    {
        // Composants principaux
        private Dicom3D render;
        private readonly DicomMetadata dicom;
        private readonly DicomDisplayManager ddm;
        private GLControl gl;
        private ProgressBar progressBar;
        private Label progressLabel;
        private Label controlsHelpLabel;
        private Label controlsLabel;

        // Propriétés de la caméra
        private Vector3 cameraPosition = new Vector3(0, 0, 3f);
        private Vector3 cameraTarget = Vector3.Zero;
        private Vector3 cameraUp = Vector3.UnitY;
        private float rotationX = 0;
        private float rotationY = 0;
        private float zoom = 3.0f;

        // Contrôles de la souris
        private Point lastMousePos;
        private bool isMouseDown = false;

        // Contrôles clavier
        private readonly float moveSpeed = 0.1f;
        private readonly HashSet<Keys> pressedKeys = new HashSet<Keys>();
        private System.Windows.Forms.Timer moveTimer;

        // Contrôles des couches
        private TrackBar frontClipTrackBar;
        private TrackBar backClipTrackBar;
        private Label frontClipLabel;
        private Label backClipLabel;

        // Slice Button
        private Button sliceButton;
        private PictureBox slicePreview;
        private NumericUpDown slicePosition;
        private CheckBox checkBox;

        // Slice indicator
        private int[] sliceIndicatorVBO;
        private readonly float[] sliceIndicatorVertices = {
            // Front face vertices - making it slightly larger for better visibility
            0f, 0.6f, 0.6f,    // Top-right
            0f, -0.6f, 0.6f,   // Bottom-right
            0f, -0.6f, -0.6f,  // Bottom-left
            0f, 0.6f, -0.6f,   // Top-left
        };
        private int sliceWidth;

        // Shaders
        private int shaderProgram;
        private int ColorShaderProgram;
        private const string vertexShaderSource = @"
        #version 330 core
        layout(location = 0) in vec3 aPosition;
        layout(location = 1) in vec3 aColor;
        out vec3 vertexColor;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main()
        {
            gl_Position = projection * view * model * vec4(aPosition, 1.0);
            vertexColor = aColor;
        }";

        private const string fragmentShaderSource = @"
        #version 330 core
        in vec3 vertexColor;
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(vertexColor, 1.0);
        }";

        private const string ColorVertexShader = @"#version 330 core
        layout(location = 0) in vec3 aPosition;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPosition, 1.0);
        }";

        private const string ColorFragmentShader = @"#version 330 core
        uniform vec3 color;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(color, 1.0);
        }";

        public RenderDicomForm(DicomDisplayManager ddm)
        {
            this.ddm = ddm;
            this.dicom = this.ddm.globalView;
            this.sliceWidth = ddm.GetSlice(0).Columns;
            InitializeComponents();
            InitializeKeyboardControls();
        }

        private void InitializeComponents()
        {
            this.Size = new Size(1424, 768);
            this.Text = "DICOM Render";

            InitializeLeftPanel();
            InitializeGLControl();
            InitializeProgressBar();
            this.Controls.Add(gl);
        }

        private void InitializeLeftPanel()
        {
            var leftPanel = new Panel
            {
                Dock = DockStyle.Left,
                Width = 200,
                Padding = new Padding(5, 5, 5, 10),
                BackColor = Color.FromArgb(40, 40, 40)
            };

            // Créer un FlowLayoutPanel pour organiser les contrôles verticalement
            var flowPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                //AutoSize = true,
                Width = 200,
                WrapContents = false
            };

            // Label des contrôles
            controlsLabel = new Label
            {
                AutoSize = true,
                ForeColor = Color.White,
                Font = new Font(Font.FontFamily, 9),
                Text = "Contrôles:\n\n" +
                       "ZQSD/WASD :\nDéplacement\n\n" +
                       "E/C :\nMonter/Descendre\n\n" +
                       "Souris :\nRotation\n\n" +
                       "Molette :\nZoom\n\n" +
                       "R :\nRéinitialiser la vue\n\n"
            };

            // Panel pour les contrôles de découpage
            var clipPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(40, 40, 40),
                Width = 200,
                //Margin = new Padding(5)
            };

            // Label pour le titre des trackbars
            var clipLabel = new Label
            {
                Text = "Contrôles de découpage:",
                ForeColor = Color.White,
                AutoSize = true,
                Location = new Point(5, 5)
            };

            frontClipTrackBar = new TrackBar
            {
                Minimum = 0,
                Maximum = ddm.GetTotalSlices() - 1,
                Value = 0,
                Location = new Point(10, 25),
                Width = 180
            };
            frontClipTrackBar.ValueChanged += ClipTrackBar_ValueChanged;

            backClipTrackBar = new TrackBar
            {
                Minimum = 0,
                Maximum = ddm.GetTotalSlices() - 1,
                Value = 0,
                Location = new Point(10, 65),
                Width = 180
            };
            backClipTrackBar.ValueChanged += ClipTrackBar_ValueChanged;

            // Labels pour les trackbars
            frontClipLabel = new Label
            {
                Text = "Couches avant: 0",
                ForeColor = Color.White,
                AutoSize = true,
                Location = new Point(10, 45)
            };

            backClipLabel = new Label
            {
                Text = "Couches arrière: 0",
                ForeColor = Color.White,
                AutoSize = true,
                Location = new Point(10, 85)
            };

            // Ajouter les contrôles au FlowLayoutPanel
            checkBox = new CheckBox
            {
                Text = "Show Extract Position",
                ForeColor = Color.White,
                AutoSize = true,
                Checked = true  // Enable by default for better visibility
            };
            checkBox.CheckedChanged += (s, e) => gl.Invalidate();

            slicePosition = new NumericUpDown
            {
                Minimum = 0,                  // First pixel row
                Maximum = sliceWidth - 1,     // Last pixel row
                Value = sliceWidth / 2,       // Start at middle
            };
            slicePosition.ValueChanged += (s, e) => gl.Invalidate();

            // Label to show slice position
            var slicePositionLabel = new Label
            {
                Text = "Slice Position",
                ForeColor = Color.White,
                AutoSize = true
            };

            // Add slice button
            sliceButton = new Button
            {
                Dock = DockStyle.Bottom,
                Text = "Extract Slice",
                AutoSize = true,
                ForeColor = Color.White
            };
            sliceButton.Click += SliceButton_Click;

            //clipPanel.Controls.AddRange(new Control[] {
            //    clipLabel,
            //    frontClipTrackBar,
            //    backClipTrackBar,
            //    frontClipLabel,
            //    backClipLabel,
            //});

            // Add preview PictureBox for the slice
            slicePreview = new PictureBox
            {
                SizeMode = PictureBoxSizeMode.Zoom,
                BorderStyle = BorderStyle.FixedSingle,
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                Visible = false  // Hide initially
            };
            
            // Add a button directly over the preview for quick access to annotation
            var annotateButton = new Button
            {
                Text = "Annotate Carotids",
                BackColor = Color.FromArgb(0, 120, 215),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Visible = false,
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                Size = new Size(120, 30)
            };
            
            // Position the button at the bottom of the preview
            slicePreview.Resize += (s, e) => {
                annotateButton.Location = new Point(
                    slicePreview.Location.X + (slicePreview.Width - annotateButton.Width) / 2,
                    slicePreview.Location.Y + slicePreview.Height + 5
                );
            };
            
            // Show/hide the button with the preview
            slicePreview.VisibleChanged += (s, e) => {
                annotateButton.Visible = slicePreview.Visible;
            };
            
            // Handle the click event
            annotateButton.Click += (s, e) => {
                if (slicePreview.Image != null)
                {
                    SaveSlice((Bitmap)slicePreview.Image);
                }
            };

            this.Controls.Add(slicePreview);
            this.Controls.Add(annotateButton);

            // Ajouter les contrôles au FlowLayoutPanel
            flowPanel.Controls.Add(controlsLabel);
            //flowPanel.Controls.Add(clipPanel);
            //flowPanel.Controls.Add(sliceButton);
            flowPanel.Controls.Add(checkBox);
            flowPanel.Controls.Add(slicePositionLabel);
            flowPanel.Controls.Add(slicePosition);

            // Ajouter le FlowLayoutPanel au panel gauche
            leftPanel.Controls.Add(flowPanel);
            leftPanel.Controls.Add(sliceButton);
            this.Controls.Add(leftPanel);
        }

        private void InitializeGLControl()
        {
            try
            {
                Console.WriteLine("Creating GLControl...");
                
                // In OpenTK 4.x, GLControl doesn't use GraphicsMode constructor parameters
                // Instead, it uses the default setup which should work for most use cases
                gl = new GLControl()
                { 
                    Dock = DockStyle.Fill 
                };
                
                // Register event handlers
                gl.Resize += GLControl_Resize;
                gl.MouseDown += GL_MouseDown;
                gl.MouseUp += GL_MouseUp;
                gl.MouseMove += GL_MouseMove;
                gl.MouseWheel += GL_MouseWheel;
                
                gl.Focus();
                Console.WriteLine("GLControl created successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error creating GLControl: {ex}");
                throw new ApplicationException("Failed to create OpenGL control", ex);
            }
        }

        private void InitializeProgressBar()
        {
            progressBar = new ProgressBar
            {
                Width = 300,
                Height = 23,
                Style = ProgressBarStyle.Continuous,
                Visible = false
            };

            progressLabel = new Label
            {
                AutoSize = true,
                Width = 300,
                TextAlign = ContentAlignment.MiddleCenter,
                Visible = false
            };

            progressBar.Location = new Point(
                (this.ClientSize.Width - progressBar.Width) / 2,
                (this.ClientSize.Height - progressBar.Height) / 2
            );
            progressLabel.Location = new Point(
                (this.ClientSize.Width - progressLabel.Width) / 2,
                progressBar.Location.Y - 25
            );

            this.Controls.Add(progressBar);
            this.Controls.Add(progressLabel);
        }

        private void InitializeKeyboardControls()
        {
            this.KeyPreview = true;
            this.KeyDown += RenderDicomForm_KeyDown;
            this.KeyUp += RenderDicomForm_KeyUp;
            this.Activated += (s, e) => gl?.Focus();

            moveTimer = new System.Windows.Forms.Timer
            {
                Interval = 16 // ~60 FPS
            };
            moveTimer.Tick += MoveTimer_Tick;
            moveTimer.Start();
        }


        private void ClipTrackBar_ValueChanged(object sender, EventArgs e)
        {
            // Vérifier que les valeurs sont valides
            if (frontClipTrackBar.Value + backClipTrackBar.Value >= ddm.GetTotalSlices())
            {
                if (sender == frontClipTrackBar)
                    frontClipTrackBar.Value = ddm.GetTotalSlices() - 1 - backClipTrackBar.Value;
                else
                    backClipTrackBar.Value = ddm.GetTotalSlices() - 1 - frontClipTrackBar.Value;
            }

            controlsLabel.Text = $"Contrôles:\n\n" +
                                 "ZQSD/WASD :\nDéplacement\n\n" +
                                 "E/C :\nMonter/Descendre\n\n" +
                                 "Souris :\nRotation\n\n" +
                                 "Molette :\nZoom\n\n" +
                                 "R :\nRéinitialiser la vue\n\n" +
                                 $"Couches avant: {frontClipTrackBar.Value}\n" +
                                 $"Couches arrière: {backClipTrackBar.Value}";

            render.SetClipPlanes(frontClipTrackBar.Value, backClipTrackBar.Value);
            gl.Invalidate();
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            this.Shown += RenderDicomForm_Load;
        }

        private Bitmap RotateImage(Bitmap original)
        {
            if (original == null)
                return null;
                
            if (original.Width < 1 || original.Height < 1)
            {
                Console.WriteLine("Invalid bitmap dimensions for rotation");
                return null;
            }
                
            try
            {
                // Use a safer approach to clone the bitmap
                Bitmap clone = null;
                try
                {
                    // Lock the original bitmap to ensure it's valid
                    System.Drawing.Imaging.BitmapData bmpData = original.LockBits(
                        new Rectangle(0, 0, original.Width, original.Height),
                        System.Drawing.Imaging.ImageLockMode.ReadOnly,
                        original.PixelFormat);
                        
                    // Unlock immediately - this just verifies the bitmap is accessible
                    original.UnlockBits(bmpData);
                    
                    // Create a new bitmap with swapped dimensions
                    clone = new Bitmap(original.Height, original.Width, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error verifying bitmap for rotation: {ex.Message}");
                    return null;
                }
                
                // Use a graphics object to rotate and draw the image
                using (Graphics g = Graphics.FromImage(clone))
                {
                    // Set high quality rendering
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                    
                    // Clear background
                    g.Clear(Color.Black);
                    
                    // Apply rotation transform
                    g.TranslateTransform(0, original.Width);
                    g.RotateTransform(-90);
                    
                    // Draw the image with explicit rectangle to ensure bounds are correct
                    g.DrawImage(original, 
                        new Rectangle(0, 0, original.Width, original.Height),
                        0, 0, original.Width, original.Height,
                        GraphicsUnit.Pixel);
                }
                
                return clone;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error rotating image: {ex}");
                return null;
            }
        }

        private void SaveSlice(Bitmap slice)
        {
            if (slice == null)
            {
                MessageBox.Show("Error: No slice image available to annotate", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            
            try
            {
                Console.WriteLine($"SaveSlice called with bitmap: {slice.Width}x{slice.Height}, PixelFormat: {slice.PixelFormat}");
                
                // Open the annotation form before saving - let the form create its own copy
                using (var annotationForm = new SliceAnnotationForm(slice))
                {
                    if (annotationForm.ShowDialog() == DialogResult.OK)
                    {
                        try
                        {
                            // Get the annotated bitmap from the form
                            Bitmap annotatedSlice = annotationForm.AnnotatedImage;
                            
                            if (annotatedSlice == null)
                            {
                                MessageBox.Show("Could not get annotated image", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                                return;
                            }
                            
                            using (SaveFileDialog saveDialog = new SaveFileDialog())
                            {
                                saveDialog.Filter = "PNG Image|*.png|JPEG Image|*.jpg|Bitmap Image|*.bmp";
                                saveDialog.Title = "Save Annotated Slice Image";
                                saveDialog.DefaultExt = "png";

                                if (saveDialog.ShowDialog() == DialogResult.OK)
                                {
                                    try
                                    {
                                        // Use lossless PNG format for best quality
                                        annotatedSlice.Save(saveDialog.FileName, ImageFormat.Png);
                                        MessageBox.Show("Slice with carotid annotations saved successfully!", "Save Complete", MessageBoxButtons.OK, MessageBoxIcon.Information);
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Error saving annotated slice: {ex}");
                                        MessageBox.Show($"Error saving slice: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                                    }
                                    finally
                                    {
                                        // Clean up the bitmap
                                        if (annotatedSlice != null)
                                        {
                                            annotatedSlice.Dispose();
                                            annotatedSlice = null;
                                        }
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error in annotation process: {ex}");
                            MessageBox.Show($"Error processing annotated image: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error opening annotation form: {ex}");
                MessageBox.Show($"Error opening annotation form: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                // Force garbage collection
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        private async void SliceButton_Click(object sender, EventArgs e)
        {
            if (render != null)
            {
                try
                {
                    // Disable button while processing
                    sliceButton.Enabled = false;
                    sliceButton.Text = "Processing...";
                    Cursor = Cursors.WaitCursor;

                    // Report memory usage before extraction
                    long beforeMem = GC.GetTotalMemory(true) / (1024 * 1024);
                    Console.WriteLine($"Memory before slice extraction: {beforeMem} MB");

                    // Convert pixel position to normalized coordinate for ExtractSlice
                    float normalizedPos = ((float)slicePosition.Value / (sliceWidth - 1)) - 0.5f;

                    Console.WriteLine($"Extracting slice at pixel row: {slicePosition.Value} (normalized: {normalizedPos})");
                    
                    // Use our optimized memory-efficient extraction
                    Bitmap slice = null;
                    try 
                    {
                        slice = await Task.Run(() => 
                        {
                            // Force a garbage collection before the operation
                            GC.Collect();
                            GC.WaitForPendingFinalizers();
                            
                            return render.ExtractSlice(normalizedPos);
                        });
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Error extracting slice: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }

                    if (slice != null)
                    {
                        try
                        {
                            // If there's an existing image, dispose it to free memory
                            if (slicePreview.Image != null)
                            {
                                var oldImage = slicePreview.Image;
                                slicePreview.Image = null;
                                oldImage.Dispose();
                            }

                            // Rotate the slice - make sure we dispose the original after
                            Bitmap rotatedSlice = null;
                            try
                            {
                                rotatedSlice = RotateImage(slice);
                            }
                            finally
                            {
                                // Dispose the original slice to free memory
                                if (slice != null)
                                {
                                    slice.Dispose();
                                    slice = null;
                                }
                            }

                            if (rotatedSlice != null)
                            {
                                // Resize preview area
                                slicePreview.Width = rotatedSlice.Width + 10;
                                slicePreview.Height = rotatedSlice.Height + 10;
                                slicePreview.Location = new Point(
                                    this.ClientSize.Width - slicePreview.Width - 10,
                                    10
                                );

                                // Set the new image and show the preview
                                slicePreview.Image = rotatedSlice;
                                slicePreview.Visible = true;

                                // Remove any existing annotation buttons to avoid duplicates
                                foreach (Control c in this.Controls)
                                {
                                    if (c is Button btn && btn.Text == "Annotate Carotids")
                                    {
                                        this.Controls.Remove(c);
                                        btn.Dispose();
                                    }
                                }
                                
                                // Create a prominent annotation button that's VERY visible
                                Button annotateButton = new Button
                                {
                                    Text = "Annotate Carotids",
                                    BackColor = Color.FromArgb(0, 120, 215),
                                    ForeColor = Color.White,
                                    Font = new Font("Arial", 12, FontStyle.Bold),
                                    Width = 180,
                                    Height = 50,
                                    FlatStyle = FlatStyle.Flat,
                                    Visible = true,
                                    Cursor = Cursors.Hand,
                                    Name = "annotateCarotidsButton"  // Give it a name for later reference
                                };
                                
                                // Add a distinctive border to make it stand out
                                annotateButton.FlatAppearance.BorderSize = 3;
                                annotateButton.FlatAppearance.BorderColor = Color.Yellow;
                                
                                // Position the button below the preview
                                annotateButton.Location = new Point(
                                    slicePreview.Location.X + (slicePreview.Width - annotateButton.Width) / 2,
                                    slicePreview.Location.Y + slicePreview.Height + 10
                                );
                                
                                // Handle click to open annotation dialog
                                annotateButton.Click += (s, args) => {
                                    // Simplified approach - don't create a copy here, pass reference directly
                                    // The SaveSlice method will handle the copy internally
                                    try
                                    {
                                        if (slicePreview.Image != null)
                                        {
                                            SaveSlice((Bitmap)slicePreview.Image);
                                        }
                                        else
                                        {
                                            MessageBox.Show("No image available to annotate", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Error in annotation button click: {ex}");
                                        MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                                    }
                                };
                                
                                // Add tooltip
                                var toolTip = new ToolTip();
                                toolTip.SetToolTip(annotateButton, "Click to mark carotid positions for AI training");
                                
                                // Add button to form
                                this.Controls.Add(annotateButton);
                                annotateButton.BringToFront();
                                
                                // Also create a context menu for right-click options
                                var saveMenu = new ContextMenuStrip();
                                
                                // Add option to annotate the slice for carotid positions
                                var annotateItem = new ToolStripMenuItem("Annotate and Save...");
                                annotateItem.Click += (s, args) => {
                                    try
                                    {
                                        if (slicePreview.Image != null)
                                        {
                                            SaveSlice((Bitmap)slicePreview.Image);
                                        }
                                        else
                                        {
                                            MessageBox.Show("No image available to annotate", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Error in context menu click: {ex}");
                                        MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                                    }
                                };
                                annotateItem.Font = new Font(annotateItem.Font, FontStyle.Bold); // Make it stand out
                                saveMenu.Items.Add(annotateItem);
                                
                                // Also keep the original save option
                                var saveItem = new ToolStripMenuItem("Save Without Annotations...");
                                saveItem.Click += (s, args) => {
                                    try
                                    {
                                        // Direct save without annotation
                                        if (slicePreview.Image != null)
                                        {
                                            using (SaveFileDialog saveDialog = new SaveFileDialog())
                                            {
                                                saveDialog.Filter = "PNG Image|*.png|JPEG Image|*.jpg|Bitmap Image|*.bmp";
                                                saveDialog.Title = "Save Slice Image";
                                                saveDialog.DefaultExt = "png";

                                                if (saveDialog.ShowDialog() == DialogResult.OK)
                                                {
                                                    try
                                                    {
                                                        // Save directly using the image format based on extension
                                                        string ext = System.IO.Path.GetExtension(saveDialog.FileName).ToLower();
                                                        ImageFormat format = ImageFormat.Png; // Default to PNG
                                                        
                                                        if (ext == ".jpg" || ext == ".jpeg")
                                                            format = ImageFormat.Jpeg;
                                                        else if (ext == ".bmp")
                                                            format = ImageFormat.Bmp;
                                                            
                                                        slicePreview.Image.Save(saveDialog.FileName, format);
                                                        
                                                        MessageBox.Show("Slice saved successfully!", "Save Complete", 
                                                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                                                    }
                                                    catch (Exception ex)
                                                    {
                                                        Console.WriteLine($"Error saving slice: {ex}");
                                                        MessageBox.Show($"Error saving slice: {ex.Message}", "Error", 
                                                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            MessageBox.Show("No image available to save", "Error", 
                                                MessageBoxButtons.OK, MessageBoxIcon.Warning);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Error in save menu click: {ex}");
                                        MessageBox.Show($"Error: {ex.Message}", "Error", 
                                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                                    }
                                };
                                saveMenu.Items.Add(saveItem);
                                
                                slicePreview.ContextMenuStrip = saveMenu;
                                
                                // Show a notification to the user that the annotation option is available
                                MessageBox.Show(
                                    "Slice extracted successfully. Click 'Annotate Carotids' button to mark carotids for AI training.",
                                    "Slice Ready",
                                    MessageBoxButtons.OK,
                                    MessageBoxIcon.Information);
                                
                                // Force cleanup
                                GC.Collect();
                                GC.WaitForPendingFinalizers();
                                
                                // Report memory usage after extraction
                                long afterMem = GC.GetTotalMemory(true) / (1024 * 1024);
                                Console.WriteLine($"Memory after slice extraction: {afterMem} MB");
                                Console.WriteLine($"Memory change: {afterMem - beforeMem} MB");
                            }
                            else
                            {
                                MessageBox.Show("Failed to rotate slice image.");
                            }
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show($"Error processing slice: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                    else
                    {
                        MessageBox.Show("Failed to extract slice. No points found at the specified position.");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error extracting slice: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    // Re-enable button and restore cursor
                    sliceButton.Enabled = true;
                    sliceButton.Text = "Extract Slice";
                    Cursor = Cursors.Default;
                }
            }
        }

        // Memory usage monitoring label
        private Label memoryUsageLabel;
        
        private async void RenderDicomForm_Load(object sender, EventArgs e)
        {
            Console.WriteLine("Loading 3D render...");
            ShowProgress(true);
            
            try
            {
                // Add memory usage monitoring with 8GB limit warning
                memoryUsageLabel = new Label
                {
                    AutoSize = true,
                    BackColor = Color.Black,
                    ForeColor = Color.LimeGreen,
                    Text = "Memory: Initializing...",
                    Location = new Point(10, 10),
                    Font = new Font("Arial", 9, FontStyle.Bold)
                };
                this.Controls.Add(memoryUsageLabel);
                
                // Start memory monitoring timer with less frequent updates
                var memoryTimer = new System.Windows.Forms.Timer
                {
                    Interval = 5000  // Update less frequently (every 5 seconds)
                };
                memoryTimer.Tick += (s, ev) =>
                {
                    try
                    {
                        // Get current memory usage
                        long usedMemory = GC.GetTotalMemory(false) / (1024 * 1024);
                        
                        // Set color based on memory usage (green->yellow->red as memory increases)
                        if (usedMemory > 6000) // Over 6GB is critical with 8GB limit
                        {
                            memoryUsageLabel.ForeColor = Color.Red;
                            memoryUsageLabel.Text = $"WARNING! Memory: {usedMemory} MB";
                            
                            // Force garbage collection when close to limit
                            GC.Collect(2, GCCollectionMode.Forced);
                            GC.WaitForPendingFinalizers();
                        }
                        else if (usedMemory > 4000) // Over 4GB is warning level
                        {
                            memoryUsageLabel.ForeColor = Color.Yellow;
                            memoryUsageLabel.Text = $"Memory: {usedMemory} MB";
                            
                            // Gentle collection when approaching high memory
                            GC.Collect(0, GCCollectionMode.Optimized);
                        }
                        else
                        {
                            memoryUsageLabel.ForeColor = Color.LimeGreen;
                            memoryUsageLabel.Text = $"Memory: {usedMemory} MB";
                        }
                    }
                    catch (Exception timerEx)
                    {
                        Console.WriteLine($"Error in memory timer: {timerEx.Message}");
                    }
                };
                memoryTimer.Start();

                // Force garbage collection before starting
                GC.Collect();
                GC.WaitForPendingFinalizers();
                
                long startMemory = GC.GetTotalMemory(true) / (1024 * 1024);
                Console.WriteLine($"Memory before 3D initialization: {startMemory} MB");
                
                bool openGlInitialized = false;
                
                try
                {
                    // First, initialize OpenGL synchronously to catch any immediate errors
                    Console.WriteLine("Initializing OpenGL...");
                    InitializeOpenGL();
                    openGlInitialized = true;
                    Console.WriteLine("OpenGL initialized successfully");
                }
                catch (Exception glEx)
                {
                    Console.WriteLine($"Error initializing OpenGL: {glEx}");
                    MessageBox.Show($"Error initializing OpenGL: {glEx.Message}", "OpenGL Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;  // Exit early
                }
                
                if (!openGlInitialized)
                {
                    MessageBox.Show("Failed to initialize OpenGL. The application will now exit.", "Critical Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;  // Exit early
                }
                
                gl.Focus();
                
                try
                {
                    // Create the 3D renderer with our ultra-lightweight approach
                    Console.WriteLine("Creating Dicom3D instance...");
                    this.render = new Dicom3D(this.ddm, UpdateProgress);
                    Console.WriteLine("Dicom3D instance created successfully");
                }
                catch (Exception renderEx)
                {
                    Console.WriteLine($"Error creating 3D renderer: {renderEx}");
                    MessageBox.Show($"Error creating 3D renderer: {renderEx.Message}", "Renderer Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;  // Exit early
                }
                
                try
                {
                    Console.WriteLine("Initializing shaders...");
                    InitializeShaders();
                    Console.WriteLine("Shaders initialized successfully");
                }
                catch (Exception shaderEx)
                {
                    Console.WriteLine($"Error initializing shaders: {shaderEx}");
                    MessageBox.Show($"Error initializing shaders: {shaderEx.Message}", "Shader Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;  // Exit early
                }
                
                try
                {
                    Console.WriteLine("Initializing renderer GL resources...");
                    render.InitializeGL();
                    Console.WriteLine("Renderer GL resources initialized successfully");
                }
                catch (Exception rendererGlEx)
                {
                    Console.WriteLine($"Error initializing renderer GL resources: {rendererGlEx}");
                    MessageBox.Show($"Error initializing renderer GL resources: {rendererGlEx.Message}", "Renderer GL Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;  // Exit early
                }
                
                // Setup paint handler
                try
                {
                    Console.WriteLine("Setting up GL paint handler...");
                    gl.Paint += GLControl_Paint;
                    Console.WriteLine("GL paint handler set up successfully");
                }
                catch (Exception paintEx)
                {
                    Console.WriteLine($"Error setting up paint handler: {paintEx}");
                }
                
                ShowProgress(false);
                gl.Invalidate();
                gl.Focus();
                
                // Force cleanup after initialization
                GC.Collect();
                GC.WaitForPendingFinalizers();
                
                long endMemory = GC.GetTotalMemory(true) / (1024 * 1024);
                Console.WriteLine($"Memory after 3D initialization: {endMemory} MB");
                Console.WriteLine($"Memory used by 3D renderer: {endMemory - startMemory} MB");
                
                // Update memory label
                memoryUsageLabel.Text = $"Memory: {endMemory} MB";
                
                // Bring the memory label to front so it's always visible
                memoryUsageLabel.BringToFront();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in RenderDicomForm_Load: {ex}");
                MessageBox.Show($"Error initializing 3D render: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                ShowProgress(false);
            }
        }

        private void InitializeOpenGL()
        {
            try
            {
                Console.WriteLine("Making OpenGL context current...");
                gl.MakeCurrent();
                
                // Print OpenGL information
                Console.WriteLine($"OpenGL version: {GL.GetString(StringName.Version)}");
                Console.WriteLine($"OpenGL vendor: {GL.GetString(StringName.Vendor)}");
                Console.WriteLine($"OpenGL renderer: {GL.GetString(StringName.Renderer)}");
                
                // Set clear color
                GL.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Black
                
                // Enable depth testing
                GL.Enable(EnableCap.DepthTest);
                
                // Safer texture initialization
                try
                {
                    GL.Enable(EnableCap.Texture2D);
                }
                catch (Exception texEx)
                {
                    Console.WriteLine($"Warning: Could not enable Texture2D: {texEx.Message}");
                    // Continue anyway - this might not be critical
                }
                
                // Check for OpenGL errors
                ErrorCode error = GL.GetError();
                if (error != ErrorCode.NoError)
                {
                    Console.WriteLine($"OpenGL error after initialization: {error}");
                }
                else
                {
                    Console.WriteLine("OpenGL initialized without errors");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in InitializeOpenGL: {ex}");
                throw;
            }
        }

        private void ShowProgress(bool visible)
        {
            progressBar.Visible = visible;
            progressLabel.Visible = visible;
            if (visible)
            {
                progressBar.Value = 0;
                progressBar.Maximum = 100;
            }
        }

        private void UpdateProgress(ProcessingProgress progress)
        {
            this.Invoke((MethodInvoker)delegate
            {
                progressBar.Value = (int)progress.Percentage;
                progressLabel.Text = $"{progress.CurrentStep} - {progress.CurrentValue} of {progress.TotalValue} slices ({progress.Percentage:F1}%)";
            });
        }

        #region Keyboard Controls

        private void RenderDicomForm_KeyDown(object sender, KeyEventArgs e)
        {
            pressedKeys.Add(e.KeyCode);
        }

        private void RenderDicomForm_KeyUp(object sender, KeyEventArgs e)
        {
            pressedKeys.Remove(e.KeyCode);
        }

        private void MoveTimer_Tick(object sender, EventArgs e)
        {
            bool moved = false;
            Vector3 viewDir = (cameraTarget - cameraPosition).Normalized();
            Vector3 right = Vector3.Cross(cameraUp, viewDir).Normalized();
            Vector3 up = Vector3.Cross(viewDir, right);

            // Déplacement avant/arrière (Z/S)
            if (pressedKeys.Contains(Keys.Z) || pressedKeys.Contains(Keys.W))
            {
                cameraPosition += viewDir * moveSpeed;
                cameraTarget += viewDir * moveSpeed;
                moved = true;
            }
            if (pressedKeys.Contains(Keys.S))
            {
                cameraPosition -= viewDir * moveSpeed;
                cameraTarget -= viewDir * moveSpeed;
                moved = true;
            }

            // Déplacement gauche/droite (Q/D)
            if (pressedKeys.Contains(Keys.Q) || pressedKeys.Contains(Keys.A))
            {
                cameraPosition += right * moveSpeed;
                cameraTarget += right * moveSpeed;
                moved = true;
            }
            if (pressedKeys.Contains(Keys.D))
            {
                cameraPosition -= right * moveSpeed;
                cameraTarget -= right * moveSpeed;
                moved = true;
            }

            // Déplacement haut/bas (E/C)
            if (pressedKeys.Contains(Keys.E))
            {
                cameraPosition += up * moveSpeed;
                cameraTarget += up * moveSpeed;
                moved = true;
            }
            if (pressedKeys.Contains(Keys.C))
            {
                cameraPosition -= up * moveSpeed;
                cameraTarget -= up * moveSpeed;
                moved = true;
            }

            // Reset position (R)
            if (pressedKeys.Contains(Keys.R))
            {
                ResetCamera();
                moved = true;
            }

            if (moved)
            {
                gl.Invalidate();
            }
        }

        private void ResetCamera()
        {
            cameraPosition = new Vector3(0, 0, 3f);
            cameraTarget = Vector3.Zero;
            cameraUp = Vector3.UnitY;
            rotationX = 0;
            rotationY = 0;
            zoom = 3.0f;
        }

        #endregion

        #region Mouse Controls

        private void GL_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                isMouseDown = true;
                lastMousePos = e.Location;
            }
        }

        private void GL_MouseUp(object sender, MouseEventArgs e)
        {
            isMouseDown = false;
        }

        private void GL_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isMouseDown) return;

            float deltaX = -(e.X - lastMousePos.X) * 0.5f;
            float deltaY = (e.Y - lastMousePos.Y) * 0.5f;

            Vector3 viewDir = (cameraTarget - cameraPosition).Normalized();
            Vector3 right = Vector3.Cross(cameraUp, viewDir).Normalized();
            Vector3 up = Vector3.Cross(viewDir, right);

            Quaternion rotX = Quaternion.FromAxisAngle(up, MathHelper.DegreesToRadians(deltaX));
            Quaternion rotY = Quaternion.FromAxisAngle(right, MathHelper.DegreesToRadians(deltaY));
            Quaternion finalRotation = rotX * rotY;

            cameraPosition = Vector3.Transform(cameraPosition - cameraTarget, finalRotation) + cameraTarget;
            cameraUp = Vector3.Transform(cameraUp, finalRotation);

            lastMousePos = e.Location;
            gl.Invalidate();
        }

        private void GL_MouseWheel(object sender, MouseEventArgs e)
        {
            float zoomFactor = 1.0f - (e.Delta * 0.001f);
            Vector3 zoomDir = cameraPosition - cameraTarget;
            cameraPosition = cameraTarget + zoomDir * zoomFactor;
            gl.Invalidate();
        }

        #endregion

        #region OpenGL Rendering

        private void InitializeShaders()
        {
            shaderProgram = CreateShaderProgram(vertexShaderSource, fragmentShaderSource);
            ColorShaderProgram = CreateShaderProgram(ColorVertexShader, ColorFragmentShader);
        }

        private int CreateShaderProgram(string vertexSource, string fragmentSource)
        {
            int vertexShader = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(vertexShader, vertexSource);
            GL.CompileShader(vertexShader);

            int fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(fragmentShader, fragmentSource);
            GL.CompileShader(fragmentShader);

            int program = GL.CreateProgram();
            GL.AttachShader(program, vertexShader);
            GL.AttachShader(program, fragmentShader);
            GL.LinkProgram(program);

            GL.DeleteShader(vertexShader);
            GL.DeleteShader(fragmentShader);

            return program;
        }

        private void GLControl_Paint(object sender, PaintEventArgs e)
        {
            // Check if GL context is valid and all GL resources are initialized
            if (gl == null || render == null || shaderProgram <= 0)
            {
                Console.WriteLine("Skipping paint - GL resources not fully initialized");
                return;
            }
            
            try
            {
                // Make OpenGL context current
                gl.MakeCurrent();
                
                // Set a solid black background
                GL.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                
                // Clear the buffers
                GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                
                // Use our main shader program
                GL.UseProgram(shaderProgram);
                
                // Check for any errors after setting shader program
                ErrorCode error = GL.GetError();
                if (error != ErrorCode.NoError)
                {
                    Console.WriteLine($"OpenGL error after setting shader program: {error}");
                    return;
                }
                
                // Calculate aspect ratio safely
                float aspect = 1.0f;  // Default to 1.0 if dimensions are invalid
                if (gl.ClientSize.Width > 0 && gl.ClientSize.Height > 0)
                {
                    aspect = (float)gl.ClientSize.Width / gl.ClientSize.Height;
                }
                
                // Create projection matrix
                Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(
                    MathHelper.PiOver4,  // 45 degrees field of view
                    aspect,              // Aspect ratio
                    0.1f,                // Near plane
                    100f                 // Far plane
                );
                
                // Create view matrix from camera position
                Matrix4 view = Matrix4.LookAt(
                    cameraPosition,      // Eye position
                    cameraTarget,        // Look target
                    cameraUp             // Up vector
                );
                
                // Create model matrix
                Matrix4 model = Matrix4.Identity;
                
                // Apply rotations
                model *= Matrix4.CreateRotationX(MathHelper.DegreesToRadians(rotationX));
                model *= Matrix4.CreateRotationY(MathHelper.DegreesToRadians(rotationY));
                
                // Apply scale - increase size for better visibility
                float scaleFactor = 1.25f;  // Reduced from 1.5f for performance
                model *= Matrix4.CreateScale(scaleFactor);
                
                // Apply translation
                Vector3 centerOffset = new Vector3(0f, 0f, 0f);
                model *= Matrix4.CreateTranslation(centerOffset);
                
                // Get uniform locations
                int modelLoc = GL.GetUniformLocation(shaderProgram, "model");
                int viewLoc = GL.GetUniformLocation(shaderProgram, "view");
                int projLoc = GL.GetUniformLocation(shaderProgram, "projection");
                
                // Set matrix uniforms
                GL.UniformMatrix4(modelLoc, false, ref model);
                GL.UniformMatrix4(viewLoc, false, ref view);
                GL.UniformMatrix4(projLoc, false, ref projection);
                
                // Check for any errors after setting uniforms
                error = GL.GetError();
                if (error != ErrorCode.NoError)
                {
                    Console.WriteLine($"OpenGL error after setting matrix uniforms: {error}");
                    return;
                }
                
                // Enable alpha blending for better point visibility
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                
                // Enable point smoothing/anti-aliasing if available
                GL.Enable(EnableCap.PointSmooth);
                
                // Render the DICOM 3D data
                render.Render(shaderProgram, model, view, projection);
                
                error = GL.GetError();
                if (error != ErrorCode.NoError)
                {
                    Console.WriteLine($"OpenGL error after rendering: {error}");
                    // Continue anyway - we can still try to draw other elements
                }
                
                // Draw additional helpers
                try
                {
                    DrawBoundingBox(model, view, projection);
                    DrawSliceIndicator(model, view, projection);
                }
                catch (Exception helperEx)
                {
                    Console.WriteLine($"Error drawing helpers: {helperEx.Message}");
                }
                
                // Swap the buffers to display the rendered image
                gl.SwapBuffers();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in GLControl_Paint: {ex}");
            }
        }

        private void DrawBoundingBox(Matrix4 model, Matrix4 view, Matrix4 projection)
        {
            // Save current OpenGL state
            bool depthTestEnabled = GL.IsEnabled(EnableCap.DepthTest);
            if (depthTestEnabled)
                GL.Disable(EnableCap.DepthTest);  // Make sure box is always visible
                
            GL.UseProgram(ColorShaderProgram);

            // Use the shader program for colored lines
            GL.UniformMatrix4(GL.GetUniformLocation(ColorShaderProgram, "model"), false, ref model);
            GL.UniformMatrix4(GL.GetUniformLocation(ColorShaderProgram, "view"), false, ref view);
            GL.UniformMatrix4(GL.GetUniformLocation(ColorShaderProgram, "projection"), false, ref projection);
            
            // Bright red color for better visibility
            GL.Uniform3(GL.GetUniformLocation(ColorShaderProgram, "color"), 1.0f, 0.0f, 0.0f);
            
            // Set line width to make it more visible
            float originalLineWidth;
            GL.GetFloat(GetPName.LineWidth, out originalLineWidth);
            GL.LineWidth(2.0f);  // Thicker lines for better visibility

            GL.Begin(PrimitiveType.Lines);

            // Bottom face
            GL.Vertex3(-0.5f, -0.5f, -0.5f); GL.Vertex3(0.5f, -0.5f, -0.5f);
            GL.Vertex3(0.5f, -0.5f, -0.5f); GL.Vertex3(0.5f, -0.5f, 0.5f);
            GL.Vertex3(0.5f, -0.5f, 0.5f); GL.Vertex3(-0.5f, -0.5f, 0.5f);
            GL.Vertex3(-0.5f, -0.5f, 0.5f); GL.Vertex3(-0.5f, -0.5f, -0.5f);

            // Top face
            GL.Vertex3(-0.5f, 0.5f, -0.5f); GL.Vertex3(0.5f, 0.5f, -0.5f);
            GL.Vertex3(0.5f, 0.5f, -0.5f); GL.Vertex3(0.5f, 0.5f, 0.5f);
            GL.Vertex3(0.5f, 0.5f, 0.5f); GL.Vertex3(-0.5f, 0.5f, 0.5f);
            GL.Vertex3(-0.5f, 0.5f, 0.5f); GL.Vertex3(-0.5f, 0.5f, -0.5f);

            // Vertical edges
            GL.Vertex3(-0.5f, -0.5f, -0.5f); GL.Vertex3(-0.5f, 0.5f, -0.5f);
            GL.Vertex3(0.5f, -0.5f, -0.5f); GL.Vertex3(0.5f, 0.5f, -0.5f);
            GL.Vertex3(0.5f, -0.5f, 0.5f); GL.Vertex3(0.5f, 0.5f, 0.5f);
            GL.Vertex3(-0.5f, -0.5f, 0.5f); GL.Vertex3(-0.5f, 0.5f, 0.5f);

            GL.End();
            
            // Restore original line width
            GL.LineWidth(originalLineWidth);
            
            // Restore depth testing if it was enabled
            if (depthTestEnabled)
                GL.Enable(EnableCap.DepthTest);
                
            GL.UseProgram(shaderProgram);
        }

        // Track last position to reduce logging
        private static int lastSlicePosition = -1;
        
        private void DrawSliceIndicator(Matrix4 model, Matrix4 view, Matrix4 projection)
        {
            // Always show the slice indicator when checkbox is checked
            if (!checkBox.Checked) return;
            
            // Log only when position changes to reduce console spam
            if (lastSlicePosition != (int)slicePosition.Value)
            {
                Console.WriteLine($"Drawing slice plane at position: {slicePosition.Value}");
                lastSlicePosition = (int)slicePosition.Value;
            }
            
            // Save current OpenGL state
            bool depthTestEnabled = GL.IsEnabled(EnableCap.DepthTest);
            bool blendEnabled = GL.IsEnabled(EnableCap.Blend);
            
            // Disable depth test so slice plane is always visible
            GL.Disable(EnableCap.DepthTest);
            
            // Enable blending for semi-transparency
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            
            GL.UseProgram(ColorShaderProgram);

            // Convert pixel position to normalized coordinate for rendering
            float normalizedPos = ((float)slicePosition.Value / (sliceWidth - 1)) - 0.5f;

            // Create a separate model matrix for the slice plane that only includes the translation on X axis
            Matrix4 sliceModel = Matrix4.Identity;
            sliceModel *= Matrix4.CreateTranslation(normalizedPos, 0, 0);
            
            // Multiply by the original model's rotation component only
            // Extract rotation from model matrix by zeroing out translation
            Matrix4 rotationOnly = model;
            rotationOnly.Row3.X = 0;
            rotationOnly.Row3.Y = 0;
            rotationOnly.Row3.Z = 0;
            
            sliceModel *= rotationOnly;

            // Set uniforms
            GL.UniformMatrix4(GL.GetUniformLocation(ColorShaderProgram, "model"), false, ref sliceModel);
            GL.UniformMatrix4(GL.GetUniformLocation(ColorShaderProgram, "view"), false, ref view);
            GL.UniformMatrix4(GL.GetUniformLocation(ColorShaderProgram, "projection"), false, ref projection);

            // Use a very thick line width for better visibility
            float originalLineWidth;
            GL.GetFloat(GetPName.LineWidth, out originalLineWidth);
            GL.LineWidth(8.0f);  // Extra thick lines
            
            // Draw crosshairs at the slice position - makes it much more visible
            GL.Begin(PrimitiveType.Lines);
            
            // Vertical line (red)
            GL.Uniform3(GL.GetUniformLocation(ColorShaderProgram, "color"), 1.0f, 0.0f, 0.0f); // Red
            GL.Vertex3(0.0f, -0.7f, 0.0f);
            GL.Vertex3(0.0f, 0.7f, 0.0f);
            
            // Horizontal line (green)
            GL.Uniform3(GL.GetUniformLocation(ColorShaderProgram, "color"), 0.0f, 1.0f, 0.0f); // Green
            GL.Vertex3(0.0f, 0.0f, -0.7f);
            GL.Vertex3(0.0f, 0.0f, 0.7f);
            
            GL.End();
            
            // Draw slice plane as filled quad with semi-transparency
            GL.Uniform3(GL.GetUniformLocation(ColorShaderProgram, "color"), 1.0f, 1.0f, 0.0f); // Yellow
            
            // Make the slice plane MUCH more visible by making it completely opaque
            GL.BlendFunc(BlendingFactor.One, BlendingFactor.Zero);
            
            GL.Begin(PrimitiveType.Quads);
            // Make the plane larger for better visibility
            GL.Vertex3(0.0f, 0.8f, 0.8f);    // Top-right
            GL.Vertex3(0.0f, -0.8f, 0.8f);   // Bottom-right
            GL.Vertex3(0.0f, -0.8f, -0.8f);  // Bottom-left
            GL.Vertex3(0.0f, 0.8f, -0.8f);   // Top-left
            GL.End();
            
            // Restore normal blending
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            
            // Draw outline for better visibility
            GL.Uniform3(GL.GetUniformLocation(ColorShaderProgram, "color"), 1.0f, 0.5f, 0.0f); // Orange
            GL.LineWidth(10.0f); // Make outline thicker
            GL.Begin(PrimitiveType.LineLoop);
            GL.Vertex3(0.0f, 0.8f, 0.8f);    // Top-right
            GL.Vertex3(0.0f, -0.8f, 0.8f);   // Bottom-right
            GL.Vertex3(0.0f, -0.8f, -0.8f);  // Bottom-left
            GL.Vertex3(0.0f, 0.8f, -0.8f);   // Top-left
            GL.End();
            
            // Restore OpenGL state
            GL.LineWidth(originalLineWidth);
            
            if (depthTestEnabled)
                GL.Enable(EnableCap.DepthTest);
            if (!blendEnabled)
                GL.Disable(EnableCap.Blend);
                
            GL.UseProgram(shaderProgram);
        }

        private void GLControl_Resize(object sender, EventArgs e)
        {
            if (gl == null) return;

            gl.MakeCurrent();
            GL.Viewport(0, 0, gl.ClientSize.Width, gl.ClientSize.Height);
        }
        #endregion
    }
}