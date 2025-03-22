using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace DeepBridgeWindowsApp
{
    /// <summary>
    /// Form for annotating DICOM slice images with carotid markers
    /// </summary>
    public class SliceAnnotationForm : Form
    {
        // The original image
        private Bitmap originalImage;
        
        // The annotated image with markers
        private Bitmap workingImage;
        
        // List of annotation points (carotid markers)
        private List<Point> annotationPoints = new List<Point>();
        
        // Current point being dragged (or null if none)
        private int? dragPointIndex = null;
        
        // Annotation display settings
        private const int MarkerSize = 10;
        private readonly Color MarkerColor = Color.Red;
        private readonly Color MarkerOutlineColor = Color.White;
        private readonly Color MarkerLabelColor = Color.Yellow;
        private readonly Font LabelFont = new Font("Arial", 8, FontStyle.Bold);
        
        // Flag to check if the object has been disposed
        private bool isDisposed = false;
        
        // PictureBox for displaying the image
        private PictureBox pictureBox;
        
        // Buttons for controls
        private Button addMarkerButton;
        private Button clearMarkersButton;
        private Button acceptButton;
        private Button cancelButton;
        
        // Label to show instructions
        private Label instructionsLabel;
        
        /// <summary>
        /// Gets the annotated image with all markers
        /// </summary>
        public Bitmap AnnotatedImage => GenerateFinalImage();

        public SliceAnnotationForm(Bitmap sourceImage)
        {
            if (sourceImage == null)
                throw new ArgumentNullException(nameof(sourceImage), "Source image cannot be null");
            
            if (sourceImage.Width <= 0 || sourceImage.Height <= 0)
                throw new ArgumentException("Source image has invalid dimensions", nameof(sourceImage));
                
            try
            {
                Console.WriteLine($"Creating SliceAnnotationForm with image: {sourceImage.Width}x{sourceImage.Height}, Format: {sourceImage.PixelFormat}");
                
                // Create a blank image with known good format as fallback
                Bitmap safeImage = null;
                
                try
                {
                    // Try to verify image is valid
                    System.Drawing.Imaging.BitmapData bmpData = sourceImage.LockBits(
                        new Rectangle(0, 0, sourceImage.Width, sourceImage.Height),
                        System.Drawing.Imaging.ImageLockMode.ReadOnly,
                        sourceImage.PixelFormat);
                    sourceImage.UnlockBits(bmpData);
                    
                    // If we get here, the source image is valid
                    safeImage = sourceImage;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"WARNING: Source image could not be locked: {ex.Message}");
                    
                    // Since the source image is problematic, create a blank image instead
                    Console.WriteLine("Creating blank image as fallback...");
                    
                    try
                    {
                        // Create a simple blank image with standard format
                        safeImage = new Bitmap(400, 400, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                        
                        // Fill with black background
                        using (Graphics g = Graphics.FromImage(safeImage))
                        {
                            g.Clear(Color.Black);
                            
                            // Draw text explaining the error
                            using (Font font = new Font("Arial", 12, FontStyle.Bold))
                            using (SolidBrush brush = new SolidBrush(Color.White))
                            {
                                g.DrawString("Error loading source image.", 
                                    font, brush, 100, 180);
                                g.DrawString("You can still annotate this blank image.", 
                                    font, brush, 70, 200);
                            }
                        }
                    }
                    catch (Exception fallbackEx)
                    {
                        Console.WriteLine($"Error creating fallback image: {fallbackEx}");
                        throw new ApplicationException("Cannot create a valid image for annotation", fallbackEx);
                    }
                }
                
                // Now use our safe image to create working copies
                // Use the converter to ensure standard format
                originalImage = ConvertToArgb32(safeImage);
                workingImage = ConvertToArgb32(safeImage);
                
                // Verify the images were created successfully
                if (originalImage == null || workingImage == null)
                {
                    throw new ApplicationException("Failed to create working image copies");
                }
                
                InitializeComponents();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error initializing SliceAnnotationForm: {ex}");
                throw new ApplicationException("Error initializing SliceAnnotationForm: " + ex.Message, ex);
            }
        }
        
        // Helper method to safely convert any bitmap to a standard 32-bit ARGB format
        private Bitmap ConvertToArgb32(Bitmap source)
        {
            if (source == null) return null;
            
            try
            {
                // Create a new bitmap with a standard format
                Bitmap converted = new Bitmap(source.Width, source.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                
                using (Graphics g = Graphics.FromImage(converted))
                {
                    // Set high quality rendering
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.SmoothingMode = SmoothingMode.AntiAlias;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                    
                    // Draw the source onto the new bitmap
                    Rectangle destRect = new Rectangle(0, 0, source.Width, source.Height);
                    g.DrawImage(source, destRect, 0, 0, source.Width, source.Height, GraphicsUnit.Pixel);
                }
                
                return converted;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error converting bitmap format: {ex}");
                return null;
            }
        }
        
        private void InitializeComponents()
        {
            // Form settings
            this.Text = "Annotate Slice";
            this.Size = new Size(Math.Max(800, originalImage.Width + 40), 
                                 Math.Max(600, originalImage.Height + 120));
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.StartPosition = FormStartPosition.CenterParent;
            
            // Create instructions label
            instructionsLabel = new Label
            {
                Text = "Click to add carotid markers. Drag markers to adjust position. " +
                      "These annotations will help our AI model learn carotid positions.",
                AutoSize = false,
                TextAlign = ContentAlignment.MiddleCenter,
                Width = this.ClientSize.Width,
                Height = 40,
                Dock = DockStyle.Top
            };
            
            // Create picture box for the image
            pictureBox = new PictureBox
            {
                SizeMode = PictureBoxSizeMode.Zoom,
                Dock = DockStyle.Fill,
                Image = workingImage
            };
            
            // Add mouse events for interaction
            pictureBox.MouseClick += PictureBox_MouseClick;
            pictureBox.MouseDown += PictureBox_MouseDown;
            pictureBox.MouseMove += PictureBox_MouseMove;
            pictureBox.MouseUp += PictureBox_MouseUp;
            pictureBox.Paint += PictureBox_Paint;
            
            // Create buttons panel
            var buttonPanel = new Panel
            {
                Height = 50,
                Dock = DockStyle.Bottom
            };
            
            // Create buttons
            addMarkerButton = new Button
            {
                Text = "Add Marker",
                Width = 120,
                Height = 30,
                Location = new Point(10, 10)
            };
            addMarkerButton.Click += AddMarkerButton_Click;
            
            clearMarkersButton = new Button
            {
                Text = "Clear All",
                Width = 120,
                Height = 30,
                Location = new Point(140, 10)
            };
            clearMarkersButton.Click += ClearMarkersButton_Click;
            
            acceptButton = new Button
            {
                Text = "Save",
                Width = 120,
                Height = 30,
                DialogResult = DialogResult.OK,
                Location = new Point(this.ClientSize.Width - 260, 10)
            };
            
            cancelButton = new Button
            {
                Text = "Cancel",
                Width = 120,
                Height = 30,
                DialogResult = DialogResult.Cancel,
                Location = new Point(this.ClientSize.Width - 130, 10)
            };
            
            // Add buttons to panel
            buttonPanel.Controls.Add(addMarkerButton);
            buttonPanel.Controls.Add(clearMarkersButton);
            buttonPanel.Controls.Add(acceptButton);
            buttonPanel.Controls.Add(cancelButton);
            
            // Add controls to form
            this.Controls.Add(pictureBox);
            this.Controls.Add(instructionsLabel);
            this.Controls.Add(buttonPanel);
            
            // Set cancel button
            this.CancelButton = cancelButton;
            this.AcceptButton = acceptButton;
        }

        private void PictureBox_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                // Convert screen coordinates to image coordinates
                Point imagePoint = ConvertToImageCoordinates(e.Location);
                
                // Check if we're near an existing point
                int index = FindNearbyMarker(imagePoint);
                if (index == -1)
                {
                    // Add new point
                    annotationPoints.Add(imagePoint);
                    pictureBox.Invalidate();
                }
            }
        }

        private void PictureBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                Point imagePoint = ConvertToImageCoordinates(e.Location);
                int index = FindNearbyMarker(imagePoint);
                if (index != -1)
                {
                    dragPointIndex = index;
                }
            }
        }

        private void PictureBox_MouseMove(object sender, MouseEventArgs e)
        {
            if (dragPointIndex.HasValue)
            {
                Point imagePoint = ConvertToImageCoordinates(e.Location);
                annotationPoints[dragPointIndex.Value] = imagePoint;
                pictureBox.Invalidate();
            }
        }

        private void PictureBox_MouseUp(object sender, MouseEventArgs e)
        {
            dragPointIndex = null;
        }

        private void PictureBox_Paint(object sender, PaintEventArgs e)
        {
            // Get the scaling factor of the PictureBox (for zoom)
            float scaleFactor = Math.Min(
                (float)pictureBox.Width / originalImage.Width,
                (float)pictureBox.Height / originalImage.Height);
            
            // Calculate offset for centering image in the PictureBox
            float offsetX = (pictureBox.Width - (originalImage.Width * scaleFactor)) / 2;
            float offsetY = (pictureBox.Height - (originalImage.Height * scaleFactor)) / 2;
            
            // Draw markers
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            
            for (int i = 0; i < annotationPoints.Count; i++)
            {
                Point p = annotationPoints[i];
                
                // Convert image coordinates back to screen coordinates
                float screenX = p.X * scaleFactor + offsetX;
                float screenY = p.Y * scaleFactor + offsetY;
                
                // Draw the marker with white outline
                e.Graphics.FillEllipse(
                    new SolidBrush(MarkerOutlineColor),
                    screenX - (MarkerSize + 2) / 2,
                    screenY - (MarkerSize + 2) / 2,
                    MarkerSize + 2,
                    MarkerSize + 2);
                
                // Draw the red marker
                e.Graphics.FillEllipse(
                    new SolidBrush(MarkerColor),
                    screenX - MarkerSize / 2,
                    screenY - MarkerSize / 2,
                    MarkerSize,
                    MarkerSize);
                
                // Draw marker number
                string label = (i + 1).ToString();
                SizeF labelSize = e.Graphics.MeasureString(label, LabelFont);
                e.Graphics.DrawString(
                    label,
                    LabelFont,
                    new SolidBrush(MarkerLabelColor),
                    screenX - labelSize.Width / 2,
                    screenY - labelSize.Height / 2);
            }
        }

        private void AddMarkerButton_Click(object sender, EventArgs e)
        {
            // Add a marker in the center
            Point center = new Point(originalImage.Width / 2, originalImage.Height / 2);
            annotationPoints.Add(center);
            pictureBox.Invalidate();
        }

        private void ClearMarkersButton_Click(object sender, EventArgs e)
        {
            annotationPoints.Clear();
            pictureBox.Invalidate();
        }

        private int FindNearbyMarker(Point point)
        {
            for (int i = 0; i < annotationPoints.Count; i++)
            {
                Point p = annotationPoints[i];
                double distance = Math.Sqrt(Math.Pow(p.X - point.X, 2) + Math.Pow(p.Y - point.Y, 2));
                if (distance < MarkerSize)
                {
                    return i;
                }
            }
            return -1;
        }

        private Point ConvertToImageCoordinates(Point screenPoint)
        {
            // Get the scaling factor of the PictureBox (for zoom)
            float scaleFactor = Math.Min(
                (float)pictureBox.Width / originalImage.Width,
                (float)pictureBox.Height / originalImage.Height);
            
            // Calculate offset for centering image in the PictureBox
            float offsetX = (pictureBox.Width - (originalImage.Width * scaleFactor)) / 2;
            float offsetY = (pictureBox.Height - (originalImage.Height * scaleFactor)) / 2;
            
            // Convert screen coordinates to image coordinates
            int imageX = (int)((screenPoint.X - offsetX) / scaleFactor);
            int imageY = (int)((screenPoint.Y - offsetY) / scaleFactor);
            
            // Clamp to image boundaries
            imageX = Math.Max(0, Math.Min(imageX, originalImage.Width - 1));
            imageY = Math.Max(0, Math.Min(imageY, originalImage.Height - 1));
            
            return new Point(imageX, imageY);
        }

        private Bitmap GenerateFinalImage()
        {
            try
            {
                // Check if we have a valid original image
                if (originalImage == null || originalImage.Width <= 0 || originalImage.Height <= 0)
                {
                    throw new InvalidOperationException("Invalid original image");
                }
                
                // Create a new bitmap with the same dimensions and format as the original
                Bitmap finalImage = new Bitmap(originalImage.Width, originalImage.Height, originalImage.PixelFormat);
                
                // Create a graphics object to draw on the bitmap
                using (Graphics g = Graphics.FromImage(finalImage))
                {
                    // Set high quality drawing
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.SmoothingMode = SmoothingMode.AntiAlias;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                
                    // First draw the original image onto the new bitmap
                    Rectangle destRect = new Rectangle(0, 0, originalImage.Width, originalImage.Height);
                    g.DrawImage(originalImage, destRect, 0, 0, originalImage.Width, originalImage.Height, GraphicsUnit.Pixel);
                    
                    // Draw all markers
                    for (int i = 0; i < annotationPoints.Count; i++)
                    {
                        Point p = annotationPoints[i];
                        
                        // Ensure point is within image bounds
                        if (p.X < 0 || p.X >= originalImage.Width || p.Y < 0 || p.Y >= originalImage.Height)
                            continue;
                        
                        // Draw the marker with white outline
                        using (SolidBrush outlineBrush = new SolidBrush(MarkerOutlineColor))
                        {
                            g.FillEllipse(
                                outlineBrush,
                                p.X - (MarkerSize + 2) / 2,
                                p.Y - (MarkerSize + 2) / 2,
                                MarkerSize + 2,
                                MarkerSize + 2);
                        }
                        
                        // Draw the red marker
                        using (SolidBrush markerBrush = new SolidBrush(MarkerColor))
                        {
                            g.FillEllipse(
                                markerBrush,
                                p.X - MarkerSize / 2,
                                p.Y - MarkerSize / 2,
                                MarkerSize,
                                MarkerSize);
                        }
                        
                        // Draw marker number
                        string label = (i + 1).ToString();
                        SizeF labelSize = g.MeasureString(label, LabelFont);
                        
                        using (SolidBrush labelBrush = new SolidBrush(MarkerLabelColor))
                        {
                            g.DrawString(
                                label,
                                LabelFont,
                                labelBrush,
                                p.X - labelSize.Width / 2,
                                p.Y - labelSize.Height / 2);
                        }
                    }
                }
                
                return finalImage;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating annotated image: {ex.Message}");
                MessageBox.Show($"Error generating annotated image: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                
                // In case of error, create a blank bitmap
                return new Bitmap(1, 1);
            }
        }

        // Override Dispose to clean up resources properly
        protected override void Dispose(bool disposing)
        {
            if (!isDisposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    if (LabelFont != null) LabelFont.Dispose();
                    if (workingImage != null) workingImage.Dispose();
                    if (originalImage != null) originalImage.Dispose();
                    if (pictureBox != null && pictureBox.Image != null) pictureBox.Image.Dispose();
                }
                
                isDisposed = true;
            }
            
            base.Dispose(disposing);
        }
        
        // Clean up resources when form closes
        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            base.OnFormClosed(e);
        }
    }
}