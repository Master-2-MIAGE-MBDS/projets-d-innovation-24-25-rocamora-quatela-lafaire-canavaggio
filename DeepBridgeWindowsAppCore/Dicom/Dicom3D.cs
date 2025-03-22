using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;

namespace DeepBridgeWindowsApp.Dicom
{
    public class ProcessingProgress
    {
        public string CurrentStep { get; set; }
        public int CurrentValue { get; set; }
        public int TotalValue { get; set; }
        public float Percentage => (float)CurrentValue / TotalValue * 100;
    }

    /// <summary>
    /// Ultra lightweight 3D DICOM renderer that processes slices on-demand
    /// and avoids storing all slices in memory at once.
    /// </summary>
    public class Dicom3D : IDisposable
    {
        // Constants for memory optimization
        private const int MAX_VISIBLE_SLICES = 60;  // Increased to ensure all slices are visible without gaps
        private const float INTENSITY_THRESHOLD = 0.15f;  // Threshold for detecting non-background pixels
        private const int SAMPLE_RATE = 3;  // Reduced sampling rate for higher point density
        
        // OpenGL resources
        private int[] vertexBufferObject;
        private int[] colorBufferObject;
        private int[] elementBufferObject;
        private int vertexArrayObject;
        
        // Source data
        private DicomDisplayManager dicomDisplayManager;
        private readonly object lockObject = new object();
        private int totalSlices;
        private int sliceWidth;
        private int sliceHeight;
        
        // Display settings
        private int frontClip = 0;
        private int backClip = 0;
        
        // Cached data for current view
        private List<Vector3> visibleVertices = new List<Vector3>();
        private List<Vector3> visibleColors = new List<Vector3>();
        private List<int> visibleIndices = new List<int>();
        
        // Current view state
        private Matrix4 currentModelMatrix = Matrix4.Identity;
        private Vector3 cameraPosition = new Vector3(0, 0, 3);
        private Vector3 cameraTarget = Vector3.Zero;
        private bool isInitialized = false;
        
        // Metadata for extracting slices
        private Dictionary<(int z, int row), Dictionary<int, float>> sliceMap = 
            new Dictionary<(int z, int row), Dictionary<int, float>>();
        
        // Physical dimensions
        private float physicalWidth, physicalHeight, physicalDepth;
        private float firstSliceLocation;
        private float pixelSpacingX, pixelSpacingY;

        /// <summary>
        /// Creates a new ultra-lightweight 3D DICOM renderer
        /// </summary>
        public Dicom3D(DicomDisplayManager ddm, Action<ProcessingProgress> progressCallback = null)
        {
            Console.WriteLine("Initializing lightweight 3D renderer");
            dicomDisplayManager = ddm;
            totalSlices = ddm.GetTotalSlices();
            
            // Get basic metadata without loading any images
            InitializeMetadata(ddm);
            
            // We don't preload or process any slices until they're actually needed for rendering
            Console.WriteLine($"Ready for on-demand 3D rendering: {totalSlices} slices, {sliceWidth}x{sliceHeight}");
        }

        /// <summary>
        /// Get basic metadata without loading any pixel data
        /// </summary>
        private void InitializeMetadata(DicomDisplayManager ddm)
        {
            // Get pixel spacing (in mm)
            var pixelSpacing = ddm.GetSlice(0).PixelSpacing;
            pixelSpacingX = (float)pixelSpacing;
            pixelSpacingY = (float)pixelSpacing;

            // Get first slice to determine dimensions
            var firstSlice = ddm.GetCurrentSliceImage();
            sliceWidth = firstSlice.Width;
            sliceHeight = firstSlice.Height;
            
            // Calculate physical dimensions
            physicalWidth = sliceWidth * pixelSpacingX;
            physicalHeight = sliceHeight * pixelSpacingY;
            
            // Get z-axis physical dimensions
            firstSliceLocation = (float)ddm.GetSlice(0).SliceLocation;
            float lastSliceLocation = (float)ddm.GetSlice(ddm.GetTotalSlices() - 1).SliceLocation;
            physicalDepth = Math.Abs(lastSliceLocation - firstSliceLocation);
            
            // Clean up immediately
            firstSlice.Dispose();
            GC.Collect();
            
            Console.WriteLine($"3D volume dimensions: {sliceWidth}x{sliceHeight}x{totalSlices}");
        }
        
        /// <summary>
        /// Sets front and back clipping planes
        /// </summary>
        public void SetClipPlanes(int front, int back)
        {
            frontClip = front;
            backClip = back;
            
            // Force regeneration of visible vertices with new clipping
            if (isInitialized)
            {
                RegenerateVisibleVertices();
            }
        }
        
        /// <summary>
        /// Determines which slices are visible from the current camera position
        /// and processes only those slices
        /// </summary>
        private void RegenerateVisibleVertices()
        {
            try
            {
                // Clear previous data
                visibleVertices.Clear();
                visibleColors.Clear();
                visibleIndices.Clear();
                
                // Calculate which slices to process based on camera position
                List<int> slicesToProcess = DetermineVisibleSlices();
                
                // Track successfully processed slices
                int successfulSlices = 0;
                
                // Process only these slices with failure handling
                foreach (int z in slicesToProcess)
                {
                    try
                    {
                        ProcessSingleSlice(z);
                        successfulSlices++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to process slice {z}: {ex.Message}");
                        // Continue with other slices even if this one fails
                    }
                }
                
                // Ensure we have at least some vertices to render
                if (visibleVertices.Count == 0 && successfulSlices == 0)
                {
                    // Generate some fallback vertices to ensure rendering doesn't fail completely
                    GenerateFallbackVertices();
                }
                
                // Update OpenGL buffers
                UpdateGLBuffers();
                
                // Force cleanup to reduce memory usage
                GC.Collect();
                
                Console.WriteLine($"Regenerated 3D view with {visibleVertices.Count} vertices from {successfulSlices}/{slicesToProcess.Count} slices");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error regenerating vertices: {ex.Message}");
                
                // Ensure we have some vertices even after an error
                if (visibleVertices.Count == 0)
                {
                    GenerateFallbackVertices();
                    UpdateGLBuffers();
                }
            }
        }
        
        /// <summary>
        /// Creates fallback vertices when normal slice processing fails
        /// </summary>
        private void GenerateFallbackVertices()
        {
            Console.WriteLine("Generating fallback vertices");
            
            // Add a simple cube as fallback geometry
            // This ensures rendering doesn't completely fail
            
            // Front face
            visibleVertices.Add(new Vector3(-0.5f, -0.5f, 0.5f));
            visibleVertices.Add(new Vector3(0.5f, -0.5f, 0.5f));
            visibleVertices.Add(new Vector3(0.5f, 0.5f, 0.5f));
            visibleVertices.Add(new Vector3(-0.5f, 0.5f, 0.5f));
            
            // Back face
            visibleVertices.Add(new Vector3(-0.5f, -0.5f, -0.5f));
            visibleVertices.Add(new Vector3(0.5f, -0.5f, -0.5f));
            visibleVertices.Add(new Vector3(0.5f, 0.5f, -0.5f));
            visibleVertices.Add(new Vector3(-0.5f, 0.5f, -0.5f));
            
            // Add colors (red for fallback to make it obvious)
            for (int i = 0; i < 8; i++)
            {
                visibleColors.Add(new Vector3(1.0f, 0.0f, 0.0f));
                visibleIndices.Add(i);
            }
            
            Console.WriteLine("Added fallback geometry with 8 vertices");
        }
        
        /// <summary>
        /// Determines which slices are most visible from the current camera position
        /// </summary>
        private List<int> DetermineVisibleSlices()
        {
            // Apply clipping planes
            int minSlice = frontClip;
            int maxSlice = totalSlices - backClip - 1;
            
            // Calculate available slice range
            int sliceRange = maxSlice - minSlice + 1;
            if (sliceRange <= 0) return new List<int>();
            
            // Normalize z-coordinate of camera position to slice space
            float normalizedCameraZ = 0.5f; // Default middle
            
            // If camera is inside the volume or we have a rotation, select slices
            // closest to facing the camera
            if (Math.Abs(currentModelMatrix.M33) > 0.7f) 
            {
                // Camera is mainly looking along z-axis
                if (cameraPosition.Z > 0)
                {
                    // Looking from front to back
                    normalizedCameraZ = 0.0f;
                }
                else
                {
                    // Looking from back to front 
                    normalizedCameraZ = 1.0f;
                }
            }
            
            // Calculate how many slices to select (based on available slices)
            int slicesToSelect = Math.Min(MAX_VISIBLE_SLICES, sliceRange);
            
            List<int> selectedSlices = new List<int>();
            
            // Distributed slice selection across the volume
            if (slicesToSelect >= sliceRange)
            {
                // If we can show all slices, just take them all
                for (int i = minSlice; i <= maxSlice; i++)
                {
                    selectedSlices.Add(i);
                }
            }
            else
            {
                // Select ALL available slices to completely eliminate gaps
                // Gaps occur when we don't process enough consecutive slices
                
                // Process ALL slices if possible to eliminate gaps
                // If we can't process all slices, distribute them evenly
                int startIndex;
                int stepSize = 1;
                
                if (sliceRange > slicesToSelect) {
                    // Calculate appropriate step to distribute slices evenly through the volume
                    stepSize = Math.Max(1, sliceRange / slicesToSelect);
                    Console.WriteLine($"Using step size {stepSize} to distribute {slicesToSelect} slices across {sliceRange} range");
                }
                
                // Start from the minimum slice
                startIndex = minSlice;
                
                // Add evenly distributed slices to ensure whole volume coverage without gaps
                for (int i = 0; i < sliceRange; i += stepSize)
                {
                    int sliceIndex = startIndex + i;
                    if (sliceIndex <= maxSlice && selectedSlices.Count < slicesToSelect)
                    {
                        selectedSlices.Add(sliceIndex);
                    }
                }
                
                // If we didn't add enough slices, add more until we reach the target
                while (selectedSlices.Count < slicesToSelect && selectedSlices.Count < sliceRange)
                {
                    // Find gaps and fill them
                    for (int i = minSlice; i <= maxSlice && selectedSlices.Count < slicesToSelect; i++)
                    {
                        if (!selectedSlices.Contains(i))
                        {
                            selectedSlices.Add(i);
                            break;
                        }
                    }
                }
                
                // Make sure we include the front and back slice when selecting a subset
                if (!selectedSlices.Contains(minSlice) && minSlice < maxSlice)
                {
                    selectedSlices.Add(minSlice);
                }
                
                if (!selectedSlices.Contains(maxSlice) && maxSlice > minSlice)
                {
                    selectedSlices.Add(maxSlice);
                }
            }
            
            // Sort the slices for consistent rendering
            selectedSlices.Sort();
            
            Console.WriteLine($"Selected {selectedSlices.Count} slices from range {minSlice}-{maxSlice}");
            
            return selectedSlices;
        }
        
        /// <summary>
        /// Processes a single slice into vertices for rendering
        /// </summary>
        private void ProcessSingleSlice(int z)
        {
            try
            {
                Console.WriteLine($"Processing slice {z}/{totalSlices}");
                // Calculate normalized z position without offset to make slices stick together
                float normalizedZ = (z - frontClip) / (float)(totalSlices - frontClip - backClip);
                
                // Use actual slice location for more accurate physical positioning in Z direction
                if (physicalDepth > 0)
                {
                    float sliceLocation = 0;
                    try {
                        sliceLocation = (float)dicomDisplayManager.GetSlice(z).SliceLocation;
                        // Calculate exact physical position based on actual slice location without subtraction
                        normalizedZ = (sliceLocation - firstSliceLocation) / physicalDepth;
                    }
                    catch {
                        // Fallback to index-based position if slice location isn't available
                    }
                }
                
                // Center the model by subtracting 0.5f after all calculations
                normalizedZ -= 0.5f;
                
                // Get the slice image (loaded on demand)
                dicomDisplayManager.SetSliceIndex(z);
                
                // Safely get the slice image with error handling
                Bitmap slice = null;
                try
                {
                    slice = dicomDisplayManager.GetCurrentSliceImage();
                    Console.WriteLine($"Retrieved slice {z}: {(slice != null ? $"{slice.Width}x{slice.Height}" : "null")}");
                    
                    // Skip if slice is null
                    if (slice == null)
                    {
                        Console.WriteLine($"Skip slice {z}: null bitmap returned");
                        return;
                    }
                    
                    // Validate dimensions
                    if (slice.Width <= 0 || slice.Height <= 0)
                    {
                        Console.WriteLine($"Skip slice {z}: invalid dimensions {slice.Width}x{slice.Height}");
                        return;
                    }
                    
                    // Get slice location if available
                    float sliceLocation = 0;
                    try {
                        sliceLocation = (float)dicomDisplayManager.GetSlice(z).SliceLocation;
                    }
                    catch (Exception ex) {
                        Console.WriteLine($"Could not get slice location for slice {z}: {ex.Message}");
                    }
                    
                    // Verify that the bitmap can be locked
                    System.Drawing.Imaging.PixelFormat format = slice.PixelFormat;
                    if (!IsValidPixelFormat(format))
                    {
                        Console.WriteLine($"Skip slice {z}: unsupported pixel format {format}");
                        return;
                    }
                    
                    // Lock the bitmap to access pixel data
                    BitmapData bitmapData = null;
                    try
                    {
                        bitmapData = slice.LockBits(
                            new Rectangle(0, 0, slice.Width, slice.Height),
                            ImageLockMode.ReadOnly,
                            System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                            
                        if (bitmapData.Scan0 == IntPtr.Zero)
                        {
                            Console.WriteLine($"Skip slice {z}: null scan pointer");
                            return;
                        }
                        
                        unsafe
                        {
                            byte* ptr = (byte*)bitmapData.Scan0;
                            int vertexCount = 0;
                            int maxVerticesPerSlice = 12000; // Increased to ensure denser slice representation
                            
                            // Use much denser sampling to completely eliminate gaps between slices
                            int localSampleRate = 1; // Fixed to 1 (every pixel) for critical slices
                            
                            // Process every SAMPLE_RATE-th pixel to reduce memory usage
                            for (int y = 0; y < slice.Height; y += localSampleRate)
                            {
                                // Skip processing if we already have enough vertices for this slice
                                if (vertexCount >= maxVerticesPerSlice) break;
                                
                                for (int x = 0; x < slice.Width; x += localSampleRate)
                                {
                                    // Check vertex count limit for better performance
                                    if (vertexCount >= maxVerticesPerSlice) break;
                                    
                                    // Safety check for offset calculation
                                    if (y * bitmapData.Stride + x * 4 + 2 >= bitmapData.Stride * slice.Height)
                                    {
                                        continue; // Skip pixels that would cause an overflow
                                    }
                                    
                                    int offset = y * bitmapData.Stride + x * 4;
                                    byte b = ptr[offset];
                                    byte g = ptr[offset + 1];
                                    byte r = ptr[offset + 2];
                                    
                                    // Calculate grayscale intensity
                                    float intensity = (r * 0.299f + g * 0.587f + b * 0.114f) / 255f;
                                    
                                    // Use a lower threshold to include more points for slice continuity
                                    // This helps fill gaps between slices
                                    if (intensity <= INTENSITY_THRESHOLD * 0.8f)
                                        continue;
                                    
                                    // Use more dense sampling for higher intensity pixels
                                    // to make sure important features have no gaps
                                    if (intensity > 0.6f)
                                    {
                                        // Always include high-intensity pixels
                                    }
                                    else if (intensity > 0.4f)
                                    {
                                        // Use moderate subsampling for medium intensity pixels
                                        if ((x + y) % 2 != 0)
                                            continue;
                                    }
                                    else
                                    {
                                        // Use maximum subsampling for low (but above threshold) intensity pixels
                                        if ((x + y) % 3 != 0)
                                            continue;
                                    }
                                    
                                    // Calculate normalized position in 3D space
                                    float normalizedX = (float)x / slice.Width - 0.5f;
                                    float normalizedY = (float)y / slice.Height - 0.5f;
                                    
                                    // Create a vertex at this position
                                    Vector3 vertex = new Vector3(normalizedX, normalizedY, normalizedZ);
                                    
                                    // Enhanced color scheme that connects slices visually
                                    Vector3 color;
                                    float normalizedSlicePosition = (float)z / totalSlices;
                                    
                                    // Apply more consistent coloring between slices
                                    if (intensity > 0.7f) {
                                        // Brightest areas: white with consistent tint across slices
                                        color = new Vector3(
                                            0.95f, 
                                            0.95f, 
                                            0.95f
                                        );
                                    } else if (intensity > 0.5f) {
                                        // Medium-high intensity: nearly white with slight depth coloring
                                        color = new Vector3(
                                            0.85f + (normalizedSlicePosition * 0.1f),
                                            0.85f,
                                            0.85f + ((1 - normalizedSlicePosition) * 0.1f)
                                        );
                                    } else {
                                        // Lower intensity: more pronounced depth coloring
                                        float baseIntensity = intensity * 0.85f; // Higher base value to make all slices brighter
                                        color = new Vector3(
                                            baseIntensity + (normalizedSlicePosition * 0.15f),  // Red increases with depth
                                            baseIntensity,
                                            baseIntensity + ((1 - normalizedSlicePosition) * 0.15f)  // Blue decreases with depth
                                        );
                                    }
                                    
                                    // Store in our visible set
                                    lock (lockObject)
                                    {
                                        visibleVertices.Add(vertex);
                                        visibleColors.Add(color);
                                        visibleIndices.Add(visibleVertices.Count - 1);
                                        vertexCount++;
                                        
                                        // Only store critical points for slicing to reduce memory
                                        if (intensity > INTENSITY_THRESHOLD + 0.1f)
                                        {
                                            StoreForSlicing(normalizedX, normalizedY, z, intensity);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    finally
                    {
                        // Make sure we unlock the bitmap if it was locked
                        if (bitmapData != null)
                        {
                            try
                            {
                                slice.UnlockBits(bitmapData);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error unlocking bitmap for slice {z}: {ex.Message}");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing slice {z}: {ex.Message}");
                }
                finally
                {
                    // Always dispose the bitmap to free memory
                    if (slice != null)
                    {
                        try
                        {
                            slice.Dispose();
                            slice = null;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error disposing bitmap for slice {z}: {ex.Message}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Critical error processing slice {z}: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Check if a pixel format is valid for bitmap locking
        /// </summary>
        private bool IsValidPixelFormat(System.Drawing.Imaging.PixelFormat format)
        {
            // These formats are known to work reliably with LockBits
            return format == System.Drawing.Imaging.PixelFormat.Format24bppRgb ||
                   format == System.Drawing.Imaging.PixelFormat.Format32bppRgb ||
                   format == System.Drawing.Imaging.PixelFormat.Format32bppArgb ||
                   format == System.Drawing.Imaging.PixelFormat.Format32bppPArgb;
        }
        
        /// <summary>
        /// Store a point for later slicing operations
        /// </summary>
        private void StoreForSlicing(float normalizedX, float normalizedY, int z, float intensity)
        {
            // Convert to pixel coordinates
            int row = (int)((normalizedX + 0.5f) * sliceWidth);
            int col = (int)((normalizedY + 0.5f) * sliceHeight);
            
            // Store in sparse dictionary structure
            var key = (z, row);
            if (!sliceMap.ContainsKey(key))
            {
                sliceMap[key] = new Dictionary<int, float>();
            }
            sliceMap[key][col] = intensity;
        }

        /// <summary>
        /// Updates the OpenGL buffers with only the visible vertices
        /// </summary>
        private void UpdateGLBuffers()
        {
            if (!isInitialized) return;
            
            // Update vertex buffer
            GL.BindBuffer(BufferTarget.ArrayBuffer, vertexBufferObject[0]);
            GL.BufferData(BufferTarget.ArrayBuffer, visibleVertices.Count * Vector3.SizeInBytes, 
                visibleVertices.ToArray(), BufferUsageHint.DynamicDraw);
                
            // Update color buffer
            GL.BindBuffer(BufferTarget.ArrayBuffer, colorBufferObject[0]);
            GL.BufferData(BufferTarget.ArrayBuffer, visibleColors.Count * Vector3.SizeInBytes, 
                visibleColors.ToArray(), BufferUsageHint.DynamicDraw);
                
            // Update index buffer
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, elementBufferObject[0]);
            GL.BufferData(BufferTarget.ElementArrayBuffer, visibleIndices.Count * sizeof(int),
                visibleIndices.ToArray(), BufferUsageHint.DynamicDraw);
        }

        /// <summary>
        /// Initializes OpenGL resources
        /// </summary>
        public void InitializeGL()
        {
            Console.WriteLine("Initializing OpenGL resources");
            
            // Create OpenGL resources
            vertexBufferObject = new int[1];
            colorBufferObject = new int[1];
            elementBufferObject = new int[1];
            vertexArrayObject = GL.GenVertexArray();
            GL.GenBuffers(1, vertexBufferObject);
            GL.GenBuffers(1, colorBufferObject);
            GL.GenBuffers(1, elementBufferObject);

            GL.BindVertexArray(vertexArrayObject);

            // Create empty buffers initially
            GL.BindBuffer(BufferTarget.ArrayBuffer, vertexBufferObject[0]);
            GL.BufferData(BufferTarget.ArrayBuffer, 0, IntPtr.Zero, BufferUsageHint.DynamicDraw);
            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, Vector3.SizeInBytes, 0);
            GL.EnableVertexAttribArray(0);

            GL.BindBuffer(BufferTarget.ArrayBuffer, colorBufferObject[0]);
            GL.BufferData(BufferTarget.ArrayBuffer, 0, IntPtr.Zero, BufferUsageHint.DynamicDraw);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, Vector3.SizeInBytes, 0);
            GL.EnableVertexAttribArray(1);

            GL.BindBuffer(BufferTarget.ElementArrayBuffer, elementBufferObject[0]);
            GL.BufferData(BufferTarget.ElementArrayBuffer, 0, IntPtr.Zero, BufferUsageHint.DynamicDraw);
                
            isInitialized = true;
        }

        // Track previous camera position to detect significant changes
        private Vector3 previousCameraPosition = new Vector3(0, 0, 0);
        private Matrix4 previousModelMatrix = Matrix4.Identity;
        private int frameCounter = 0;
        private int consecutiveRenderFailures = 0;
        private const int MAX_FAILURES_BEFORE_FALLBACK = 3;
        
        // Track when last regeneration happened for throttling
        private DateTime lastRegenerationTime = DateTime.MinValue;
        private const int MIN_REGEN_INTERVAL_MS = 300; // Minimum time between regenerations
        
        /// <summary>
        /// Renders the 3D DICOM data using currently visible slices
        /// </summary>
        public void Render(int shader, Matrix4 model, Matrix4 view, Matrix4 projection)
        {
            try
            {
                if (!isInitialized) 
                {
                    Console.WriteLine("Render called before initialization");
                    return;
                }
                
                // Save the current model matrix and camera information
                currentModelMatrix = model;
                
                try
                {
                    // Extract camera position from view matrix
                    Matrix4 invView = Matrix4.Invert(view);
                    cameraPosition = invView.ExtractTranslation();
                }
                catch (Exception matrixEx)
                {
                    Console.WriteLine($"Error extracting camera position: {matrixEx.Message}");
                    // Use default camera position if extraction fails
                    cameraPosition = new Vector3(0, 0, 3);
                }
                
                // Check if we need to regenerate vertices based on new view
                bool needsRegeneration = visibleVertices.Count == 0;
                
                // Detect significant camera position change
                float cameraMovementDistance = Vector3.Distance(cameraPosition, previousCameraPosition);
                bool significantModelChange = false;
                
                try
                {
                    significantModelChange = !Matrix4.Equals(currentModelMatrix, previousModelMatrix);
                }
                catch (Exception matrixEx)
                {
                    Console.WriteLine($"Error comparing matrices: {matrixEx.Message}");
                }
                
                // Force regeneration periodically but less frequently to reduce CPU/memory usage
                frameCounter++;
                bool periodicUpdate = frameCounter % 150 == 0; // Much less frequent updates (was 60)
                
                // Check if enough time has passed since last regeneration (throttling)
                TimeSpan timeSinceLastRegen = DateTime.Now - lastRegenerationTime;
                bool canRegenerate = timeSinceLastRegen.TotalMilliseconds > MIN_REGEN_INTERVAL_MS;
                
                // Check for significant camera movement that would require regeneration
                if ((cameraMovementDistance > 0.7f || significantModelChange || periodicUpdate) && canRegenerate)
                {
                    needsRegeneration = true;
                    previousCameraPosition = cameraPosition;
                    previousModelMatrix = currentModelMatrix;
                    lastRegenerationTime = DateTime.Now;
                    Console.WriteLine($"Regenerating due to camera movement: {cameraMovementDistance:F2} or model change: {significantModelChange}");
                }
                
                // Regenerate if needed
                if (needsRegeneration)
                {
                    try
                    {
                        // Clear memory before regeneration
                        GC.Collect(0, GCCollectionMode.Optimized);
                        
                        RegenerateVisibleVertices();
                        consecutiveRenderFailures = 0; // Reset failure counter on success
                    }
                    catch (Exception regEx)
                    {
                        Console.WriteLine($"Error during vertex regeneration: {regEx.Message}");
                        consecutiveRenderFailures++;
                        
                        // After multiple failures, try to create fallback geometry
                        if (consecutiveRenderFailures >= MAX_FAILURES_BEFORE_FALLBACK)
                        {
                            Console.WriteLine("Too many consecutive failures, creating fallback geometry");
                            visibleVertices.Clear();
                            visibleColors.Clear();
                            visibleIndices.Clear();
                            GenerateFallbackVertices();
                            UpdateGLBuffers();
                        }
                    }
                }
                
                // Skip rendering if no vertices
                if (visibleVertices.Count == 0 || visibleIndices.Count == 0)
                {
                    Console.WriteLine("No vertices to render");
                    return;
                }
                
                // Verify array object is valid
                if (vertexArrayObject <= 0)
                {
                    Console.WriteLine("Invalid vertex array object");
                    return;
                }
                
                // Set shader uniforms
                GL.UseProgram(shader);
                
                // Check for OpenGL errors after setting shader
                ErrorCode error = GL.GetError();
                if (error != ErrorCode.NoError)
                {
                    Console.WriteLine($"OpenGL error after setting shader: {error}");
                }
                
                try
                {
                    // Set uniforms
                    GL.UniformMatrix4(GL.GetUniformLocation(shader, "model"), false, ref model);
                    GL.UniformMatrix4(GL.GetUniformLocation(shader, "view"), false, ref view);
                    GL.UniformMatrix4(GL.GetUniformLocation(shader, "projection"), false, ref projection);
                    
                    // Check for OpenGL errors after setting uniforms
                    error = GL.GetError();
                    if (error != ErrorCode.NoError)
                    {
                        Console.WriteLine($"OpenGL error after setting uniforms: {error}");
                    }
                }
                catch (Exception uniformEx)
                {
                    Console.WriteLine($"Error setting shader uniforms: {uniformEx.Message}");
                    return;
                }
                
                try
                {
                    // Use larger point size to fill gaps between points
                    GL.PointSize(1.5f);
                    GL.BindVertexArray(vertexArrayObject);
                    
                    // Check if we have a reasonable number of indices
                    if (visibleIndices.Count > 0 && visibleIndices.Count < 1000000)
                    {
                        GL.DrawElements(PrimitiveType.Points, visibleIndices.Count, DrawElementsType.UnsignedInt, 0);
                    }
                    else
                    {
                        Console.WriteLine($"Suspicious index count: {visibleIndices.Count}, skipping render");
                    }
                    
                    // Check for errors after rendering
                    error = GL.GetError();
                    if (error != ErrorCode.NoError)
                    {
                        Console.WriteLine($"OpenGL error after drawing: {error}");
                    }
                }
                catch (Exception renderEx)
                {
                    Console.WriteLine($"Error during rendering: {renderEx.Message}");
                    consecutiveRenderFailures++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Critical error in Render: {ex.Message}");
                consecutiveRenderFailures++;
            }
        }

        /// <summary>
        /// Extracts a 2D slice through the 3D volume with optimized memory usage
        /// </summary>
        public Bitmap ExtractSlice(float xPosition)
        {
            // Convert normalized position to slice row
            int sliceRow = (int)((xPosition + 0.5f) * sliceWidth);
            
            // Calculate dimensions for the slice with reduced resolution for better performance
            int width = Math.Min(512, totalSlices);  // Limit width for performance
            int height = Math.Min(512, sliceHeight); // Limit height for performance
            float scaleX = (float)width / totalSlices;
            float scaleY = (float)height / sliceHeight;
            
            // Create an empty bitmap with smaller dimensions
            var bitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            
            // Start with black background
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                g.Clear(Color.Black);
            }
            
            // Use direct bitmap access for faster rendering
            BitmapData bitmapData = null;
            try
            {
                bitmapData = bitmap.LockBits(
                    new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                    ImageLockMode.WriteOnly,
                    bitmap.PixelFormat);
                
                int stride = bitmapData.Stride;
                int bytesPerPixel = System.Drawing.Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
                byte[] pixelData = new byte[stride * bitmap.Height];
                
                // Process only points at the correct row
                foreach (var entry in sliceMap)
                {
                    // Only use points at the correct x position (row)
                    if (entry.Key.row == sliceRow)
                    {
                        int z = entry.Key.z;
                        
                        foreach (var point in entry.Value)
                        {
                            int col = point.Key;
                            float intensity = point.Value;
                            
                            // Scale coordinates to the smaller bitmap
                            int scaledZ = (int)(z * scaleX);
                            int scaledCol = (int)(col * scaleY);
                            
                            // Check bounds
                            if (scaledZ >= 0 && scaledZ < bitmap.Width && scaledCol >= 0 && scaledCol < bitmap.Height)
                            {
                                // Calculate position in the byte array
                                int position = scaledCol * stride + scaledZ * bytesPerPixel;
                                
                                // Set color (BGR format for 24bpp)
                                byte colorValue = (byte)(intensity * 255);
                                if (position + 2 < pixelData.Length)
                                {
                                    pixelData[position] = colorValue;     // B
                                    pixelData[position + 1] = colorValue; // G
                                    pixelData[position + 2] = colorValue; // R
                                }
                            }
                        }
                    }
                }
                
                // Copy the data back to the bitmap
                System.Runtime.InteropServices.Marshal.Copy(pixelData, 0, bitmapData.Scan0, pixelData.Length);
            }
            finally
            {
                // Make sure to unlock the bitmap
                if (bitmapData != null)
                {
                    bitmap.UnlockBits(bitmapData);
                }
            }
            
            // Collect garbage after the intensive operation
            GC.Collect(0, GCCollectionMode.Optimized);
            
            // Return the optimized bitmap (lower resolution but much faster)
            
            return bitmap;
        }

        /// <summary>
        /// Clean up OpenGL resources and free memory
        /// </summary>
        public void Dispose()
        {
            // Clean up OpenGL resources
            if (isInitialized)
            {
                try 
                {
                    GL.DeleteBuffer(vertexBufferObject[0]);
                    GL.DeleteBuffer(colorBufferObject[0]);
                    GL.DeleteBuffer(elementBufferObject[0]);
                    GL.DeleteVertexArray(vertexArrayObject);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error disposing OpenGL resources: {ex.Message}");
                    // Continue with other cleanup even if OpenGL cleanup fails
                }
            }
            
            try
            {
                // Clear all data
                visibleVertices.Clear();
                visibleColors.Clear();
                visibleIndices.Clear();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error clearing vertex data: {ex.Message}");
            }
            
            try
            {
                // Clear the slice map which can consume significant memory
                sliceMap.Clear();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error clearing slice map: {ex.Message}");
            }
            
            // Free references
            dicomDisplayManager = null;
            
            // Force aggressive garbage collection
            for (int i = 0; i < 3; i++)
            {
                GC.Collect(2, GCCollectionMode.Forced, true);
                GC.WaitForPendingFinalizers();
            }
            
            Console.WriteLine("3D renderer disposed - memory cleaned up");
        }
    }
}
