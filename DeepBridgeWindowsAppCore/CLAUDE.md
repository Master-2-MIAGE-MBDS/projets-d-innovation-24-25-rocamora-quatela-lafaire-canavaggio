# DeepBridgeWindowsAppCore - Development Guide

## Build/Run Commands
- Build: `dotnet build`
- Run: `dotnet run`
- Debug: Build and run in Visual Studio with Debug profile
- Release: `dotnet build -c Release`

## Code Style Guidelines
- **Naming**: PascalCase for classes/methods/properties, camelCase for parameters/variables
- **Imports**: Group System imports first, followed by external libraries, then project namespaces
- **Error Handling**: Use try/catch with specific exceptions, log errors to console
- **Formatting**: Use 4-space indentation, braces on new lines
- **Comments**: XML documentation for public methods/classes, inline comments for complex logic
- **CUDA**: Initialize GPU resources early, always dispose properly
- **DICOM**: Follow EvilDICOM patterns for tag access, handle missing tags with null coalescence
- **Memory Management**: Use 'using' statements for disposable GPU resources
- **Type Safety**: Prefer explicit types over var except for complex declarations
- **Avoid Magic Numbers**: Use constants for window/level presets and other values

## Tech Stack
- .NET 8.0 Windows Application
- ILGPU/CUDA for GPU acceleration
- EvilDICOM for DICOM file handling
- OpenTK for rendering