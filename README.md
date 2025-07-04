# VTK File Viewer with Seismic Support

A comprehensive GUI application for viewing VTK files and working with seismic data, built with PySide6 and VTK.

## Features

- **Multiple Format Support**: Supports various VTK, seismic, and 3D model formats
  - VTK legacy format (.vtk)
  - VTK XML formats (.vtp, .vtu, .vti, .vtr, .vts)
  - Seismic formats (.segy, .sgy)
  - 3D model formats (.obj, .ply, .stl)

- **Interactive 3D Viewer**: 
  - Mouse controls for rotation, zoom, and pan
  - Camera reset functionality
  - Multiple background color options

- **User-Friendly Interface**:
  - Clean and modern GUI with control panel
  - File menu with keyboard shortcuts
  - Status bar for feedback
  - Drag-and-drop file loading (via command line)

- **Seismic Data Processing**:
  - VTK ↔ SEG-Y conversion capabilities
  - Specialized seismic visualization with slice planes
  - Volume rendering for seismic volumes
  - Seismic-optimized colormaps (blue-white-red)
  - Support for inline/crossline/time slice viewing

## Requirements

- Python 3.7 or higher
- PySide6 >= 6.5.0
- VTK >= 9.2.0
- NumPy >= 1.21.0
- segyio >= 1.9.0 (for seismic data support)
- obspy >= 1.3.0 (for advanced seismic processing)
- scipy >= 1.7.0 (for scientific computing)
- matplotlib >= 3.5.0 (for plotting support)

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd vtk-file-viewer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install PySide6 vtk numpy
   ```

## Usage

### Running the Application

1. **Basic usage**:
   ```bash
   python main.py
   ```

2. **Open a file directly**:
   ```bash
   python main.py path/to/your/file.vtk
   # or
   python main.py path/to/your/seismic.segy
   ```

3. **Create test files**:
   ```bash
   # Create sample VTK files
   python examples/create_sample_vtk.py
   
   # Create sample SEG-Y files
   python examples/create_sample_segy.py
   ```

### Using the Application

1. **Loading Files**:
   - Use `File > Open...` menu or `Ctrl+O`
   - Click the "Open VTK File" button in the control panel
   - Or run with a file path as command line argument
   - SEG-Y files are automatically detected and loaded with specialized visualization

2. **Viewing Controls**:
   - **Left click + drag**: Rotate the 3D model
   - **Right click + drag**: Zoom in/out
   - **Middle click + drag**: Pan the view
   - **Reset Camera button**: Fit the model in view
   - **Keyboard shortcut 'R'**: Reset camera

3. **Interface Controls**:
   - **Background colors**: Choose from Dark, Light, or Blue backgrounds
   - **Clear Scene**: Remove all objects from the viewer
   - **File Info**: Shows currently loaded file name

4. **Seismic Conversion**:
   - **VTK → SEG-Y**: Convert VTK files to SEG-Y format with configurable parameters
   - **SEG-Y → VTK**: Convert SEG-Y files to VTK format for 3D visualization
   - Access via `Tools` menu or control panel buttons

5. **Keyboard Shortcuts**:
   - `Ctrl+O`: Open file
   - `Ctrl+Q`: Quit application
   - `R`: Reset camera
   - `C`: Clear scene

## Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| VTK Legacy | .vtk | Original VTK format |
| VTK XML PolyData | .vtp | XML-based polygon data |
| VTK XML UnstructuredGrid | .vtu | XML-based unstructured grid |
| VTK XML ImageData | .vti | XML-based image data |
| VTK XML RectilinearGrid | .vtr | XML-based rectilinear grid |
| VTK XML StructuredGrid | .vts | XML-based structured grid |
| SEG-Y | .segy, .sgy | Seismic data format |
| Wavefront OBJ | .obj | Common 3D model format |
| PLY | .ply | Polygon file format |
| STL | .stl | Stereolithography format |

## Project Structure

```
vtk-file-viewer/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── setup.py                   # Installation script
├── README.md                  # This file
├── src/
│   ├── __init__.py           # Package initialization
│   ├── main_window.py        # Main GUI window
│   ├── vtk_viewer_widget.py  # VTK viewer widget
│   └── seismic_converter.py  # Seismic data conversion utilities
└── examples/
    ├── create_sample_vtk.py   # Generate sample VTK files
    └── create_sample_segy.py  # Generate sample SEG-Y files
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'vtk'"**
   - Solution: Install VTK with `pip install vtk`

2. **"ModuleNotFoundError: No module named 'PySide6'"**
   - Solution: Install PySide6 with `pip install PySide6`

3. **"ModuleNotFoundError: No module named 'segyio'"**
   - Solution: Install segyio with `pip install segyio`

4. **VTK widget not displaying on Linux**
   - Solution: You may need to install additional system packages:
     ```bash
     sudo apt-get install python3-vtk9 libgl1-mesa-glx
     ```

5. **File fails to load**
   - Check that the file format is supported
   - Verify the file is not corrupted
   - Check the console output for detailed error messages
   - For SEG-Y files, ensure they follow standard formatting

6. **SEG-Y conversion issues**
   - Verify input files are valid VTK or SEG-Y formats
   - Check conversion parameters (sample rate, trace spacing)
   - Large files may take time to process

### Performance Tips

- For large files, the initial loading may take some time
- Use the "Clear Scene" button to free memory when switching between large files
- Background rendering continues even when the window is minimized

## Development

### Adding New File Formats

To add support for additional file formats, modify the `load_vtk_file` method in `src/vtk_viewer_widget.py`:

1. Add the file extension to the condition checks
2. Create the appropriate VTK reader
3. Update the file dialog filters in `src/main_window.py`

### Customizing the Interface

- Modify `src/main_window.py` to change the GUI layout
- Add new controls to the `create_control_panel` method
- Customize colors and styling using Qt stylesheets

## License

This project is open source. Feel free to modify and distribute as needed.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 