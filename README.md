# VTK Viewer with Seismic Support

A simple GUI app for viewing VTK files and seismic data using Python.

## What it does

- Opens VTK files (.vtk, .vtp, .vtu, .vti, .vtr, .vts)
- Can load seismic data (.segy, .sgy files)
- Also supports some 3D formats like .obj, .ply, .stl
- Shows everything in a 3D viewer you can rotate and zoom

## Setup

You need Python 3.7+ and these packages:

```bash
pip install -r requirements.txt
```

Or install them manually:
```bash
pip install PySide6 vtk numpy segyio obspy scipy matplotlib
```

## How to use it

Run the app:
```bash
python main.py
```

Or open a file directly:
```bash
python main.py your_file.vtk
```

### Controls
- Left click and drag to rotate
- Right click and drag to zoom
- Middle click and drag to pan
- Press 'R' to reset the camera
- Use Ctrl+O to open files

## File structure

```
├── main.py              # Start here
├── src/
│   ├── main_window.py   # Main GUI
│   ├── vtk_viewer_widget.py  # 3D viewer
│   └── seismic_converter.py  # Seismic tools
└── examples/            # Sample files
```

## Common problems

If you get import errors, make sure you installed all the packages:
- `pip install vtk` for VTK support
- `pip install PySide6` for the GUI
- `pip install segyio` for seismic files

On Linux you might need: `sudo apt-get install python3-vtk9 libgl1-mesa-glx`

That's it! Open an issue if something doesn't work. 