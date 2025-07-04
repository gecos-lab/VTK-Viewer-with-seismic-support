"""
VTK Viewer Widget for PySide6 integration
"""
import vtk
import os
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QProgressDialog
from PySide6.QtCore import Qt, QTimer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from .seismic_converter import SeismicConverter


class VTKViewerWidget(QWidget):
    """Widget that embeds a VTK render window into PySide6"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.seismic_converter = SeismicConverter()
        self.setup_ui()
        self.setup_vtk()
        
    def setup_ui(self):
        """Initialize the Qt UI components"""
        layout = QVBoxLayout()
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        self.setLayout(layout)
        
    def setup_vtk(self):
        """Initialize VTK rendering components"""
        # Create renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray background
        
        # Add renderer to render window
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        # Create interactor style for mouse interaction
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_widget.SetInteractorStyle(self.interactor_style)
        
        # Initialize the interactor
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        
    def clear_scene(self):
        """Remove all actors from the scene"""
        self.renderer.RemoveAllViewProps()
        self.vtk_widget.GetRenderWindow().Render()
        
    def add_actor(self, actor):
        """Add an actor to the scene"""
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        
    def load_vtk_file(self, file_path):
        """Load and display a VTK file"""
        try:
            # Determine file type first
            file_extension = file_path.lower().split('.')[-1]
            
            # Check file size for large files
            file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
            if file_size > 2.0:
                print(f"Large file detected: {file_size:.2f} GB")
                return self._load_large_vtk_file(file_path, file_extension)
            
            # Clear existing scene
            self.clear_scene()
            
            # Use file extension to determine reader type
            
            if file_extension == 'vtk':
                # Legacy VTK format - need to determine dataset type first
                return self._load_legacy_vtk_file(file_path)
            elif file_extension == 'vtp':
                # VTK XML PolyData format
                reader = vtk.vtkXMLPolyDataReader()
            elif file_extension == 'vtu':
                # VTK XML UnstructuredGrid format
                reader = vtk.vtkXMLUnstructuredGridReader()
            elif file_extension == 'vti':
                # VTK XML ImageData format
                reader = vtk.vtkXMLImageDataReader()
            elif file_extension == 'vtr':
                # VTK XML RectilinearGrid format
                reader = vtk.vtkXMLRectilinearGridReader()
            elif file_extension == 'vts':
                # VTK XML StructuredGrid format
                reader = vtk.vtkXMLStructuredGridReader()
            elif file_extension == 'obj':
                # Wavefront OBJ format
                reader = vtk.vtkOBJReader()
            elif file_extension == 'ply':
                # PLY format
                reader = vtk.vtkPLYReader()
            elif file_extension == 'stl':
                # STL format
                reader = vtk.vtkSTLReader()
            elif file_extension in ['segy', 'sgy']:
                # SEG-Y format - convert to VTK first
                return self.load_segy_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Read the file
            reader.SetFileName(file_path)
            reader.Update()
            
            # Get the data
            data = reader.GetOutput()
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            
            # Handle different data types
            if isinstance(data, vtk.vtkPolyData):
                mapper.SetInputData(data)
            elif isinstance(data, vtk.vtkUnstructuredGrid):
                # Convert unstructured grid to polydata
                geometry_filter = vtk.vtkGeometryFilter()
                geometry_filter.SetInputData(data)
                geometry_filter.Update()
                mapper.SetInputConnection(geometry_filter.GetOutputPort())
            elif isinstance(data, vtk.vtkImageData):
                # Create contour for image data
                contour = vtk.vtkContourFilter()
                contour.SetInputData(data)
                contour.SetValue(0, data.GetScalarRange()[1] * 0.5)
                contour.Update()
                mapper.SetInputConnection(contour.GetOutputPort())
            else:
                # Try to convert to polydata
                geometry_filter = vtk.vtkGeometryFilter()
                geometry_filter.SetInputData(data)
                geometry_filter.Update()
                mapper.SetInputConnection(geometry_filter.GetOutputPort())
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Set some basic properties
            actor.GetProperty().SetColor(0.8, 0.8, 0.9)  # Light blue
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(30)
            
            # Add to scene
            self.add_actor(actor)
            
            return True
            
        except Exception as e:
            print(f"Error loading VTK file: {str(e)}")
            return False
    
    def reset_camera(self):
        """Reset camera to show entire scene"""
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    def set_background_color(self, r, g, b):
        """Set background color (values between 0 and 1)"""
        self.renderer.SetBackground(r, g, b)
        self.vtk_widget.GetRenderWindow().Render()
    
    def load_segy_file(self, file_path):
        """Load and display a SEG-Y file"""
        try:
            # Clear existing scene
            self.clear_scene()
            
            # Read SEG-Y file using seismic converter
            seismic_data, headers = self.seismic_converter._read_segy_file(file_path)
            if seismic_data is None:
                return False
            
            # Convert to VTK ImageData
            vtk_data = self.seismic_converter._seismic_volume_to_vtk(seismic_data, headers)
            if vtk_data is None:
                return False
            
            # Create visualization for seismic data
            success = self._visualize_seismic_data(vtk_data, headers)
            
            return success
            
        except Exception as e:
            print(f"Error loading SEG-Y file: {str(e)}")
            return False
    
    def _visualize_seismic_data(self, vtk_data, headers):
        """Create appropriate visualization for seismic data"""
        try:
            # Get data dimensions
            dimensions = vtk_data.GetDimensions()
            
            # Create slice planes for seismic visualization
            self._create_seismic_slices(vtk_data, dimensions)
            
            # Optionally create volume rendering
            if dimensions[0] * dimensions[1] * dimensions[2] < 1000000:  # Only for smaller volumes
                self._create_volume_rendering(vtk_data)
            
            # Reset camera to fit the data
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            return True
            
        except Exception as e:
            print(f"Error visualizing seismic data: {str(e)}")
            return False
    
    def _create_seismic_slices(self, vtk_data, dimensions):
        """Create slice planes through seismic volume"""
        try:
            # Create three orthogonal slice planes
            
            # XY slice (horizontal slice - time slice)
            xy_plane = vtk.vtkImageSlice()
            xy_mapper = vtk.vtkImageSliceMapper()
            xy_mapper.SetInputData(vtk_data)
            xy_mapper.SetOrientation(2)  # Z direction
            xy_mapper.SetSliceNumber(dimensions[2] // 4)  # Quarter depth
            xy_plane.SetMapper(xy_mapper)
            
            # Set color map for seismic data
            lut = self._create_seismic_colormap()
            xy_plane.GetProperty().SetLookupTable(lut)
            xy_plane.GetProperty().UseLookupTableScalarRangeOn()
            
            self.renderer.AddActor(xy_plane)
            
            # XZ slice (inline slice)
            xz_plane = vtk.vtkImageSlice()
            xz_mapper = vtk.vtkImageSliceMapper()
            xz_mapper.SetInputData(vtk_data)
            xz_mapper.SetOrientation(1)  # Y direction
            xz_mapper.SetSliceNumber(dimensions[1] // 2)  # Middle crossline
            xz_plane.SetMapper(xz_mapper)
            xz_plane.GetProperty().SetLookupTable(lut)
            xz_plane.GetProperty().UseLookupTableScalarRangeOn()
            
            self.renderer.AddActor(xz_plane)
            
            # YZ slice (crossline slice)
            yz_plane = vtk.vtkImageSlice()
            yz_mapper = vtk.vtkImageSliceMapper()
            yz_mapper.SetInputData(vtk_data)
            yz_mapper.SetOrientation(0)  # X direction
            yz_mapper.SetSliceNumber(dimensions[0] // 2)  # Middle inline
            yz_plane.SetMapper(yz_mapper)
            yz_plane.GetProperty().SetLookupTable(lut)
            yz_plane.GetProperty().UseLookupTableScalarRangeOn()
            
            self.renderer.AddActor(yz_plane)
            
        except Exception as e:
            print(f"Error creating seismic slices: {str(e)}")
    
    def _create_seismic_colormap(self):
        """Create a colormap suitable for seismic data"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        lut.SetHueRange(0.667, 0.0)  # Blue to red
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetTableRange(-1.0, 1.0)  # Typical seismic amplitude range
        lut.Build()
        return lut
    
    def _create_volume_rendering(self, vtk_data):
        """Create volume rendering for seismic data"""
        try:
            # Create volume mapper
            volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
            volume_mapper.SetInputData(vtk_data)
            
            # Create volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetInterpolationTypeToLinear()
            volume_property.ShadeOn()
            volume_property.SetAmbient(0.4)
            volume_property.SetDiffuse(0.6)
            volume_property.SetSpecular(0.2)
            
            # Create color and opacity transfer functions
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(-1.0, 0.0, 0.0, 1.0)  # Blue for negative
            color_func.AddRGBPoint(0.0, 1.0, 1.0, 1.0)   # White for zero
            color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)   # Red for positive
            
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(-1.0, 0.0)
            opacity_func.AddPoint(-0.5, 0.1)
            opacity_func.AddPoint(0.0, 0.0)
            opacity_func.AddPoint(0.5, 0.1)
            opacity_func.AddPoint(1.0, 0.0)
            
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)
            
            # Create volume
            volume = vtk.vtkVolume()
            volume.SetMapper(volume_mapper)
            volume.SetProperty(volume_property)
            
            self.renderer.AddVolume(volume)
            
        except Exception as e:
            print(f"Error creating volume rendering: {str(e)}")
    
    def convert_vtk_to_segy(self, vtk_file_path, segy_output_path, 
                           sample_rate=4000.0, trace_spacing=25.0):
        """Convert VTK file to SEG-Y format"""
        try:
            return self.seismic_converter.vtk_to_segy(
                vtk_file_path, segy_output_path, sample_rate, trace_spacing
            )
        except Exception as e:
            print(f"Error converting VTK to SEG-Y: {str(e)}")
            return False
    
    def convert_segy_to_vtk(self, segy_file_path, vtk_output_path, scale_factor=1.0):
        """Convert SEG-Y file to VTK format"""
        try:
            return self.seismic_converter.segy_to_vtk(
                segy_file_path, vtk_output_path, scale_factor
            )
        except Exception as e:
            print(f"Error converting SEG-Y to VTK: {str(e)}")
            return False
    
    def get_conversion_info(self):
        """Get information about the last conversion"""
        return self.seismic_converter.get_conversion_info()
    
    def validate_segy_file(self, file_path):
        """Validate SEG-Y file and get information"""
        return self.seismic_converter.validate_segy_file(file_path)
    
    def _load_legacy_vtk_file(self, file_path):
        """Load legacy VTK file by detecting its dataset type"""
        try:
            # Check file size first
            file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
            print(f"File size: {file_size:.2f} GB")
            
            if file_size > 2.0:  # Files larger than 2GB need special handling
                return self._load_large_vtk_file(file_path)
            
            # Read the first few lines to determine dataset type
            dataset_type = self._detect_vtk_dataset_type(file_path)
            
            if dataset_type == 'POLYDATA':
                reader = vtk.vtkPolyDataReader()
            elif dataset_type == 'STRUCTURED_POINTS':
                reader = vtk.vtkStructuredPointsReader()
            elif dataset_type == 'STRUCTURED_GRID':
                reader = vtk.vtkStructuredGridReader()
            elif dataset_type == 'UNSTRUCTURED_GRID':
                reader = vtk.vtkUnstructuredGridReader()
            elif dataset_type == 'RECTILINEAR_GRID':
                reader = vtk.vtkRectilinearGridReader()
            else:
                # Default to generic data reader
                reader = vtk.vtkDataSetReader()
            
            reader.SetFileName(file_path)
            reader.Update()
            
            # Get the data
            data = reader.GetOutput()
            
            # Create mapper and actor based on data type
            return self._create_actor_from_data(data)
            
        except Exception as e:
            print(f"Error loading legacy VTK file: {str(e)}")
            return False
    
    def _detect_vtk_dataset_type(self, file_path):
        """Detect the dataset type from a legacy VTK file"""
        try:
            # Try to read as text first, with different encodings
            encodings = ['utf-8', 'ascii', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                        
                    # Look for DATASET line (usually line 3 or 4)
                    for line in lines[:10]:  # Check first 10 lines
                        line = line.strip().upper()
                        if line.startswith('DATASET'):
                            dataset_type = line.split()[-1]
                            return dataset_type
                    break  # If we got here, file was readable but no DATASET found
                    
                except UnicodeDecodeError:
                    continue  # Try next encoding
                    
            # If we couldn't read as text, it might be binary
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(1024).decode('ascii', errors='ignore')
                    
                for line in header.split('\n')[:10]:
                    line = line.strip().upper()
                    if line.startswith('DATASET'):
                        dataset_type = line.split()[-1]
                        return dataset_type
            except Exception:
                pass
                    
            return 'UNKNOWN'
            
        except Exception as e:
            print(f"Error detecting VTK dataset type: {str(e)}")
            return 'UNKNOWN'
    
    def _create_actor_from_data(self, data):
        """Create appropriate actor from VTK data object"""
        try:
            # Clear existing scene
            self.clear_scene()
            
            # Handle different data types
            if isinstance(data, vtk.vtkPolyData):
                # PolyData can be directly mapped
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(data)
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.8, 0.8, 0.9)
                actor.GetProperty().SetSpecular(0.3)
                actor.GetProperty().SetSpecularPower(30)
                
                self.add_actor(actor)
                
            elif isinstance(data, vtk.vtkImageData):
                # ImageData (structured points) - create slice visualization
                self._visualize_image_data(data)
                
            elif isinstance(data, vtk.vtkStructuredGrid):
                # Structured grid - extract surface
                geometry_filter = vtk.vtkGeometryFilter()
                geometry_filter.SetInputData(data)
                geometry_filter.Update()
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(geometry_filter.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.8, 0.8, 0.9)
                
                self.add_actor(actor)
                
            elif isinstance(data, vtk.vtkUnstructuredGrid):
                # Unstructured grid - extract surface
                geometry_filter = vtk.vtkGeometryFilter()
                geometry_filter.SetInputData(data)
                geometry_filter.Update()
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(geometry_filter.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.8, 0.8, 0.9)
                
                self.add_actor(actor)
                
            elif isinstance(data, vtk.vtkRectilinearGrid):
                # Rectilinear grid - extract surface
                geometry_filter = vtk.vtkGeometryFilter()
                geometry_filter.SetInputData(data)
                geometry_filter.Update()
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(geometry_filter.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.8, 0.8, 0.9)
                
                self.add_actor(actor)
                
            else:
                print(f"Unsupported data type: {type(data)}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error creating actor from data: {str(e)}")
            return False
    
    def _visualize_image_data(self, image_data):
        """Create visualization for VTK ImageData (structured points)"""
        try:
            dimensions = image_data.GetDimensions()
            
            # If it's a 3D volume, create slice planes
            if dimensions[2] > 1:
                # Create slice planes similar to seismic data
                self._create_image_slices(image_data, dimensions)
                
                # Optionally create volume rendering for smaller volumes
                if dimensions[0] * dimensions[1] * dimensions[2] < 1000000:
                    self._create_image_volume_rendering(image_data)
            else:
                # 2D image - create a single slice
                image_slice = vtk.vtkImageSlice()
                image_mapper = vtk.vtkImageSliceMapper()
                image_mapper.SetInputData(image_data)
                image_slice.SetMapper(image_mapper)
                
                # Set appropriate color map
                scalar_range = image_data.GetScalarRange()
                lut = vtk.vtkLookupTable()
                lut.SetRange(scalar_range)
                lut.SetNumberOfColors(256)
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.Build()
                
                image_slice.GetProperty().SetLookupTable(lut)
                image_slice.GetProperty().UseLookupTableScalarRangeOn()
                
                self.renderer.AddActor(image_slice)
            
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Error visualizing image data: {str(e)}")
    
    def _create_image_slices(self, image_data, dimensions):
        """Create orthogonal slice planes for 3D image data"""
        try:
            # Get scalar range for color mapping
            scalar_range = image_data.GetScalarRange()
            
            # Create lookup table
            lut = vtk.vtkLookupTable()
            lut.SetRange(scalar_range)
            lut.SetNumberOfColors(256)
            lut.SetHueRange(0.667, 0.0)  # Blue to red
            lut.Build()
            
            # XY slice (axial)
            xy_slice = vtk.vtkImageSlice()
            xy_mapper = vtk.vtkImageSliceMapper()
            xy_mapper.SetInputData(image_data)
            xy_mapper.SetOrientation(2)  # Z direction
            xy_mapper.SetSliceNumber(dimensions[2] // 2)  # Middle slice
            xy_slice.SetMapper(xy_mapper)
            xy_slice.GetProperty().SetLookupTable(lut)
            xy_slice.GetProperty().UseLookupTableScalarRangeOn()
            self.renderer.AddActor(xy_slice)
            
            # XZ slice (coronal)
            if dimensions[1] > 1:
                xz_slice = vtk.vtkImageSlice()
                xz_mapper = vtk.vtkImageSliceMapper()
                xz_mapper.SetInputData(image_data)
                xz_mapper.SetOrientation(1)  # Y direction
                xz_mapper.SetSliceNumber(dimensions[1] // 2)
                xz_slice.SetMapper(xz_mapper)
                xz_slice.GetProperty().SetLookupTable(lut)
                xz_slice.GetProperty().UseLookupTableScalarRangeOn()
                self.renderer.AddActor(xz_slice)
            
            # YZ slice (sagittal)
            if dimensions[0] > 1:
                yz_slice = vtk.vtkImageSlice()
                yz_mapper = vtk.vtkImageSliceMapper()
                yz_mapper.SetInputData(image_data)
                yz_mapper.SetOrientation(0)  # X direction
                yz_mapper.SetSliceNumber(dimensions[0] // 2)
                yz_slice.SetMapper(yz_mapper)
                yz_slice.GetProperty().SetLookupTable(lut)
                yz_slice.GetProperty().UseLookupTableScalarRangeOn()
                self.renderer.AddActor(yz_slice)
                
        except Exception as e:
            print(f"Error creating image slices: {str(e)}")
    
    def _create_image_volume_rendering(self, image_data):
        """Create volume rendering for 3D image data"""
        try:
            # Create volume mapper
            volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
            volume_mapper.SetInputData(image_data)
            
            # Create volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetInterpolationTypeToLinear()
            volume_property.ShadeOn()
            
            # Get scalar range
            scalar_range = image_data.GetScalarRange()
            
            # Create color transfer function
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 1.0)  # Blue for low values
            color_func.AddRGBPoint((scalar_range[0] + scalar_range[1]) / 2, 1.0, 1.0, 1.0)  # White for middle
            color_func.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)  # Red for high values
            
            # Create opacity transfer function
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(scalar_range[0], 0.0)
            opacity_func.AddPoint(scalar_range[0] + 0.2 * (scalar_range[1] - scalar_range[0]), 0.1)
            opacity_func.AddPoint(scalar_range[1] - 0.2 * (scalar_range[1] - scalar_range[0]), 0.1)
            opacity_func.AddPoint(scalar_range[1], 0.0)
            
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)
            
            # Create volume
            volume = vtk.vtkVolume()
            volume.SetMapper(volume_mapper)
            volume.SetProperty(volume_property)
            
            self.renderer.AddVolume(volume)
            
        except Exception as e:
            print(f"Error creating image volume rendering: {str(e)}")
    
    def _load_large_vtk_file(self, file_path, file_extension):
        """Handle loading of large VTK files with memory optimization"""
        try:
            # Show warning dialog
            reply = QMessageBox.question(
                self.parent(),
                "Large File Warning",
                f"This file is very large ({os.path.getsize(file_path) / (1024**3):.2f} GB).\n\n"
                "Loading may take time and use significant memory.\n"
                "Would you like to:\n\n"
                "• Load full file (may be slow)\n"
                "• Load subsampled version (faster, lower resolution)\n"
                "• Cancel loading",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.No:
                # Load subsampled version
                return self._load_subsampled_vtk_file(file_path, file_extension)
            else:
                # Load full file with streaming
                return self._load_full_large_vtk_file(file_path, file_extension)
                
        except Exception as e:
            print(f"Error handling large VTK file: {str(e)}")
            return False
    
    def _load_subsampled_vtk_file(self, file_path, file_extension):
        """Load a subsampled version of a large VTK file"""
        try:
            print("Loading subsampled version of large file...")
            
            # Create progress dialog
            progress = QProgressDialog("Loading large file (subsampled)...", "Cancel", 0, 100, self.parent())
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Determine data type and reader based on file extension
            if file_extension == 'vti':
                return self._load_subsampled_image_data(file_path, progress, 'vti')
            elif file_extension == 'vtk':
                # Legacy VTK - detect dataset type
                dataset_type = self._detect_vtk_dataset_type(file_path)
                if dataset_type == 'STRUCTURED_POINTS':
                    return self._load_subsampled_image_data(file_path, progress, 'vtk')
                else:
                    return self._load_with_memory_limit(file_path, file_extension, progress)
            else:
                # For other types, try standard loading with memory limits
                return self._load_with_memory_limit(file_path, file_extension, progress)
                
        except Exception as e:
            print(f"Error loading subsampled VTK file: {str(e)}")
            if 'progress' in locals():
                progress.close()
            return False
    
    def _load_subsampled_image_data(self, file_path, progress, file_format):
        """Load subsampled image data (VTI or legacy structured points)"""
        try:
            # Use appropriate reader based on file format
            if file_format == 'vti':
                reader = vtk.vtkXMLImageDataReader()
            else:  # vtk legacy
                reader = vtk.vtkStructuredPointsReader()
            
            reader.SetFileName(file_path)
            
            # For large files, try to read metadata first
            try:
                reader.Update()
            except Exception as read_error:
                print(f"Error reading large file: {read_error}")
                progress.close()
                return False
            
            progress.setValue(30)
            if progress.wasCanceled():
                return False
            
            # Get the full data
            full_data = reader.GetOutput()
            dimensions = full_data.GetDimensions()
            
            print(f"Original dimensions: {dimensions}")
            
            # Calculate subsampling factor to keep memory reasonable
            total_points = dimensions[0] * dimensions[1] * dimensions[2]
            max_points = 10_000_000  # 10M points max
            
            if total_points > max_points:
                subsample_factor = int(np.ceil((total_points / max_points) ** (1/3)))
            else:
                subsample_factor = 1
            
            print(f"Subsampling factor: {subsample_factor}")
            
            progress.setValue(50)
            if progress.wasCanceled():
                return False
            
            # Create subsampling filter
            if subsample_factor > 1:
                subsample = vtk.vtkImageShrink3D()
                subsample.SetInputData(full_data)
                subsample.SetShrinkFactors(subsample_factor, subsample_factor, subsample_factor)
                subsample.Update()
                subsampled_data = subsample.GetOutput()
            else:
                subsampled_data = full_data
            
            progress.setValue(80)
            if progress.wasCanceled():
                return False
            
            # Visualize the subsampled data
            self._visualize_image_data(subsampled_data)
            
            progress.setValue(100)
            progress.close()
            
            print(f"Loaded subsampled data with dimensions: {subsampled_data.GetDimensions()}")
            return True
            
        except Exception as e:
            print(f"Error loading subsampled image data: {str(e)}")
            if progress:
                progress.close()
            return False
    
    def _load_with_memory_limit(self, file_path, file_extension, progress):
        """Load file with memory monitoring"""
        try:
            # Select appropriate reader based on file extension
            if file_extension == 'vtk':
                # Legacy VTK - detect dataset type
                dataset_type = self._detect_vtk_dataset_type(file_path)
                if dataset_type == 'POLYDATA':
                    reader = vtk.vtkPolyDataReader()
                elif dataset_type == 'STRUCTURED_GRID':
                    reader = vtk.vtkStructuredGridReader()
                elif dataset_type == 'UNSTRUCTURED_GRID':
                    reader = vtk.vtkUnstructuredGridReader()
                elif dataset_type == 'RECTILINEAR_GRID':
                    reader = vtk.vtkRectilinearGridReader()
                else:
                    reader = vtk.vtkDataSetReader()
            elif file_extension == 'vtp':
                reader = vtk.vtkXMLPolyDataReader()
            elif file_extension == 'vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()
            elif file_extension == 'vts':
                reader = vtk.vtkXMLStructuredGridReader()
            elif file_extension == 'vtr':
                reader = vtk.vtkXMLRectilinearGridReader()
            else:
                reader = vtk.vtkDataSetReader()
            
            reader.SetFileName(file_path)
            
            progress.setValue(20)
            if progress.wasCanceled():
                return False
            
            # Try to read with error handling
            try:
                reader.Update()
                progress.setValue(60)
                
                if progress.wasCanceled():
                    return False
                
                data = reader.GetOutput()
                
                progress.setValue(80)
                if progress.wasCanceled():
                    return False
                
                # Check if data is too large for direct visualization
                if hasattr(data, 'GetNumberOfPoints'):
                    num_points = data.GetNumberOfPoints()
                    if num_points > 5_000_000:  # 5M points
                        # Decimate the data
                        data = self._decimate_large_data(data)
                
                # Create visualization
                success = self._create_actor_from_data(data)
                
                progress.setValue(100)
                progress.close()
                
                return success
                
            except Exception as read_error:
                print(f"Error reading large file: {str(read_error)}")
                progress.close()
                return False
                
        except Exception as e:
            print(f"Error in memory-limited loading: {str(e)}")
            if progress:
                progress.close()
            return False
    
    def _load_full_large_vtk_file(self, file_path, file_extension):
        """Load full large VTK file with progress monitoring"""
        try:
            print("Loading full large file (this may take several minutes)...")
            
            # Create progress dialog
            progress = QProgressDialog("Loading large file (full resolution)...", "Cancel", 0, 100, self.parent())
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Use appropriate reader based on file extension
            if file_extension == 'vtk':
                # Legacy VTK - detect dataset type
                dataset_type = self._detect_vtk_dataset_type(file_path)
                if dataset_type == 'POLYDATA':
                    reader = vtk.vtkPolyDataReader()
                elif dataset_type == 'STRUCTURED_POINTS':
                    reader = vtk.vtkStructuredPointsReader()
                elif dataset_type == 'STRUCTURED_GRID':
                    reader = vtk.vtkStructuredGridReader()
                elif dataset_type == 'UNSTRUCTURED_GRID':
                    reader = vtk.vtkUnstructuredGridReader()
                elif dataset_type == 'RECTILINEAR_GRID':
                    reader = vtk.vtkRectilinearGridReader()
                else:
                    reader = vtk.vtkDataSetReader()
            elif file_extension == 'vti':
                reader = vtk.vtkXMLImageDataReader()
            elif file_extension == 'vtp':
                reader = vtk.vtkXMLPolyDataReader()
            elif file_extension == 'vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()
            elif file_extension == 'vts':
                reader = vtk.vtkXMLStructuredGridReader()
            elif file_extension == 'vtr':
                reader = vtk.vtkXMLRectilinearGridReader()
            else:
                reader = vtk.vtkDataSetReader()
            
            reader.SetFileName(file_path)
            
            progress.setValue(10)
            if progress.wasCanceled():
                return False
            
            # Set up progress monitoring
            def update_progress():
                if hasattr(reader, 'GetProgress'):
                    prog_val = int(reader.GetProgress() * 70) + 10
                    progress.setValue(prog_val)
                return not progress.wasCanceled()
            
            # Start reading with progress updates
            timer = QTimer()
            timer.timeout.connect(update_progress)
            timer.start(100)  # Update every 100ms
            
            try:
                reader.Update()
                timer.stop()
                
                progress.setValue(80)
                if progress.wasCanceled():
                    return False
                
                data = reader.GetOutput()
                
                progress.setValue(90)
                if progress.wasCanceled():
                    return False
                
                # Create visualization
                success = self._create_actor_from_data(data)
                
                progress.setValue(100)
                progress.close()
                
                return success
                
            except Exception as read_error:
                timer.stop()
                progress.close()
                
                # Show memory error dialog
                QMessageBox.critical(
                    self.parent(),
                    "Memory Error",
                    f"Unable to load the large file:\n{str(read_error)}\n\n"
                    "The file may be too large for available memory.\n"
                    "Try loading a subsampled version instead."
                )
                return False
                
        except Exception as e:
            print(f"Error loading full large VTK file: {str(e)}")
            return False
    
    def _decimate_large_data(self, data):
        """Decimate large datasets to reduce memory usage"""
        try:
            if isinstance(data, vtk.vtkPolyData):
                # Use decimation filter for polygon data
                decimate = vtk.vtkDecimatePro()
                decimate.SetInputData(data)
                decimate.SetTargetReduction(0.7)  # Reduce by 70%
                decimate.PreserveTopologyOn()
                decimate.Update()
                return decimate.GetOutput()
            
            elif isinstance(data, (vtk.vtkStructuredGrid, vtk.vtkUnstructuredGrid)):
                # Extract surface and then decimate
                geometry_filter = vtk.vtkGeometryFilter()
                geometry_filter.SetInputData(data)
                geometry_filter.Update()
                
                decimate = vtk.vtkDecimatePro()
                decimate.SetInputConnection(geometry_filter.GetOutputPort())
                decimate.SetTargetReduction(0.7)
                decimate.Update()
                
                return decimate.GetOutput()
            
            else:
                # Return original data if we can't decimate
                return data
                
        except Exception as e:
            print(f"Error decimating large data: {str(e)}")
            return data 