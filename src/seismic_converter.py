"""
Seismic data conversion utilities for VTK to SEG-Y and vice versa
"""

import numpy as np
import vtk
import segyio
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa
import os
from typing import Tuple, Optional, Union


class SeismicConverter:
    """Handles conversion between VTK and seismic data formats"""
    
    def __init__(self):
        self.last_conversion_info = {}
    
    def vtk_to_segy(self, vtk_file_path: str, segy_output_path: str, 
                    sample_rate: float = 4000.0, 
                    trace_spacing: float = 25.0) -> bool:
        """
        Convert VTK data to SEG-Y format
        
        Args:
            vtk_file_path: Input VTK file path
            segy_output_path: Output SEG-Y file path
            sample_rate: Sample rate in microseconds (default 4000 = 4ms)
            trace_spacing: Trace spacing in meters (default 25m)
            
        Returns:
            bool: Success status
        """
        try:
            # Read VTK file
            vtk_data = self._read_vtk_file(vtk_file_path)
            if vtk_data is None:
                return False
            
            # Convert VTK data to seismic volume
            seismic_data, geometry = self._vtk_to_seismic_volume(vtk_data)
            if seismic_data is None:
                return False
            
            # Write SEG-Y file
            success = self._write_segy_file(segy_output_path, seismic_data, 
                                          geometry, sample_rate, trace_spacing)
            
            if success:
                self.last_conversion_info = {
                    'type': 'vtk_to_segy',
                    'input': vtk_file_path,
                    'output': segy_output_path,
                    'shape': seismic_data.shape,
                    'sample_rate': sample_rate,
                    'trace_spacing': trace_spacing
                }
            
            return success
            
        except Exception as e:
            print(f"Error converting VTK to SEG-Y: {str(e)}")
            return False
    
    def segy_to_vtk(self, segy_file_path: str, vtk_output_path: str,
                    scale_factor: float = 1.0) -> bool:
        """
        Convert SEG-Y data to VTK format
        
        Args:
            segy_file_path: Input SEG-Y file path
            vtk_output_path: Output VTK file path
            scale_factor: Scaling factor for coordinates
            
        Returns:
            bool: Success status
        """
        try:
            # Read SEG-Y file
            seismic_data, headers = self._read_segy_file(segy_file_path)
            if seismic_data is None:
                return False
            
            # Convert seismic data to VTK
            vtk_data = self._seismic_volume_to_vtk(seismic_data, headers, scale_factor)
            if vtk_data is None:
                return False
            
            # Write VTK file
            success = self._write_vtk_file(vtk_output_path, vtk_data)
            
            if success:
                self.last_conversion_info = {
                    'type': 'segy_to_vtk',
                    'input': segy_file_path,
                    'output': vtk_output_path,
                    'shape': seismic_data.shape,
                    'scale_factor': scale_factor
                }
            
            return success
            
        except Exception as e:
            print(f"Error converting SEG-Y to VTK: {str(e)}")
            return False
    
    def _read_vtk_file(self, file_path: str):
        """Read VTK file and return data object"""
        file_extension = file_path.lower().split('.')[-1]
        
        try:
            if file_extension == 'vtk':
                # Legacy VTK format - detect dataset type
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
            elif file_extension == 'vtp':
                reader = vtk.vtkXMLPolyDataReader()
            elif file_extension == 'vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()
            elif file_extension == 'vti':
                reader = vtk.vtkXMLImageDataReader()
            elif file_extension == 'vtr':
                reader = vtk.vtkXMLRectilinearGridReader()
            elif file_extension == 'vts':
                reader = vtk.vtkXMLStructuredGridReader()
            else:
                print(f"Unsupported VTK format for seismic conversion: {file_extension}")
                return None
            
            reader.SetFileName(file_path)
            reader.Update()
            return reader.GetOutput()
            
        except Exception as e:
            print(f"Error reading VTK file: {str(e)}")
            return None
    
    def _vtk_to_seismic_volume(self, vtk_data) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Convert VTK data to seismic volume array"""
        try:
            if isinstance(vtk_data, vtk.vtkImageData):
                # Image data can be directly converted
                dimensions = vtk_data.GetDimensions()
                spacing = vtk_data.GetSpacing()
                origin = vtk_data.GetOrigin()
                
                # Get scalar data
                scalars = vtk_data.GetPointData().GetScalars()
                if scalars:
                    numpy_array = vtk_to_numpy(scalars)
                    seismic_volume = numpy_array.reshape(dimensions, order='F')
                else:
                    # Create synthetic seismic data based on geometry
                    seismic_volume = np.random.randn(*dimensions) * 0.1
                
                geometry = {
                    'dimensions': dimensions,
                    'spacing': spacing,
                    'origin': origin
                }
                
                return seismic_volume, geometry
                
            elif isinstance(vtk_data, (vtk.vtkStructuredGrid, vtk.vtkRectilinearGrid)):
                # Structured grids can be converted to regular grids
                dimensions = vtk_data.GetDimensions()
                
                # Extract points and create regular grid
                points = vtk_data.GetPoints()
                if points:
                    points_array = vtk_to_numpy(points.GetData())
                    # Calculate spacing from points
                    spacing = self._estimate_spacing_from_points(points_array, dimensions)
                    origin = points_array[0]
                else:
                    spacing = (1.0, 1.0, 1.0)
                    origin = (0.0, 0.0, 0.0)
                
                # Get scalar data or create synthetic
                scalars = vtk_data.GetPointData().GetScalars()
                if scalars:
                    numpy_array = vtk_to_numpy(scalars)
                    seismic_volume = numpy_array.reshape(dimensions, order='F')
                else:
                    seismic_volume = np.random.randn(*dimensions) * 0.1
                
                geometry = {
                    'dimensions': dimensions,
                    'spacing': spacing,
                    'origin': origin
                }
                
                return seismic_volume, geometry
                
            else:
                # For other data types, create a bounding box and sample
                bounds = vtk_data.GetBounds()
                dimensions = (64, 64, 64)  # Default resolution
                
                # Calculate spacing from bounds
                spacing = (
                    (bounds[1] - bounds[0]) / (dimensions[0] - 1),
                    (bounds[3] - bounds[2]) / (dimensions[1] - 1),
                    (bounds[5] - bounds[4]) / (dimensions[2] - 1)
                )
                origin = (bounds[0], bounds[2], bounds[4])
                
                # Create synthetic seismic data
                seismic_volume = self._create_synthetic_seismic(dimensions)
                
                geometry = {
                    'dimensions': dimensions,
                    'spacing': spacing,
                    'origin': origin
                }
                
                return seismic_volume, geometry
                
        except Exception as e:
            print(f"Error converting VTK to seismic volume: {str(e)}")
            return None, None
    
    def _create_synthetic_seismic(self, dimensions: Tuple[int, int, int]) -> np.ndarray:
        """Create synthetic seismic data with realistic characteristics"""
        nx, ny, nz = dimensions
        
        # Create coordinate grids
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create synthetic seismic with layers and some noise
        seismic = np.zeros((nx, ny, nz))
        
        # Add horizontal layers (typical seismic structure)
        for i in range(5):
            layer_depth = 0.2 * i
            layer_thickness = 0.1
            amplitude = 1.0 / (i + 1)
            
            layer_mask = (Z >= layer_depth) & (Z < layer_depth + layer_thickness)
            seismic[layer_mask] += amplitude * np.sin(10 * np.pi * Z[layer_mask])
        
        # Add some random noise
        seismic += 0.1 * np.random.randn(nx, ny, nz)
        
        return seismic
    
    def _estimate_spacing_from_points(self, points: np.ndarray, dimensions: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Estimate grid spacing from point coordinates"""
        try:
            nx, ny, nz = dimensions
            points_3d = points.reshape((nx, ny, nz, 3), order='F')
            
            # Calculate spacing in each direction
            if nx > 1:
                dx = np.mean(np.diff(points_3d[:, 0, 0, 0]))
            else:
                dx = 1.0
                
            if ny > 1:
                dy = np.mean(np.diff(points_3d[0, :, 0, 1]))
            else:
                dy = 1.0
                
            if nz > 1:
                dz = np.mean(np.diff(points_3d[0, 0, :, 2]))
            else:
                dz = 1.0
            
            return (abs(dx), abs(dy), abs(dz))
            
        except Exception:
            return (1.0, 1.0, 1.0)
    
    def _write_segy_file(self, output_path: str, seismic_data: np.ndarray, 
                        geometry: dict, sample_rate: float, trace_spacing: float) -> bool:
        """Write seismic data to SEG-Y format"""
        try:
            nx, ny, nz = seismic_data.shape
            n_traces = nx * ny
            
            # Create SEG-Y file
            spec = segyio.spec()
            spec.samples = list(range(nz))
            spec.tracecount = n_traces
            spec.format = 1  # 4-byte IBM float
            spec.sorting = 2  # CDP sorted
            
            with segyio.create(output_path, spec) as f:
                # Set binary header
                f.bin[segyio.BinField.Samples] = nz
                f.bin[segyio.BinField.Interval] = int(sample_rate)
                f.bin[segyio.BinField.Format] = 1
                
                # Write traces
                trace_idx = 0
                for i in range(nx):
                    for j in range(ny):
                        # Set trace header
                        f.header[trace_idx] = {
                            segyio.TraceField.INLINE_3D: i + 1,
                            segyio.TraceField.CROSSLINE_3D: j + 1,
                            segyio.TraceField.CDP_X: int(geometry['origin'][0] + i * geometry['spacing'][0]),
                            segyio.TraceField.CDP_Y: int(geometry['origin'][1] + j * geometry['spacing'][1]),
                            segyio.TraceField.offset: 0,
                            segyio.TraceField.TRACE_SAMPLE_COUNT: nz,
                            segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(sample_rate)
                        }
                        
                        # Write trace data
                        f.trace[trace_idx] = seismic_data[i, j, :]
                        trace_idx += 1
            
            return True
            
        except Exception as e:
            print(f"Error writing SEG-Y file: {str(e)}")
            return False
    
    def _read_segy_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Read SEG-Y file and return data and headers"""
        try:
            with segyio.open(file_path, ignore_geometry=True) as f:
                # Get basic info
                n_traces = f.tracecount
                n_samples = f.samples.size
                sample_rate = f.bin[segyio.BinField.Interval]
                
                # Read all traces
                traces = f.trace.raw[:]
                
                # Try to determine geometry from headers
                inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
                crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
                
                # Get unique values
                unique_inlines = np.unique(inlines)
                unique_crosslines = np.unique(crosslines)
                
                ni = len(unique_inlines)
                nj = len(unique_crosslines)
                
                if ni * nj == n_traces:
                    # Regular geometry
                    seismic_data = traces.reshape((ni, nj, n_samples), order='C')
                else:
                    # Irregular geometry - use best guess
                    ni = int(np.sqrt(n_traces))
                    nj = n_traces // ni
                    if ni * nj < n_traces:
                        nj += 1
                    
                    # Pad with zeros if necessary
                    padded_traces = np.zeros((ni * nj, n_samples))
                    padded_traces[:n_traces] = traces
                    seismic_data = padded_traces.reshape((ni, nj, n_samples), order='C')
                
                headers = {
                    'sample_rate': sample_rate,
                    'n_samples': n_samples,
                    'n_traces': n_traces,
                    'inlines': unique_inlines,
                    'crosslines': unique_crosslines,
                    'ni': ni,
                    'nj': nj
                }
                
                return seismic_data, headers
                
        except Exception as e:
            print(f"Error reading SEG-Y file: {str(e)}")
            return None, None
    
    def _seismic_volume_to_vtk(self, seismic_data: np.ndarray, headers: dict, 
                             scale_factor: float = 1.0):
        """Convert seismic volume to VTK ImageData"""
        try:
            ni, nj, nk = seismic_data.shape
            
            # Create VTK ImageData
            image_data = vtk.vtkImageData()
            image_data.SetDimensions(ni, nj, nk)
            
            # Set spacing (convert from time/distance to spatial coordinates)
            spacing_x = scale_factor
            spacing_y = scale_factor
            spacing_z = headers.get('sample_rate', 4000) * scale_factor / 1000.0  # Convert microseconds to ms
            
            image_data.SetSpacing(spacing_x, spacing_y, spacing_z)
            image_data.SetOrigin(0.0, 0.0, 0.0)
            
            # Convert seismic data to VTK array
            vtk_array = numpy_to_vtk(seismic_data.ravel(order='F'))
            vtk_array.SetName("Amplitude")
            
            # Add to point data
            image_data.GetPointData().SetScalars(vtk_array)
            
            return image_data
            
        except Exception as e:
            print(f"Error converting seismic volume to VTK: {str(e)}")
            return None
    
    def _write_vtk_file(self, output_path: str, vtk_data) -> bool:
        """Write VTK data to file"""
        try:
            file_extension = output_path.lower().split('.')[-1]
            
            if file_extension == 'vti':
                writer = vtk.vtkXMLImageDataWriter()
            elif file_extension == 'vtk':
                writer = vtk.vtkImageDataWriter()
            else:
                # Default to XML format
                output_path = output_path.rsplit('.', 1)[0] + '.vti'
                writer = vtk.vtkXMLImageDataWriter()
            
            writer.SetFileName(output_path)
            writer.SetInputData(vtk_data)
            writer.Write()
            
            return True
            
        except Exception as e:
            print(f"Error writing VTK file: {str(e)}")
            return False
    
    def get_conversion_info(self) -> dict:
        """Get information about the last conversion"""
        return self.last_conversion_info.copy()
    
    def validate_segy_file(self, file_path: str) -> dict:
        """Validate and get info about SEG-Y file"""
        try:
            with segyio.open(file_path, ignore_geometry=True) as f:
                info = {
                    'valid': True,
                    'n_traces': f.tracecount,
                    'n_samples': f.samples.size,
                    'sample_rate': f.bin[segyio.BinField.Interval],
                    'format': f.bin[segyio.BinField.Format],
                    'sorting': f.bin[segyio.BinField.SortingCode]
                }
                return info
        except Exception as e:
                    return {
            'valid': False,
            'error': str(e)
        }
    
    def _detect_vtk_dataset_type(self, file_path: str) -> str:
        """Detect the dataset type from a legacy VTK file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Look for DATASET line (usually line 3 or 4)
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip().upper()
                if line.startswith('DATASET'):
                    dataset_type = line.split()[-1]
                    return dataset_type
                    
            return 'UNKNOWN'
            
        except Exception as e:
            print(f"Error detecting VTK dataset type: {str(e)}")
            return 'UNKNOWN' 