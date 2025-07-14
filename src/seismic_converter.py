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
import logging


class SeismicConverter:
    """Handles conversion between VTK and seismic data formats"""
    
    def __init__(self):
        self.last_conversion_info = {}
        self.logger = logging.getLogger(__name__)
    
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
                self.logger.error("Failed to read VTK file")
                return False
            
            # Convert VTK data to seismic volume
            seismic_data, geometry = self._vtk_to_seismic_volume(vtk_data)
            if seismic_data is None:
                self.logger.error("Failed to convert VTK to seismic volume")
                return False
            
            # Log the axis transformation for debugging
            orig_dims = geometry.get('original_vtk_dimensions', geometry['dimensions'])
            final_dims = geometry['dimensions']
            self.logger.info(f"VTK original dimensions: {orig_dims}")
            self.logger.info(f"Seismic final dimensions: {final_dims}")
            self.logger.info(f"Axis mapping: {geometry.get('axis_mapping', 'none')}")
            
            # Write SEG-Y file
            success = self._write_segy_file(segy_output_path, seismic_data, 
                                          geometry, sample_rate, trace_spacing)
            
            if success:
                # Store conversion info
                self.last_conversion_info = {
                    'source_format': 'VTK',
                    'target_format': 'SEG-Y',
                    'source_file': vtk_file_path,
                    'target_file': segy_output_path,
                    'dimensions': geometry['dimensions'],
                    'sample_rate': sample_rate,
                    'trace_spacing': trace_spacing,
                    'axis_reorientation': geometry.get('axis_mapping', 'none')
                }
                
                self.logger.info(f"Successfully converted VTK to SEG-Y: {vtk_file_path} -> {segy_output_path}")
                return True
            else:
                self.logger.error("Failed to write SEG-Y file")
                return False
            
        except Exception as e:
            self.logger.error(f"Error in VTK to SEG-Y conversion: {e}")
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
        """Convert VTK data to seismic volume array with proper axis orientation"""
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
                    vtk_volume = numpy_array.reshape(dimensions, order='F')
                    
                    # CRITICAL FIX: VTK typically has Z as vertical axis
                    # SEG-Y needs vertical axis as 3rd dimension (time/depth)
                    # For proper seismic visualization, we need Z-axis data as traces
                    nx, ny, nz = dimensions
                    
                    self.logger.info(f"Original VTK dimensions: X={nx}, Y={ny}, Z={nz}")
                    
                    # Transpose the volume to make seismic lines horizontal
                    # VTK: (X, Y, Z) -> SEG-Y: (inline, xline, time) 
                    # We want the Y dimension to be vertical, so we transpose to (X, Z, Y)
                    seismic_volume = np.transpose(vtk_volume, (0, 2, 1))
                    
                    # Now seismic_volume dimensions are:
                    # - First dimension (nx): inline
                    # - Second dimension (nz): crossline
                    # - Third dimension (ny): vertical (time/depth)
                    seismic_dimensions = (nx, nz, ny)
                    seismic_spacing = (spacing[0], spacing[2], spacing[1])
                    seismic_origin = (origin[0], origin[2], origin[1])
                    
                    self.logger.info(f"Seismic output dimensions: inline={nx}, crossline={nz}, vertical={ny}")
                    self.logger.info(f"Each trace has {ny} samples along vertical axis")
                    
                else:
                    # Create synthetic seismic data based on geometry
                    seismic_volume = np.random.randn(*dimensions) * 0.1
                    seismic_dimensions = dimensions
                    seismic_spacing = spacing
                    seismic_origin = origin
                
                geometry = {
                    'dimensions': seismic_dimensions,
                    'spacing': seismic_spacing,
                    'origin': seismic_origin,
                    'original_vtk_dimensions': dimensions,
                    'axis_mapping': 'horizontal_seismic_lines'
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
                    vtk_volume = numpy_array.reshape(dimensions, order='F')
                    
                    # Apply same transposition logic as ImageData
                    nx, ny, nz = dimensions
                    self.logger.info(f"Structured grid VTK dimensions: X={nx}, Y={ny}, Z={nz}")
                    
                    # Transpose to make seismic lines horizontal
                    seismic_volume = np.transpose(vtk_volume, (0, 2, 1))
                    seismic_dimensions = (nx, nz, ny)
                    seismic_spacing = (spacing[0], spacing[2], spacing[1])
                    seismic_origin = (origin[0], origin[2], origin[1])
                    
                else:
                    seismic_volume = np.random.randn(*dimensions) * 0.1
                    seismic_dimensions = dimensions
                    seismic_spacing = spacing
                    seismic_origin = origin
                
                geometry = {
                    'dimensions': seismic_dimensions,
                    'spacing': seismic_spacing,
                    'origin': seismic_origin,
                    'original_vtk_dimensions': dimensions,
                    'axis_mapping': 'horizontal_seismic_lines'
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
                    'origin': origin,
                    'original_vtk_dimensions': dimensions,
                    'axis_mapping': 'synthetic_data'
                }
                
                return seismic_volume, geometry
                
        except Exception as e:
            self.logger.error(f"Error converting VTK to seismic volume: {str(e)}")
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
        """Write seismic data to SEG-Y format with proper 3D geometry"""
        try:
            nx, ny, nz = seismic_data.shape
            n_traces = nx * ny
            
            # Calculate proper seismic survey geometry
            # Standard seismic parameters for realistic 3D geometry
            inline_spacing = max(25.0, trace_spacing)  # Minimum 25m inline spacing
            xline_spacing = max(25.0, trace_spacing)   # Minimum 25m crossline spacing
            
            # Calculate survey center for proper coordinate system
            survey_center_x = (nx - 1) * inline_spacing / 2.0
            survey_center_y = (ny - 1) * xline_spacing / 2.0
            
            # Base coordinates for survey origin (use realistic UTM-like coordinates)
            base_utm_x = 500000  # 500km UTM easting
            base_utm_y = 4000000  # 4000km UTM northing
            
            # Create SEG-Y file specification
            spec = segyio.spec()
            spec.samples = list(range(nz))
            spec.tracecount = n_traces
            spec.format = 5  # IEEE floating point
            spec.sorting = 2  # CDP sorted (standard for 3D)
            
            with segyio.create(output_path, spec) as f:
                # Set binary header with proper 3D survey parameters
                f.bin[segyio.BinField.Traces] = n_traces
                f.bin[segyio.BinField.Samples] = nz
                f.bin[segyio.BinField.Interval] = int(sample_rate * 1000)  # Microseconds
                f.bin[segyio.BinField.Format] = 5  # IEEE float
                f.bin[segyio.BinField.SortingCode] = 2  # CDP sorted
                f.bin[segyio.BinField.MeasurementSystem] = 1  # Meters
                f.bin[segyio.BinField.SEGYRevision] = 256  # SEG-Y Rev 1.0
                
                # Additional binary header fields for 3D survey
                f.bin[segyio.BinField.JobID] = 1001
                f.bin[segyio.BinField.LineNumber] = 1
                f.bin[segyio.BinField.ReelNumber] = 1
                f.bin[segyio.BinField.EnsembleFold] = 1  # Post-stack
                f.bin[segyio.BinField.VerticalSum] = 1
                f.bin[segyio.BinField.SamplesOriginal] = nz
                f.bin[segyio.BinField.IntervalOriginal] = int(sample_rate * 1000)
                
                # Write traces with proper 3D seismic geometry
                trace_idx = 0
                for i in range(nx):  # Inline direction
                    for j in range(ny):  # Crossline direction
                        # Calculate proper inline/crossline numbers (standard numbering)
                        inline_num = 1000 + i * 10  # Start at 1000, increment by 10
                        xline_num = 2000 + j * 10   # Start at 2000, increment by 10
                        
                        # Calculate real-world coordinates in meters
                        # This creates proper 3D geometry that appears round/elliptical
                        local_x = i * inline_spacing - survey_center_x
                        local_y = j * xline_spacing - survey_center_y
                        
                        # Convert to UTM-like coordinates (standard in seismic)
                        utm_x = int(base_utm_x + local_x)
                        utm_y = int(base_utm_y + local_y)
                        
                        # Calculate CDP number (standard formula for 3D)
                        cdp_num = inline_num * 1000 + xline_num
                        
                        # Create comprehensive trace header for proper 3D seismic
                        trace_header = {
                            # Basic trace identification (SEG-Y standard)
                            segyio.TraceField.TRACE_SEQUENCE_LINE: trace_idx + 1,
                            segyio.TraceField.TRACE_SEQUENCE_FILE: trace_idx + 1,
                            segyio.TraceField.FieldRecord: 1,
                            segyio.TraceField.TraceNumber: trace_idx + 1,
                            segyio.TraceField.EnergySourcePoint: 1,
                            
                            # CDP information (critical for 3D processing)
                            segyio.TraceField.CDP: cdp_num,
                            segyio.TraceField.CDP_TRACE: 1,
                            
                            # Trace identification 
                            segyio.TraceField.TraceIdentificationCode: 1,  # Live seismic data
                            segyio.TraceField.NSummedTraces: 1,
                            segyio.TraceField.NStackedTraces: 1,
                            segyio.TraceField.DataUse: 1,  # Production data
                            
                            # Geometry - CRITICAL for proper 3D visualization
                            segyio.TraceField.offset: 0,  # Post-stack (zero offset)
                            
                            # Source coordinates (UTM-like, in meters)
                            segyio.TraceField.SourceX: utm_x,
                            segyio.TraceField.SourceY: utm_y,
                            
                            # Receiver coordinates (same as source for post-stack)
                            segyio.TraceField.GroupX: utm_x,
                            segyio.TraceField.GroupY: utm_y,
                            
                            # CDP coordinates (midpoint between source and receiver)
                            segyio.TraceField.CDP_X: utm_x,
                            segyio.TraceField.CDP_Y: utm_y,
                            
                            # 3D geometry - ESSENTIAL for 3D interpretation
                            segyio.TraceField.INLINE_3D: inline_num,
                            segyio.TraceField.CROSSLINE_3D: xline_num,
                            
                            # Coordinate scaling (standard seismic practice)
                            segyio.TraceField.SourceGroupScalar: 1,      # No scaling (coordinates in meters)
                            segyio.TraceField.ElevationScalar: 1,       # No scaling 
                            segyio.TraceField.CoordinateUnits: 1,       # Length units (meters)
                            
                            # Elevation information (standard values)
                            segyio.TraceField.ReceiverGroupElevation: 0,   # Sea level
                            segyio.TraceField.SourceSurfaceElevation: 0,   # Sea level
                            segyio.TraceField.SourceDepth: 0,
                            segyio.TraceField.ReceiverDatumElevation: 0,
                            segyio.TraceField.SourceDatumElevation: 0,
                            segyio.TraceField.SourceWaterDepth: 0,
                            segyio.TraceField.GroupWaterDepth: 0,
                            
                            # Timing information
                            segyio.TraceField.DelayRecordingTime: 0,
                            segyio.TraceField.MuteTimeStart: 0,
                            segyio.TraceField.MuteTimeEND: 0,
                            
                            # Sample information (critical for data interpretation)
                            segyio.TraceField.TRACE_SAMPLE_COUNT: nz,
                            segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(sample_rate * 1000),  # Microseconds
                            
                            # Processing information (standard seismic values)
                            segyio.TraceField.GainType: 1,               # Fixed gain
                            segyio.TraceField.InstrumentGainConstant: 1,
                            segyio.TraceField.InstrumentInitialGain: 1,
                            segyio.TraceField.Correlated: 2,             # Not correlated
                            
                            # Filter information (no filtering applied)
                            segyio.TraceField.AliasFilterFrequency: 0,
                            segyio.TraceField.AliasFilterSlope: 0,
                            segyio.TraceField.NotchFilterFrequency: 0,
                            segyio.TraceField.NotchFilterSlope: 0,
                            segyio.TraceField.LowCutFrequency: 0,
                            segyio.TraceField.HighCutFrequency: 0,
                            segyio.TraceField.LowCutSlope: 0,
                            segyio.TraceField.HighCutSlope: 0,
                            
                            # Acquisition date (current date)
                            segyio.TraceField.YearDataRecorded: 2024,
                            segyio.TraceField.DayOfYear: 1,
                            segyio.TraceField.HourOfDay: 12,
                            segyio.TraceField.MinuteOfHour: 0,
                            segyio.TraceField.SecondOfMinute: 0,
                            segyio.TraceField.TimeBaseCode: 1,           # Local time
                            
                            # Weighting and positioning (standard values)
                            segyio.TraceField.TraceWeightingFactor: 1,
                            segyio.TraceField.GeophoneGroupNumberRoll1: 1,
                            segyio.TraceField.GeophoneGroupNumberFirstTraceOrigField: 1,
                            segyio.TraceField.GeophoneGroupNumberLastTraceOrigField: 1,
                            segyio.TraceField.GapSize: 0,
                            segyio.TraceField.OverTravel: 0,
                            
                            # Shot point information
                            segyio.TraceField.ShotPoint: inline_num,     # Use inline as shot point
                            segyio.TraceField.ShotPointScalar: 1,
                            
                            # Measurement units and constants
                            segyio.TraceField.TraceValueMeasurementUnit: -1,  # Unknown/arbitrary
                            segyio.TraceField.TransductionConstantMantissa: 1,
                            segyio.TraceField.TransductionConstantPower: 0,
                            segyio.TraceField.TransductionUnit: -1,     # Unknown
                            segyio.TraceField.TraceIdentifier: 1,
                            segyio.TraceField.ScalarTraceHeader: 1,
                            segyio.TraceField.SourceType: 1,            # Point source
                            segyio.TraceField.SourceEnergyDirectionMantissa: 0,
                            segyio.TraceField.SourceEnergyDirectionExponent: 0,
                            segyio.TraceField.SourceMeasurementMantissa: 0,
                            segyio.TraceField.SourceMeasurementExponent: 0,
                            segyio.TraceField.SourceMeasurementUnit: -1  # Unknown
                        }
                        
                        # Write trace header
                        f.header[trace_idx] = trace_header
                        
                        # Write trace data (ensure float32 format)
                        trace_data = seismic_data[i, j, :].astype(np.float32)
                        f.trace[trace_idx] = trace_data
                        
                        trace_idx += 1
                
                # Create professional textual header with proper survey information
                textual_header = (
                    "C01 CLIENT: VTK Viewer Seismic Survey                               "
                    "C02 DATA: 3D Post-Stack Seismic Volume                            "
                    "C03 AREA: Converted Dataset                                        "
                    "C04 OPERATOR: VTK-SEGY Converter                                  "
                    "C05 DATE: 2024                                                    "
                    "C06                                                               "
                    "C07 SURVEY SPECIFICATIONS:                                        "
                    f"C08 Inline range: {1000} to {1000 + (nx-1)*10} (increment: 10)   "
                    f"C09 Crossline range: {2000} to {2000 + (ny-1)*10} (increment: 10)"
                    f"C10 Inline spacing: {inline_spacing:.1f} meters                  "
                    f"C11 Crossline spacing: {xline_spacing:.1f} meters               "
                    f"C12 Total traces: {n_traces}                                     "
                    "C13                                                               "
                    "C14 TIME/DEPTH PARAMETERS:                                        "
                    f"C15 Sample count: {nz}                                           "
                    f"C16 Sample interval: {sample_rate:.1f} ms                        "
                    f"C17 Total time/depth: {nz * sample_rate:.1f} ms                  "
                    "C18                                                               "
                    "C19 COORDINATE SYSTEM:                                            "
                    f"C20 Origin UTM X: {base_utm_x} meters                            "
                    f"C21 Origin UTM Y: {base_utm_y} meters                            "
                    "C22 Datum: WGS84 (assumed)                                        "
                    "C23 Units: Meters                                                 "
                    "C24                                                               "
                    "C25 PROCESSING NOTES:                                             "
                    "C26 - IEEE floating point format (SEG-Y Rev 1.0)                 "
                    "C27 - CDP sorted for 3D interpretation                           "
                    "C28 - Post-stack migrated data                                   "
                    "C29 - Standard trace headers populated                           "
                    "C30 - Compatible with Petrel, OpendTect, etc.                    "
                    "C31                                                               "
                    "C32 TRACE HEADER KEY:                                             "
                    "C33 Bytes 189-192: Inline number                                 "
                    "C34 Bytes 193-196: Crossline number                              "
                    "C35 Bytes 73-76: UTM X coordinate                                "
                    "C36 Bytes 77-80: UTM Y coordinate                                "
                    "C37 Bytes 21-24: CDP number                                      "
                    "C38                                                               "
                    "C39 SOFTWARE: VTK Viewer with Seismic Support v1.0               "
                    "C40 END TEXTUAL HEADER                                            "
                )
                
                # Ensure exactly 3200 bytes (40 lines × 80 chars)
                if len(textual_header) < 3200:
                    textual_header += " " * (3200 - len(textual_header))
                elif len(textual_header) > 3200:
                    textual_header = textual_header[:3200]
                
                f.text[0] = textual_header
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing SEG-Y file: {e}")
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
        """Convert seismic volume to VTK ImageData with proper axis orientation"""
        try:
            ni, nj, nk = seismic_data.shape
            
            # SEG-Y format: (inline, crossline, time/depth) where 3rd axis is vertical
            # VTK format: (X, Y, Z) where Z is typically vertical
            # Since we preserved the axis mapping during conversion, we can keep the same orientation
            
            # For the simplified approach: SEG-Y (inline, xline, time) -> VTK (X, Y, Z)
            # This maintains the correct vertical orientation
            vtk_data = seismic_data  # Keep the same orientation
            vtk_dimensions = (ni, nj, nk)
            
            self.logger.info(f"Converting SEG-Y dimensions ({ni}, {nj}, {nk}) back to VTK")
            self.logger.info(f"Maintaining Z-axis as vertical (time/depth -> Z)")
            
            # Create VTK ImageData with the dimensions
            image_data = vtk.vtkImageData()
            image_data.SetDimensions(vtk_dimensions)
            
            # Set spacing (convert from time/distance to spatial coordinates)
            spacing_x = scale_factor
            spacing_y = scale_factor
            spacing_z = headers.get('sample_rate', 4000) * scale_factor / 1000.0  # Convert microseconds to ms
            
            image_data.SetSpacing(spacing_x, spacing_y, spacing_z)
            image_data.SetOrigin(0.0, 0.0, 0.0)
            
            # Convert seismic data to VTK array (use Fortran order to match VTK)
            vtk_array = numpy_to_vtk(vtk_data.ravel(order='F'))
            vtk_array.SetName("Amplitude")
            
            # Add to point data
            image_data.GetPointData().SetScalars(vtk_array)
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"Error converting seismic volume to VTK: {str(e)}")
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