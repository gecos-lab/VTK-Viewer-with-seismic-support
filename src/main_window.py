"""
Main Window for VTK Viewer Application
"""
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMenuBar, QMenu, QStatusBar, QFileDialog, 
                               QMessageBox, QPushButton, QLabel, QGroupBox,
                               QSpinBox, QDoubleSpinBox, QFormLayout, QDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence

from .vtk_viewer_widget import VTKViewerWidget


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
    def setup_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("VTK File Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create VTK viewer
        self.vtk_viewer = VTKViewerWidget()
        
        # Create control panel
        control_panel = self.create_control_panel()
        
        # Add widgets to layout
        main_layout.addWidget(control_panel, 0)  # Fixed size
        main_layout.addWidget(self.vtk_viewer, 1)  # Expandable
        
    def create_control_panel(self):
        """Create the control panel with buttons and options"""
        panel = QGroupBox("Controls")
        panel.setFixedWidth(200)
        layout = QVBoxLayout()
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.open_button = QPushButton("Open VTK File")
        self.open_button.clicked.connect(self.open_file)
        file_layout.addWidget(self.open_button)
        
        self.clear_button = QPushButton("Clear Scene")
        self.clear_button.clicked.connect(self.clear_scene)
        file_layout.addWidget(self.clear_button)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # View controls
        view_group = QGroupBox("View Controls")
        view_layout = QVBoxLayout()
        
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(self.reset_camera)
        view_layout.addWidget(self.reset_camera_button)
        
        # Background color buttons
        bg_label = QLabel("Background:")
        view_layout.addWidget(bg_label)
        
        self.bg_dark_button = QPushButton("Dark")
        self.bg_dark_button.clicked.connect(lambda: self.set_background(0.1, 0.1, 0.1))
        view_layout.addWidget(self.bg_dark_button)
        
        self.bg_light_button = QPushButton("Light")
        self.bg_light_button.clicked.connect(lambda: self.set_background(0.9, 0.9, 0.9))
        view_layout.addWidget(self.bg_light_button)
        
        self.bg_blue_button = QPushButton("Blue")
        self.bg_blue_button.clicked.connect(lambda: self.set_background(0.2, 0.3, 0.6))
        view_layout.addWidget(self.bg_blue_button)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # Current file info
        info_group = QGroupBox("File Info")
        info_layout = QVBoxLayout()
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        info_layout.addWidget(self.file_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Seismic conversion controls
        seismic_group = QGroupBox("Seismic Conversion")
        seismic_layout = QVBoxLayout()
        
        self.convert_to_segy_button = QPushButton("Convert VTK → SEG-Y")
        self.convert_to_segy_button.clicked.connect(self.convert_to_segy)
        seismic_layout.addWidget(self.convert_to_segy_button)
        
        self.convert_to_vtk_button = QPushButton("Convert SEG-Y → VTK")
        self.convert_to_vtk_button.clicked.connect(self.convert_to_vtk)
        seismic_layout.addWidget(self.convert_to_vtk_button)
        
        seismic_group.setLayout(seismic_layout)
        layout.addWidget(seismic_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def setup_menu(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip("Open a VTK file")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        reset_camera_action = QAction("Reset Camera", self)
        reset_camera_action.setShortcut("R")
        reset_camera_action.setStatusTip("Reset camera to fit all objects")
        reset_camera_action.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_camera_action)
        
        clear_action = QAction("Clear Scene", self)
        clear_action.setShortcut("C")
        clear_action.setStatusTip("Clear all objects from scene")
        clear_action.triggered.connect(self.clear_scene)
        view_menu.addAction(clear_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        convert_to_segy_action = QAction("Convert VTK to SEG-Y...", self)
        convert_to_segy_action.setStatusTip("Convert current VTK file to SEG-Y format")
        convert_to_segy_action.triggered.connect(self.convert_to_segy)
        tools_menu.addAction(convert_to_segy_action)
        
        convert_to_vtk_action = QAction("Convert SEG-Y to VTK...", self)
        convert_to_vtk_action.setStatusTip("Convert SEG-Y file to VTK format")
        convert_to_vtk_action.triggered.connect(self.convert_to_vtk)
        tools_menu.addAction(convert_to_vtk_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def open_file(self):
        """Open a VTK file dialog and load the selected file"""
        file_types = (
            "VTK Files (*.vtk *.vtp *.vtu *.vti *.vtr *.vts);;"
            "Seismic Files (*.segy *.sgy);;"
            "3D Model Files (*.obj *.ply *.stl);;"
            "All Files (*.*)"
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open VTK File",
            "",
            file_types
        )
        
        if file_path:
            self.load_file(file_path)
            
    def load_file(self, file_path):
        """Load a VTK file"""
        try:
            self.status_bar.showMessage(f"Loading {os.path.basename(file_path)}...")
            
            # Use QTimer to update UI before loading
            QTimer.singleShot(100, lambda: self._load_file_delayed(file_path))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self.status_bar.showMessage("Error loading file")
            
    def _load_file_delayed(self, file_path):
        """Delayed file loading to allow UI update"""
        try:
            success = self.vtk_viewer.load_vtk_file(file_path)
            
            if success:
                self.current_file = file_path
                filename = os.path.basename(file_path)
                self.file_label.setText(f"File: {filename}")
                self.status_bar.showMessage(f"Loaded: {filename}")
                self.setWindowTitle(f"VTK File Viewer - {filename}")
            else:
                QMessageBox.warning(self, "Warning", "Failed to load the VTK file.")
                self.status_bar.showMessage("Failed to load file")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self.status_bar.showMessage("Error loading file")
            
    def clear_scene(self):
        """Clear the VTK scene"""
        self.vtk_viewer.clear_scene()
        self.current_file = None
        self.file_label.setText("No file loaded")
        self.setWindowTitle("VTK File Viewer")
        self.status_bar.showMessage("Scene cleared")
        
    def reset_camera(self):
        """Reset the camera view"""
        self.vtk_viewer.reset_camera()
        self.status_bar.showMessage("Camera reset")
        
    def set_background(self, r, g, b):
        """Set the background color"""
        self.vtk_viewer.set_background_color(r, g, b)
        self.status_bar.showMessage("Background color changed")
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About VTK File Viewer",
            "VTK File Viewer with Seismic Support v1.1\n\n"
            "A comprehensive application for viewing VTK files and working with seismic data.\n\n"
            "Supported formats:\n"
            "• VTK legacy format (.vtk)\n"
            "• VTK XML formats (.vtp, .vtu, .vti, .vtr, .vts)\n"
            "• Seismic formats (.segy, .sgy)\n"
            "• 3D model formats (.obj, .ply, .stl)\n\n"
            "Seismic features:\n"
            "• VTK ↔ SEG-Y conversion\n"
            "• Seismic data visualization with slice planes\n"
            "• Volume rendering for smaller datasets\n"
            "• Specialized seismic colormaps\n\n"
            "Mouse controls:\n"
            "• Left click + drag: Rotate\n"
            "• Right click + drag: Zoom\n"
            "• Middle click + drag: Pan"
        )
        
    def convert_to_segy(self):
        """Convert current VTK file to SEG-Y format"""
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "Please load a VTK file first.")
            return
        
        # Check if current file is a VTK file
        if not self.current_file.lower().endswith(('.vtk', '.vtp', '.vtu', '.vti', '.vtr', '.vts')):
            QMessageBox.warning(self, "Warning", "Current file is not a VTK file.")
            return
        
        # Show conversion dialog
        dialog = SeismicConversionDialog(conversion_type="vtk_to_segy", parent=self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_parameters()
            
            # Get output file path
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save SEG-Y File",
                "",
                "SEG-Y Files (*.segy *.sgy);;All Files (*.*)"
            )
            
            if output_path:
                self.status_bar.showMessage("Converting VTK to SEG-Y...")
                
                # Perform conversion
                success = self.vtk_viewer.convert_vtk_to_segy(
                    self.current_file,
                    output_path,
                    params['sample_rate'],
                    params['trace_spacing']
                )
                
                if success:
                    QMessageBox.information(
                        self, "Success", 
                        f"Successfully converted VTK to SEG-Y:\n{output_path}"
                    )
                    self.status_bar.showMessage("Conversion completed")
                else:
                    QMessageBox.critical(
                        self, "Error", 
                        "Failed to convert VTK to SEG-Y. Check console for details."
                    )
                    self.status_bar.showMessage("Conversion failed")
    
    def convert_to_vtk(self):
        """Convert SEG-Y file to VTK format"""
        # Get input SEG-Y file
        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SEG-Y File",
            "",
            "SEG-Y Files (*.segy *.sgy);;All Files (*.*)"
        )
        
        if not input_path:
            return
        
        # Show conversion dialog
        dialog = SeismicConversionDialog(conversion_type="segy_to_vtk", parent=self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_parameters()
            
            # Get output file path
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save VTK File",
                "",
                "VTK Files (*.vti *.vtk);;All Files (*.*)"
            )
            
            if output_path:
                self.status_bar.showMessage("Converting SEG-Y to VTK...")
                
                # Perform conversion
                success = self.vtk_viewer.convert_segy_to_vtk(
                    input_path,
                    output_path,
                    params['scale_factor']
                )
                
                if success:
                    QMessageBox.information(
                        self, "Success", 
                        f"Successfully converted SEG-Y to VTK:\n{output_path}"
                    )
                    self.status_bar.showMessage("Conversion completed")
                    
                    # Ask if user wants to load the converted file
                    reply = QMessageBox.question(
                        self, "Load Converted File",
                        "Do you want to load the converted VTK file?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        self.load_file(output_path)
                else:
                    QMessageBox.critical(
                        self, "Error", 
                        "Failed to convert SEG-Y to VTK. Check console for details."
                    )
                    self.status_bar.showMessage("Conversion failed")

    def closeEvent(self, event):
        """Handle application close event"""
        event.accept()


class SeismicConversionDialog(QDialog):
    """Dialog for configuring seismic conversion parameters"""
    
    def __init__(self, conversion_type="vtk_to_segy", parent=None):
        super().__init__(parent)
        self.conversion_type = conversion_type
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        if self.conversion_type == "vtk_to_segy":
            self.setWindowTitle("VTK to SEG-Y Conversion Parameters")
        else:
            self.setWindowTitle("SEG-Y to VTK Conversion Parameters")
        
        self.setModal(True)
        self.resize(300, 200)
        
        layout = QVBoxLayout()
        
        # Parameters form
        form_layout = QFormLayout()
        
        if self.conversion_type == "vtk_to_segy":
            # Sample rate in microseconds
            self.sample_rate_spin = QDoubleSpinBox()
            self.sample_rate_spin.setRange(1000, 10000)
            self.sample_rate_spin.setValue(4000)
            self.sample_rate_spin.setSuffix(" μs")
            form_layout.addRow("Sample Rate:", self.sample_rate_spin)
            
            # Trace spacing in meters
            self.trace_spacing_spin = QDoubleSpinBox()
            self.trace_spacing_spin.setRange(1.0, 100.0)
            self.trace_spacing_spin.setValue(25.0)
            self.trace_spacing_spin.setSuffix(" m")
            form_layout.addRow("Trace Spacing:", self.trace_spacing_spin)
        
        else:  # segy_to_vtk
            # Scale factor
            self.scale_factor_spin = QDoubleSpinBox()
            self.scale_factor_spin.setRange(0.1, 10.0)
            self.scale_factor_spin.setValue(1.0)
            self.scale_factor_spin.setDecimals(2)
            form_layout.addRow("Scale Factor:", self.scale_factor_spin)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        """Get the conversion parameters"""
        if self.conversion_type == "vtk_to_segy":
            return {
                'sample_rate': self.sample_rate_spin.value(),
                'trace_spacing': self.trace_spacing_spin.value()
            }
        else:
            return {
                'scale_factor': self.scale_factor_spin.value()
            } 