#!/usr/bin/env python3
"""
VTK File Viewer Application
A simple GUI application for viewing VTK files using PySide6 and VTK
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main_window import MainWindow


def main():
    """Main application entry point"""
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("VTK File Viewer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("VTK Viewer")
    
    # Enable high DPI scaling (Qt 6 handles this automatically, but keeping for compatibility)
    try:
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # These attributes were deprecated in Qt 6, no action needed
        pass
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # If a file is provided as command line argument, load it
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            window.load_file(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 