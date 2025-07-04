#!/usr/bin/env python3
"""
Script to create sample VTK files for testing the VTK viewer application
"""

import vtk
import numpy as np
import os


def create_sphere_vtk():
    """Create a simple sphere VTK file"""
    # Create sphere source
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(30)
    sphere.SetPhiResolution(30)
    sphere.Update()
    
    # Write to VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("examples/sphere.vtk")
    writer.SetInputConnection(sphere.GetOutputPort())
    writer.Write()
    print("Created: examples/sphere.vtk")


def create_cube_vtk():
    """Create a simple cube VTK file"""
    # Create cube source
    cube = vtk.vtkCubeSource()
    cube.SetXLength(2.0)
    cube.SetYLength(2.0)
    cube.SetZLength(2.0)
    cube.Update()
    
    # Write to VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("examples/cube.vtk")
    writer.SetInputConnection(cube.GetOutputPort())
    writer.Write()
    print("Created: examples/cube.vtk")


def create_cylinder_vtp():
    """Create a cylinder in VTP format"""
    # Create cylinder source
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(0.8)
    cylinder.SetHeight(3.0)
    cylinder.SetResolution(20)
    cylinder.Update()
    
    # Write to VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("examples/cylinder.vtp")
    writer.SetInputConnection(cylinder.GetOutputPort())
    writer.Write()
    print("Created: examples/cylinder.vtp")


def create_torus_ply():
    """Create a torus and save as PLY"""
    # Create torus
    torus = vtk.vtkParametricTorus()
    torus.SetRingRadius(2.0)
    torus.SetCrossSectionRadius(0.5)
    
    # Generate torus surface
    torus_source = vtk.vtkParametricFunctionSource()
    torus_source.SetParametricFunction(torus)
    torus_source.SetUResolution(30)
    torus_source.SetVResolution(30)
    torus_source.Update()
    
    # Write to PLY file
    writer = vtk.vtkPLYWriter()
    writer.SetFileName("examples/torus.ply")
    writer.SetInputConnection(torus_source.GetOutputPort())
    writer.Write()
    print("Created: examples/torus.ply")


def create_cone_stl():
    """Create a cone and save as STL"""
    # Create cone source
    cone = vtk.vtkConeSource()
    cone.SetRadius(1.5)
    cone.SetHeight(3.0)
    cone.SetResolution(25)
    cone.Update()
    
    # Write to STL file
    writer = vtk.vtkSTLWriter()
    writer.SetFileName("examples/cone.stl")
    writer.SetInputConnection(cone.GetOutputPort())
    writer.Write()
    print("Created: examples/cone.stl")


def main():
    """Create all sample files"""
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    print("Creating sample VTK files...")
    
    try:
        create_sphere_vtk()
        create_cube_vtk()
        create_cylinder_vtp()
        create_torus_ply()
        create_cone_stl()
        
        print("\nSample files created successfully!")
        print("You can now test the VTK viewer with these files:")
        print("- examples/sphere.vtk")
        print("- examples/cube.vtk")
        print("- examples/cylinder.vtp")
        print("- examples/torus.ply")
        print("- examples/cone.stl")
        
    except Exception as e:
        print(f"Error creating sample files: {e}")


if __name__ == "__main__":
    main() 