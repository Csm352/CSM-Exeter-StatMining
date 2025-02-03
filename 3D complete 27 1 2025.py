# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:35:02 2025

@author: km923
"""

import numpy as np
from scipy.spatial import Delaunay
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
import mcubes
import ezdxf

def export_to_dxf(vertices, triangles, filename):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for triangle in triangles:
        p1 = tuple(vertices[triangle[0]])
        p2 = tuple(vertices[triangle[1]])
        p3 = tuple(vertices[triangle[2]])
        msp.add_3dface([p1, p2, p3])
    doc.saveas(filename)
    print(f"DXF file saved as: {filename}")

def detect_open_boundaries(vertices, triangles):
    """
    Detect the open boundaries in the mesh by checking for edges that appear only once in the mesh.
    An edge that appears only once is considered an open boundary.
    """
    edges = []
    edge_count = {}
    
    # Iterate over all triangles to find edges
    for triangle in triangles:
        # Define the three edges of the triangle
        edges_in_triangle = [
            tuple(sorted([triangle[0], triangle[1]])),
            tuple(sorted([triangle[1], triangle[2]])),
            tuple(sorted([triangle[2], triangle[0]]))
        ]
        
        # Count each edge's occurrences
        for edge in edges_in_triangle:
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1
    
    # Detect edges that appear only once (open boundaries)
    for edge, count in edge_count.items():
        if count == 1:
            edges.append(edge)
    
    return edges

def detect_rock_type_boundaries(grid_rock_types, rock_type):
    """
    Detect the boundaries for the specific rock type by finding the
    transitions between the rock type and neighboring regions.
    """
    # Initialize an empty array for the boundary
    boundary = np.zeros_like(grid_rock_types, dtype=bool)
    
    # Iterate over the grid to find boundaries (adjacent to other rock types or empty space)
    for i in range(1, grid_rock_types.shape[0] - 1):
        for j in range(1, grid_rock_types.shape[1] - 1):
            if grid_rock_types[i, j] == rock_type:
                # Check adjacent cells to find boundaries
                neighbors = [
                    grid_rock_types[i - 1, j], grid_rock_types[i + 1, j], 
                    grid_rock_types[i, j - 1], grid_rock_types[i, j + 1]
                ]
                if any(neighbor != rock_type for neighbor in neighbors):
                    boundary[i, j] = True
    return boundary

def extract_surface_from_boundary(rock_type_boundary, rock_volume_3d):
    """
    Extract the surface from the 3D grid by using the rock type boundary to guide the surface extraction.
    """
    # We only extract the surface within the boundary defined for the specific rock type
    rock_volume_3d_boundary = np.zeros_like(rock_volume_3d)
    rock_volume_3d_boundary[:, :, :] = rock_volume_3d[:, :, :]
    
    # Mask the surface where rock type boundary is True
    for i in range(rock_volume_3d.shape[0]):
        for j in range(rock_volume_3d.shape[1]):
            for k in range(rock_volume_3d.shape[2]):
                if not rock_type_boundary[i, j]:  # Only keep regions inside the boundary
                    rock_volume_3d_boundary[i, j, k] = 0

    return rock_volume_3d_boundary

def close_boundaries(vertices, triangles, open_edges, rock_type_boundary, rock_volume_3d):
    """
    Close the boundaries by adding new triangles and ensuring they respect the rock type's boundary.
    """
    for edge in open_edges:
        # For simplicity, we connect the boundary edges respecting the rock type's boundary
        p1, p2 = vertices[edge[0]], vertices[edge[1]]
        midpoint = (p1 + p2) / 2
        
        # Only add new triangles if the boundary point lies within the specified rock type boundary
        if rock_type_boundary[int(midpoint[0]), int(midpoint[1])]:  # Check if inside boundary
            new_triangle = [edge[0], edge[1], len(vertices)]
            triangles = np.vstack([triangles, new_triangle])
            vertices = np.vstack([vertices, midpoint])    
    return vertices, triangles

def extract_from_dxf(filename):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    vertices = []
    triangles = []

    for entity in msp:
        if entity.dxftype() == '3DFACE':
            # Extract the 4 vertices of the 3DFACE (triangle)
            points = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2, entity.dxf.vtx3]
            # Remove duplicate vertices (since 3DFACE can have 4 vertices, but triangles need 3)
            unique_points = list(set(points))
            if len(unique_points) >= 3:
                # Map vertices to indices
                for point in unique_points:
                    if point not in vertices:
                        vertices.append(point)
                # Create a triangle using the indices of the vertices
                triangle = [vertices.index(p) for p in unique_points[:3]]
                triangles.append(triangle)

    return np.array(vertices), np.array(triangles)


# Load the CSV data
df = pd.read_csv("GridWithRockTypes.csv")
# Extract X, Y, and Rock_Type for Z = 0
original_points = df[["X", "Y", "Z"]].values

rock_types = df["Rock_Type"].values
rock_type_names = {1: "Rock Type 1", 2: "Rock Type 2", 3: "Rock Type 3"}  # Example rock types
# Extract unique rock types
unique_rock_types = np.unique(rock_types)
z_levels_origin = np.unique(df["Z"].values)
rock_type_colors = {rock_type: f"hsl({(i/len(unique_rock_types))*360},100%,50%)" for i, rock_type in enumerate(unique_rock_types)}
# Initialize the figure for the current rock type
fig = go.Figure()
# Add original points for all the rock types with hover text for the rock type
fig.add_trace(
    go.Scatter3d(
        x=original_points[:, 0],
        y=original_points[:, 1],
        z=original_points[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=[rock_type_colors[rock] for rock in rock_types],  # Use rock_types to color points
            colorscale="Viridis",  # Apply a color scale for different rock types
            showscale=True  # Show the color scale
        ),
        text=[rock_type_names.get(rock, "Unknown Rock Type") for rock in rock_types],  # Hover text for each point
        hoverinfo="text",  # Display the hover text (rock type names)
        name="Rock Type Points"))

# Define layout for the current figure
fig.update_layout(
    title="Points Colored by Rock Type",
    scene=dict(
        xaxis=dict(title="X"),  # Set title for the x-axis
        yaxis=dict(title="Y"),  # Set title for the y-axis
        zaxis=dict(title="Z"),  # Set title for the z-axis
        aspectmode="data"  # Ensures the aspect ratio of the plot is preserved
    )
)
fig.show(renderer="browser")

z_min, z_max = df["Z"].min(), df["Z"].max()
spacing_origin = int((df["X"].max() - df["X"].min())/(len(df["X"].unique()) - 1))
spacing_origin_z = int((df["Z"].max() - df["Z"].min())/(len(df["Z"].unique()) - 1)) 
x_min, x_max = df["X"].min() - 2*spacing_origin, df["X"].max() + 2*spacing_origin
y_min, y_max = df["Y"].min() - 2*spacing_origin, df["Y"].max() + 2*spacing_origin
# spacing_xy = (spacing_origin - x_min) / (x_max - x_min)
# spacing_z = (spacing_origin_z - z_min) / (z_max - z_min)
# Define grid bounds
grid_min, grid_max = x_min - 2*spacing_origin, x_min + 2*spacing_origin

z_levels = np.sort(df["Z"].unique())
# Generate regular 2D grid
x_grid = np.arange(x_min, x_max + spacing_origin, spacing_origin)
y_grid = np.arange(y_min, y_max + spacing_origin, spacing_origin)

combined_vertices = []
combined_triangles = []
rock_vertices = {}
rock_triangles = {}
rock_vertices_dxf_extract = {}
rock_triangles_dxf_extract = {}
for rock_type in unique_rock_types:
    if rock_type == 0:
        continue
    all_vertices = np.empty((0, 3))
    all_triangles = np.array([]).reshape(0, 3)  # Initialize as an empty array with shape (0, 3) for triangles
    ii = 0
    for z_l in z_levels:
        # Generate regular 2D grid
        
        z_l = int(z_l)
        z_grid = np.array([int(z_l), int(z_l+spacing_origin_z)])  # Add two layers (0 and 5)
        filtered_df = df[df["Z"] == z_l]
        original_points_z = filtered_df[["X", "Y"]].values
        if len(original_points_z) == 0:
            continue
        rock_types_z = filtered_df["Rock_Type"].values.astype(int)
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    
        # Interpolate Rock_Type data onto the grid
        grid_rock_types = griddata(
            points=original_points_z,
            values=rock_types_z,
            xi=(X, Y),
            method="nearest")
    
        rock_volume = (grid_rock_types == rock_type).astype(float)
        rock_volume = np.expand_dims(rock_volume, axis=2)
        rock_volume_3d = np.zeros((rock_volume.shape[0], rock_volume.shape[1], len(z_grid)))
        for i, z_val in enumerate(z_grid):
            rock_volume_3d[:, :, i] = rock_volume[:, :, 0]  # Assuming rock_volume is a 2D slice for each Z level
        #rock_volume_3d = np.repeat(rock_volume, len(z_grid), axis=2)
        
        rock_type_boundary = detect_rock_type_boundaries(grid_rock_types, rock_type)
        rock_volume_3d_boundary = extract_surface_from_boundary(rock_type_boundary, rock_volume_3d)
        
        vertices, triangles = mcubes.marching_cubes(rock_volume_3d_boundary, 0.5)        
        
        open_edges = detect_open_boundaries(vertices, triangles)
        
        updated_vertices, updated_triangles = close_boundaries(vertices, triangles, open_edges, rock_type_boundary, rock_volume_3d)
        
        
        
        
        # Track the global index offset
        # Convert Z elevations correctly
        # Convert to original coordinates by scaling the x and y coordinates by spacing_origin
        updated_vertices[:, 0] = updated_vertices[:, 0] * spacing_origin + x_min
        updated_vertices[:, 1] = updated_vertices[:, 1] * spacing_origin + y_min
        # updated_vertices[:, 2] = updated_vertices[:, 2] + z_grid.min()
        # updated_vertices[:, 0] *= spacing_origin  # Scale the x-coordinates
        # updated_vertices[:, 1] *= spacing_origin  # Scale the y-coordinates
        updated_vertices[:, 2] = np.where(
            updated_vertices[:, 2] == 1, 
            spacing_origin_z + spacing_origin_z*ii, 
            spacing_origin_z*ii)
        
        if ii == 0 or all_vertices.size == 0:
            vertex_offset = 0
        else:
            vertex_offset = len(all_vertices)        
        if all_vertices.size == 0:
            all_vertices = updated_vertices  # First assignment
        else:
            all_vertices = np.vstack([all_vertices, updated_vertices])
        
        # Adjust triangle indices
        adjusted_triangles = updated_triangles + vertex_offset
    
        # Use np.vstack() to append the new triangles
        all_triangles = np.vstack([all_triangles, adjusted_triangles])  # Append using vstack
        
        #all_triangles.append(updated_triangles + sum(len(v) for v in all_vertices[:-1]))
        rock_vertices_f = all_vertices
        rock_triangles_f = all_triangles
        rock_triangles_f = np.round(rock_triangles_f).astype(int)
        # Combine vertices and triangles for the current rock type
        rock_vertices[rock_type] = rock_vertices_f
        rock_triangles[rock_type] = rock_triangles_f
        # print("Z levels in data:", np.unique(rock_vertices_f[:, 2]))
        # print(rock_vertices_f.shape, rock_triangles_f.shape)
        ii += 1
    dxf_filename = f"rock_type_{rock_type}.dxf"
    export_to_dxf(rock_vertices_f, rock_triangles_f, filename=dxf_filename)
    # Extract vertices and triangles from the DXF file
    rock_vertices_f, rock_triangles_f = extract_from_dxf(dxf_filename)
    rock_vertices_dxf_extract[rock_type] = rock_vertices_f
    rock_triangles_dxf_extract[rock_type] = rock_triangles_f
    filtered_df_rocktype = df[df["Rock_Type"] == rock_type]
    original_points_rocktype = filtered_df_rocktype[["X", "Y", "Z"]].values
    # Create a 3D mesh plot with mesh3d
    # Plot the mesh for the current rock type
    x, y, z = rock_vertices_f[:, 0], rock_vertices_f[:, 1], rock_vertices_f[:, 2]
    i, j, k = rock_triangles_f[:, 0], rock_triangles_f[:, 1], rock_triangles_f[:, 2]
    i = i.astype(int)
    j = j.astype(int)
    k = k.astype(int)
    fig = go.Figure()    
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=0.50,  # Adjust transparency of the mesh if needed
            color=rock_type_colors[rock_type],  # You can change the color
            showscale=True,))
    # Add original points for the current rock type
    #mask = rock_types == rock_type
    mask = (rock_types == rock_type)
    
    print("Shape of mask:", mask.shape)
    filtered_points = original_points[mask]  # Apply mask
    print("Shape of filtered_points:", filtered_points.shape)
    #filtered_points = original_points_z[mask]
    fig.add_trace(
        go.Scatter3d(
            x=filtered_points[:, 0],
            y=filtered_points[:, 1],
            z=filtered_points[:, 2],
            mode="markers",
            marker=dict(size=3, color=rock_type_colors[rock_type], colorscale="Viridis", showscale=False),
            name=f"Rock Type {rock_type} Points"))
    
    fig.update_layout(
        title=f"3D Mesh for Rock Type {rock_type}",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data"),)
    
    fig.show(renderer="browser")
       
fig = go.Figure()
for rock_type in unique_rock_types:
    dxf_filename = f"rock_type_{rock_type}.dxf"
    
    rock_vertices_f = rock_vertices_dxf_extract[rock_type]
    rock_triangles_f = rock_triangles_dxf_extract[rock_type] 
    # Create a 3D mesh plot with mesh3d
    # Plot the mesh for the current rock type
    x, y, z = rock_vertices_f[:, 0], rock_vertices_f[:, 1], rock_vertices_f[:, 2]
    i, j, k = rock_triangles_f[:, 0], rock_triangles_f[:, 1], rock_triangles_f[:, 2]
    i = i.astype(int)
    j = j.astype(int)
    k = k.astype(int)
   
    
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=0.50,  # Adjust transparency of the mesh if needed
            color=rock_type_colors[rock_type],  # You can change the color
            showscale=True,))
    # Add original points for the current rock type
    #mask = rock_types == rock_type
    mask = (rock_types == rock_type)
    
    print("Shape of mask:", mask.shape)
    filtered_points = original_points[mask]  # Apply mask
    print("Shape of filtered_points:", filtered_points.shape)
    #filtered_points = original_points_z[mask]
    fig.add_trace(
        go.Scatter3d(
            x=filtered_points[:, 0],
            y=filtered_points[:, 1],
            z=filtered_points[:, 2],
            mode="markers",
            marker=dict(size=3, color=rock_type_colors[rock_type], colorscale="Viridis", showscale=False),
            name=f"Rock Type {rock_type} Points"))
    
    fig.update_layout(
        title=f"3D Mesh for Rock Type {rock_type}",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data"),)
    
fig.show(renderer="browser")