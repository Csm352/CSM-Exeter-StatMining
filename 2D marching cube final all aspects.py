# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:40:22 2025

@author: km923
"""

import numpy as np
from scipy.spatial import Delaunay
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
import mcubes
import ezdxf

def export_to_dxf(vertices, triangles, filename, scale=(2, 2, 1)):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for triangle in triangles:
        p1 = tuple(vertices[triangle[0]] * scale)
        p2 = tuple(vertices[triangle[1]] * scale)
        p3 = tuple(vertices[triangle[2]] * scale)
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

# Load the CSV data
df = pd.read_csv("GridWithRockTypes.csv")

# Filter points where Z = 0
filtered_df = df[df["Z"] == 0]

# Extract X, Y, and Rock_Type for Z = 0
original_points = filtered_df[["X", "Y"]].values
rock_types = filtered_df["Rock_Type"].values
rock_type_names = {1: "Rock Type 1", 2: "Rock Type 2", 3: "Rock Type 3"}  # Example rock types

# User-defined grid spacing
spacing = 5  # Adjust this value based on your preference

# Define grid bounds
# Define grid bounds
x_min, x_max = df["X"].min() - 2*spacing, df["X"].max() + 2*spacing
y_min, y_max = df["Y"].min() - 2*spacing, df["Y"].max() + 2*spacing


# Generate regular 2D grid
x_grid = np.arange(x_min, x_max + spacing, spacing)
y_grid = np.arange(y_min, y_max + spacing, spacing)
z_grid = np.array([0, 5])  # Add two layers (0 and 5)

X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

# Interpolate Rock_Type data onto the grid
grid_rock_types = griddata(
    points=original_points,
    values=rock_types,
    xi=(X, Y),
    method="nearest"
)

# Extract unique rock types
unique_rock_types = np.unique(rock_types)

# Initialize the figure for the current rock type
fig = go.Figure()
# Add original points for all the rock type
print(original_points.shape, grid_rock_types.shape)

# Add original points for all the rock types with hover text for the rock type
fig.add_trace(
    go.Scatter3d(
        x=original_points[:, 0],
        y=original_points[:, 1],
        z=np.zeros_like(original_points[:, 0]),
        mode="markers",
        marker=dict(
            size=6,
            color=rock_types,  # Use rock_types to color points
            colorscale="Viridis",  # Apply a color scale for different rock types
            showscale=True  # Show the color scale
        ),
        text=[rock_type_names.get(rock, "Unknown Rock Type") for rock in rock_types],  # Hover text for each point
        hoverinfo="text",  # Display the hover text (rock type names)
        name="Rock Type Points"
    )
)

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

# Show the figure in the browser
fig.show(renderer="browser")

for rock_type in unique_rock_types:
    if rock_type == 0:
        continue
    
    rock_volume = (grid_rock_types == rock_type).astype(float)
    rock_volume = np.expand_dims(rock_volume, axis=2)
    rock_volume_3d = np.repeat(rock_volume, len(z_grid), axis=2)
    
    rock_type_boundary = detect_rock_type_boundaries(grid_rock_types, rock_type)
    rock_volume_3d_boundary = extract_surface_from_boundary(rock_type_boundary, rock_volume_3d)
    
    vertices, triangles = mcubes.marching_cubes(rock_volume_3d_boundary, 0.5)
    
    open_edges = detect_open_boundaries(vertices, triangles)
    
    updated_vertices, updated_triangles = close_boundaries(vertices, triangles, open_edges, rock_type_boundary, rock_volume_3d)
    
    dxf_filename = f"rock_type_{rock_type}.dxf"
    export_to_dxf(updated_vertices, updated_triangles, filename=dxf_filename)
    
    updated_vertices[:, 0] = updated_vertices[:, 0] * spacing + x_min
    updated_vertices[:, 1] = updated_vertices[:, 1] * spacing + y_min
    updated_vertices[:, 2] = updated_vertices[:, 2] + z_grid.min()
    
    # The updated_vertices contain the 3D vertices, and updated_triangles contain the triangle indices
    x = updated_vertices[:, 0]
    y = updated_vertices[:, 1]
    z = updated_vertices[:, 2]
    i, j, k = updated_triangles[:, 0], updated_triangles[:, 1], updated_triangles[:, 2]
    
    # Create a 3D mesh plot with mesh3d
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
            color='lightblue',  # You can change the color
            showscale=True,
        )
    )
    # Add original points for the current rock type
    mask = rock_types == rock_type
    fig.add_trace(
        go.Scatter3d(
            x=original_points[mask, 0],
            y=original_points[mask, 1],
            z=np.zeros(mask.sum()),
            mode="markers",
            marker=dict(size=2, color=rock_type, colorscale="Viridis", showscale=False),
            name=f"Rock Type {rock_type} Points"
        )
    )
    
    fig.update_layout(
        title=f"3D Mesh for Rock Type {rock_type}",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data"
        ),
    )
    
    fig.show(renderer="browser")

   