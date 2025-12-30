"""
Geometry generation for Blender visualization.

Creates meshes, point clouds, and instances from parsed lattice data.
"""

import bpy
import bmesh
import numpy as np
from dataclasses import dataclass
from typing import Optional

from . import importers


@dataclass
class GeometryOptions:
    """Options controlling how geometry is created."""
    import_density: bool = True
    import_links: bool = False
    import_plaquettes: bool = False
    density_threshold: float = 0.001
    visualization_mode: str = 'POINTS'  # 'POINTS', 'SPHERES', 'VOLUME'


def create_geometry(context, data: importers.LatticeData, options: GeometryOptions):
    """Create Blender geometry from parsed lattice data.

    Args:
        context: Blender context
        data: Parsed LatticeData from .sun file
        options: Geometry creation options
    """
    # Create a collection for this import
    collection = bpy.data.collections.new(f"SUN_Lattice_{data.header.block_size}")
    context.scene.collection.children.link(collection)

    # Create geometry for each block
    for block_idx, block in enumerate(data.blocks):
        block_name = f"Block_{block_idx}_L{block.level}"

        if options.import_density and data.header.has_psi:
            _create_density_geometry(
                collection, data, block_idx, block_name,
                options.density_threshold, options.visualization_mode
            )

        if options.import_links and data.header.has_links:
            _create_link_geometry(collection, data, block_idx, block_name)

        if options.import_plaquettes and data.header.has_plaquettes:
            _create_plaquette_geometry(collection, data, block_idx, block_name)


def _create_density_geometry(
    collection,
    data: importers.LatticeData,
    block_idx: int,
    block_name: str,
    threshold: float,
    mode: str
):
    """Create geometry visualizing probability density."""
    block = data.blocks[block_idx]
    if block.density is None:
        return

    positions = data.get_positions(block_idx)
    density = block.density

    # Filter by threshold
    mask = density > threshold
    filtered_positions = positions[mask]
    filtered_density = density[mask]

    if len(filtered_positions) == 0:
        return

    if mode == 'POINTS':
        _create_point_cloud(collection, filtered_positions, filtered_density, block_name)
    elif mode == 'SPHERES':
        _create_instanced_spheres(collection, filtered_positions, filtered_density, block_name)
    elif mode == 'VOLUME':
        _create_volume(collection, data, block_idx, block_name)


def _create_point_cloud(collection, positions: np.ndarray, density: np.ndarray, name: str):
    """Create a point cloud mesh with density as vertex attribute."""
    # Create mesh
    mesh = bpy.data.meshes.new(f"{name}_points")
    obj = bpy.data.objects.new(f"{name}_points", mesh)
    collection.objects.link(obj)

    # Create vertices
    bm = bmesh.new()
    for pos in positions:
        bm.verts.new(pos)
    bm.to_mesh(mesh)
    bm.free()

    # Add density as a custom attribute
    if "density" not in mesh.attributes:
        mesh.attributes.new(name="density", type='FLOAT', domain='POINT')

    density_attr = mesh.attributes["density"]
    for i, d in enumerate(density):
        density_attr.data[i].value = d


def _create_instanced_spheres(collection, positions: np.ndarray, density: np.ndarray, name: str):
    """Create point cloud with geometry nodes to instance spheres."""
    # First create the point cloud
    _create_point_cloud(collection, positions, density, name)

    obj = bpy.data.objects.get(f"{name}_points")
    if obj is None:
        return

    # Add geometry nodes modifier
    modifier = obj.modifiers.new(name="SUN_Spheres", type='NODES')

    # Create node group if it doesn't exist
    node_group = _get_or_create_sphere_instance_nodes()
    modifier.node_group = node_group


def _get_or_create_sphere_instance_nodes():
    """Get or create the geometry nodes setup for instancing spheres."""
    group_name = "SUN_SphereInstance"

    if group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    # Create new node group
    group = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')

    # Create interface
    group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = group.nodes
    links = group.links

    # Group Input
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)

    # Read density attribute
    attr_node = nodes.new('GeometryNodeInputNamedAttribute')
    attr_node.location = (-400, -150)
    attr_node.data_type = 'FLOAT'
    attr_node.inputs['Name'].default_value = "density"

    # Math node to scale density to radius
    math_node = nodes.new('ShaderNodeMath')
    math_node.location = (-200, -150)
    math_node.operation = 'MULTIPLY'
    math_node.inputs[1].default_value = 0.1  # Scale factor

    # Ico Sphere for instancing
    sphere_node = nodes.new('GeometryNodeMeshIcoSphere')
    sphere_node.location = (-200, -300)
    sphere_node.inputs['Radius'].default_value = 1.0
    sphere_node.inputs['Subdivisions'].default_value = 1

    # Instance on Points
    instance_node = nodes.new('GeometryNodeInstanceOnPoints')
    instance_node.location = (0, 0)

    # Realize Instances (optional, for better performance with many instances)
    realize_node = nodes.new('GeometryNodeRealizeInstances')
    realize_node.location = (200, 0)

    # Group Output
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (400, 0)

    # Link nodes
    links.new(input_node.outputs['Geometry'], instance_node.inputs['Points'])
    links.new(attr_node.outputs['Attribute'], math_node.inputs[0])
    links.new(math_node.outputs['Value'], instance_node.inputs['Scale'])
    links.new(sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
    links.new(instance_node.outputs['Instances'], realize_node.inputs['Geometry'])
    links.new(realize_node.outputs['Geometry'], output_node.inputs['Geometry'])

    return group


def _create_volume(collection, data: importers.LatticeData, block_idx: int, name: str):
    """Create a volume from density data (placeholder - requires OpenVDB)."""
    # For now, fall back to point cloud
    block = data.blocks[block_idx]
    if block.density is None:
        return

    positions = data.get_positions(block_idx)
    _create_point_cloud(collection, positions, block.density, f"{name}_volume")

    # TODO: Implement proper volume creation using OpenVDB or fog volume


def _create_link_geometry(collection, data: importers.LatticeData, block_idx: int, name: str):
    """Create geometry visualizing gauge links."""
    block = data.blocks[block_idx]
    positions = data.get_positions(block_idx)
    spacing = block.spacing

    if data.header.n_gauge == 1 and block.link_phases is not None:
        _create_u1_links(collection, positions, block.link_phases, spacing, name)
    elif block.link_matrices is not None:
        _create_sun_links(collection, positions, block.link_matrices, spacing, name, data.header.n_gauge)


def _create_u1_links(collection, positions: np.ndarray, phases: np.ndarray, spacing: float, name: str):
    """Create arrows for U(1) links colored by phase."""
    # Create a mesh with edges representing links
    mesh = bpy.data.meshes.new(f"{name}_links")
    obj = bpy.data.objects.new(f"{name}_links", mesh)
    collection.objects.link(obj)

    bm = bmesh.new()

    # Direction vectors for each dimension (skip time=0, use x,y,z)
    directions = [
        np.array([spacing, 0, 0]),
        np.array([0, spacing, 0]),
        np.array([0, 0, spacing]),
    ]

    # Create edges for spatial links (skip time direction mu=0)
    for site, pos in enumerate(positions):
        for mu in range(1, 4):  # x, y, z directions
            start = bm.verts.new(pos)
            end = bm.verts.new(pos + directions[mu - 1])
            bm.edges.new([start, end])

    bm.to_mesh(mesh)
    bm.free()

    # Add phase as edge attribute
    if "phase" not in mesh.attributes:
        mesh.attributes.new(name="phase", type='FLOAT', domain='EDGE')

    phase_attr = mesh.attributes["phase"]
    edge_idx = 0
    for site in range(len(positions)):
        for mu in range(1, 4):
            phase_attr.data[edge_idx].value = phases[site, mu]
            edge_idx += 1


def _create_sun_links(collection, positions: np.ndarray, matrices: np.ndarray, spacing: float, name: str, n_gauge: int):
    """Create geometry for SU(N) links (placeholder - just creates edges for now)."""
    # For SU(N), we could visualize as colored arrows based on matrix properties
    # For now, create simple edges like U(1)

    mesh = bpy.data.meshes.new(f"{name}_links")
    obj = bpy.data.objects.new(f"{name}_links", mesh)
    collection.objects.link(obj)

    bm = bmesh.new()

    directions = [
        np.array([spacing, 0, 0]),
        np.array([0, spacing, 0]),
        np.array([0, 0, spacing]),
    ]

    for site, pos in enumerate(positions):
        for mu in range(1, 4):
            start = bm.verts.new(pos)
            end = bm.verts.new(pos + directions[mu - 1])
            bm.edges.new([start, end])

    bm.to_mesh(mesh)
    bm.free()

    # TODO: Add matrix-derived attributes (trace, determinant, etc.)


def _create_plaquette_geometry(collection, data: importers.LatticeData, block_idx: int, name: str):
    """Create geometry visualizing plaquette field strength."""
    block = data.blocks[block_idx]
    if block.plaquette_traces is None:
        return

    positions = data.get_positions(block_idx)
    spacing = block.spacing

    # Create faces for each plaquette orientation
    # Orientations: (t,x), (t,y), (t,z), (x,y), (x,z), (y,z)
    # For 3D viz, we focus on spatial plaquettes: (x,y), (x,z), (y,z) = indices 3, 4, 5

    mesh = bpy.data.meshes.new(f"{name}_plaquettes")
    obj = bpy.data.objects.new(f"{name}_plaquettes", mesh)
    collection.objects.link(obj)

    bm = bmesh.new()

    # Direction vectors
    dx = np.array([spacing, 0, 0])
    dy = np.array([0, spacing, 0])
    dz = np.array([0, 0, spacing])

    # Create faces for spatial plaquettes
    for site, pos in enumerate(positions):
        # XY plaquette (index 3)
        v1 = bm.verts.new(pos)
        v2 = bm.verts.new(pos + dx)
        v3 = bm.verts.new(pos + dx + dy)
        v4 = bm.verts.new(pos + dy)
        bm.faces.new([v1, v2, v3, v4])

    bm.to_mesh(mesh)
    bm.free()

    # Add plaquette trace as face attribute
    if "field_strength" not in mesh.attributes:
        mesh.attributes.new(name="field_strength", type='FLOAT', domain='FACE')

    attr = mesh.attributes["field_strength"]
    for i, traces in enumerate(block.plaquette_traces):
        # Use XY plaquette (index 3)
        attr.data[i].value = traces[3]
