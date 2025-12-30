"""
Geometry Nodes setups for procedural lattice visualization.

Provides reusable node groups for:
- Density-based sphere instancing
- Link arrow visualization
- Plaquette face rendering
- Phase-to-color mapping
"""

import bpy
import math


def get_or_create_node_group(name: str, creator_func) -> bpy.types.NodeTree:
    """Get existing node group or create it using creator function."""
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    return creator_func(name)


def create_density_spheres_nodes(name: str = "SUN_DensitySpheres") -> bpy.types.NodeTree:
    """Create node group that instances spheres scaled by density.

    Inputs:
        Geometry: Point cloud with 'density' attribute
        Base Scale: Base size for spheres
        Min Density: Minimum density to show

    Outputs:
        Geometry: Mesh with instanced spheres
    """
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')

    # Create interface
    group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket('Base Scale', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Min Density', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = group.nodes
    links = group.links

    # Input
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-600, 0)

    # Read density attribute
    density_attr = nodes.new('GeometryNodeInputNamedAttribute')
    density_attr.location = (-600, -200)
    density_attr.data_type = 'FLOAT'
    density_attr.inputs['Name'].default_value = "density"

    # Compare for threshold filtering
    compare = nodes.new('FunctionNodeCompare')
    compare.location = (-400, -200)
    compare.data_type = 'FLOAT'
    compare.operation = 'GREATER_THAN'

    # Delete points below threshold
    delete_geom = nodes.new('GeometryNodeDeleteGeometry')
    delete_geom.location = (-200, 0)
    delete_geom.domain = 'POINT'
    delete_geom.mode = 'ALL'

    # Boolean NOT (delete where comparison is false)
    bool_not = nodes.new('FunctionNodeBooleanMath')
    bool_not.location = (-300, -200)
    bool_not.operation = 'NOT'

    # Scale calculation: sqrt(density) * base_scale
    sqrt_node = nodes.new('ShaderNodeMath')
    sqrt_node.location = (-200, -300)
    sqrt_node.operation = 'SQRT'

    scale_mult = nodes.new('ShaderNodeMath')
    scale_mult.location = (0, -300)
    scale_mult.operation = 'MULTIPLY'

    # Ico sphere for instancing
    sphere = nodes.new('GeometryNodeMeshIcoSphere')
    sphere.location = (-200, -450)
    sphere.inputs['Radius'].default_value = 1.0
    sphere.inputs['Subdivisions'].default_value = 2

    # Instance on points
    instance = nodes.new('GeometryNodeInstanceOnPoints')
    instance.location = (200, 0)

    # Realize instances
    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (400, 0)

    # Output
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (600, 0)

    # Link everything
    links.new(input_node.outputs['Geometry'], delete_geom.inputs['Geometry'])
    links.new(input_node.outputs['Min Density'], compare.inputs['B'])
    links.new(input_node.outputs['Base Scale'], scale_mult.inputs[1])

    links.new(density_attr.outputs['Attribute'], compare.inputs['A'])
    links.new(density_attr.outputs['Attribute'], sqrt_node.inputs[0])

    links.new(compare.outputs['Result'], bool_not.inputs[0])
    links.new(bool_not.outputs['Boolean'], delete_geom.inputs['Selection'])

    links.new(sqrt_node.outputs['Value'], scale_mult.inputs[0])

    links.new(delete_geom.outputs['Geometry'], instance.inputs['Points'])
    links.new(sphere.outputs['Mesh'], instance.inputs['Instance'])
    links.new(scale_mult.outputs['Value'], instance.inputs['Scale'])

    links.new(instance.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], output_node.inputs['Geometry'])

    return group


def create_link_arrows_nodes(name: str = "SUN_LinkArrows") -> bpy.types.NodeTree:
    """Create node group that converts link edges to arrows.

    Inputs:
        Geometry: Edge mesh with 'phase' attribute on edges
        Arrow Scale: Scale for arrow cones

    Outputs:
        Geometry: Mesh with arrow cones at edge midpoints
    """
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')

    group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket('Arrow Scale', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = group.nodes
    links = group.links

    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)

    # Mesh to Points (convert edge midpoints to points)
    to_points = nodes.new('GeometryNodeMeshToPoints')
    to_points.location = (-200, 0)
    to_points.mode = 'EDGES'  # One point per edge

    # Cone for arrows
    cone = nodes.new('GeometryNodeMeshCone')
    cone.location = (-200, -200)
    cone.inputs['Vertices'].default_value = 8
    cone.inputs['Radius Bottom'].default_value = 0.3
    cone.inputs['Radius Top'].default_value = 0.0
    cone.inputs['Depth'].default_value = 1.0

    # Instance cones on edge midpoints
    instance = nodes.new('GeometryNodeInstanceOnPoints')
    instance.location = (0, 0)

    # Realize
    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (200, 0)

    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (400, 0)

    links.new(input_node.outputs['Geometry'], to_points.inputs['Mesh'])
    links.new(input_node.outputs['Arrow Scale'], instance.inputs['Scale'])
    links.new(to_points.outputs['Points'], instance.inputs['Points'])
    links.new(cone.outputs['Mesh'], instance.inputs['Instance'])
    links.new(instance.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], output_node.inputs['Geometry'])

    return group


def create_phase_color_nodes(name: str = "SUN_PhaseColor") -> bpy.types.NodeTree:
    """Create node group that maps phase angle to RGB color.

    Inputs:
        Phase: Float in [-pi, pi]

    Outputs:
        Color: RGB color (full hue rotation)
    """
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')

    group.interface.new_socket('Phase', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Color', in_out='OUTPUT', socket_type='NodeSocketColor')

    nodes = group.nodes
    links = group.links

    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)

    # Map [-pi, pi] to [0, 1]
    map_range = nodes.new('ShaderNodeMapRange')
    map_range.location = (-200, 0)
    map_range.inputs['From Min'].default_value = -math.pi
    map_range.inputs['From Max'].default_value = math.pi
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0

    # Combine HSV (hue from phase, full saturation and value)
    combine_hsv = nodes.new('ShaderNodeCombineHSV')
    combine_hsv.location = (0, 0)
    combine_hsv.inputs['S'].default_value = 0.9
    combine_hsv.inputs['V'].default_value = 1.0

    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (200, 0)

    links.new(input_node.outputs['Phase'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], combine_hsv.inputs['H'])
    links.new(combine_hsv.outputs['Color'], output_node.inputs['Color'])

    return group


def create_density_volume_nodes(name: str = "SUN_DensityVolume") -> bpy.types.NodeTree:
    """Create node group for volume-like density visualization.

    Uses points with varying opacity/emission based on density.
    This is a workaround until proper OpenVDB support is implemented.

    Inputs:
        Geometry: Point cloud with 'density' attribute
        Emission Strength: Base emission multiplier

    Outputs:
        Geometry: Points with density-based material properties
    """
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')

    group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket('Emission Strength', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = group.nodes
    links = group.links

    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)

    # Read density
    density_attr = nodes.new('GeometryNodeInputNamedAttribute')
    density_attr.location = (-400, -150)
    density_attr.data_type = 'FLOAT'
    density_attr.inputs['Name'].default_value = "density"

    # Store emission attribute for shader
    store_emission = nodes.new('GeometryNodeStoreNamedAttribute')
    store_emission.location = (-100, 0)
    store_emission.data_type = 'FLOAT'
    store_emission.domain = 'POINT'
    store_emission.inputs['Name'].default_value = "emission"

    # Multiply density by emission strength
    mult = nodes.new('ShaderNodeMath')
    mult.location = (-250, -150)
    mult.operation = 'MULTIPLY'

    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (100, 0)

    links.new(input_node.outputs['Geometry'], store_emission.inputs['Geometry'])
    links.new(input_node.outputs['Emission Strength'], mult.inputs[1])
    links.new(density_attr.outputs['Attribute'], mult.inputs[0])
    links.new(mult.outputs['Value'], store_emission.inputs['Value'])
    links.new(store_emission.outputs['Geometry'], output_node.inputs['Geometry'])

    return group


def create_all_node_groups():
    """Create all standard node groups for the addon."""
    get_or_create_node_group("SUN_DensitySpheres", create_density_spheres_nodes)
    get_or_create_node_group("SUN_LinkArrows", create_link_arrows_nodes)
    get_or_create_node_group("SUN_PhaseColor", create_phase_color_nodes)
    get_or_create_node_group("SUN_DensityVolume", create_density_volume_nodes)


def apply_density_spheres(obj: bpy.types.Object, base_scale: float = 0.1, min_density: float = 0.001):
    """Apply density spheres modifier to an object."""
    node_group = get_or_create_node_group("SUN_DensitySpheres", create_density_spheres_nodes)

    modifier = obj.modifiers.new(name="SUN_DensitySpheres", type='NODES')
    modifier.node_group = node_group

    # Set input values
    modifier["Socket_1"] = base_scale  # Base Scale
    modifier["Socket_2"] = min_density  # Min Density


def apply_link_arrows(obj: bpy.types.Object, arrow_scale: float = 0.05):
    """Apply link arrows modifier to an edge mesh."""
    node_group = get_or_create_node_group("SUN_LinkArrows", create_link_arrows_nodes)

    modifier = obj.modifiers.new(name="SUN_LinkArrows", type='NODES')
    modifier.node_group = node_group

    modifier["Socket_1"] = arrow_scale  # Arrow Scale
