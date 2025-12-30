"""
Shader node setups for visualizing gauge theory data.

Provides materials that map data attributes to visual properties:
- Density -> color gradient and transparency
- Phase -> hue (color wheel)
- Field strength -> intensity
"""

import bpy
import math


def create_density_material(name: str = "SUN_Density") -> bpy.types.Material:
    """Create a material that maps density attribute to color.

    Uses a blue-white-red gradient for density visualization.
    Low density -> blue, medium -> white, high -> red.
    """
    # Check if material already exists
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (100, 0)

    # Attribute node to read density
    attr = nodes.new('ShaderNodeAttribute')
    attr.location = (-400, 0)
    attr.attribute_name = "density"

    # Color ramp for density -> color mapping
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-100, 0)
    ramp.color_ramp.interpolation = 'LINEAR'

    # Set up blue -> white -> red gradient
    ramp.color_ramp.elements[0].color = (0.0, 0.2, 0.8, 1.0)  # Blue
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].color = (0.9, 0.1, 0.1, 1.0)  # Red
    ramp.color_ramp.elements[1].position = 1.0

    # Add middle element for white
    middle = ramp.color_ramp.elements.new(0.5)
    middle.color = (1.0, 1.0, 1.0, 1.0)

    # Math node to normalize density (log scale often works better)
    log_node = nodes.new('ShaderNodeMath')
    log_node.location = (-250, 0)
    log_node.operation = 'LOGARITHM'
    log_node.inputs[1].default_value = 10.0  # Base 10

    # Map range to normalize log values
    map_range = nodes.new('ShaderNodeMapRange')
    map_range.location = (-100, -150)
    map_range.inputs['From Min'].default_value = -6.0
    map_range.inputs['From Max'].default_value = 0.0
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0
    map_range.clamp = True

    # Link nodes
    links.new(attr.outputs['Fac'], log_node.inputs[0])
    links.new(log_node.outputs['Value'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Also use density for emission (makes points visible)
    links.new(ramp.outputs['Color'], principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value = 0.5

    return mat


def create_phase_material(name: str = "SUN_Phase") -> bpy.types.Material:
    """Create a material that maps phase angle to hue.

    Phase in [-pi, pi] maps to full color wheel (red -> yellow -> green -> cyan -> blue -> magenta -> red).
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (100, 0)

    # Attribute for phase
    attr = nodes.new('ShaderNodeAttribute')
    attr.location = (-400, 0)
    attr.attribute_name = "phase"

    # Map phase [-pi, pi] to [0, 1] for hue
    map_range = nodes.new('ShaderNodeMapRange')
    map_range.location = (-200, 0)
    map_range.inputs['From Min'].default_value = -math.pi
    map_range.inputs['From Max'].default_value = math.pi
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0

    # HSV node to create color from hue
    hsv = nodes.new('ShaderNodeHueSaturation')
    hsv.location = (-50, 0)
    hsv.inputs['Saturation'].default_value = 1.0
    hsv.inputs['Value'].default_value = 1.0

    # Combine HSV to RGB
    combine = nodes.new('ShaderNodeCombineHSV')
    combine.location = (-50, -150)
    combine.inputs['S'].default_value = 0.9
    combine.inputs['V'].default_value = 1.0

    links.new(attr.outputs['Fac'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], combine.inputs['H'])
    links.new(combine.outputs['Color'], principled.inputs['Base Color'])
    links.new(combine.outputs['Color'], principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value = 0.3
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat


def create_field_strength_material(name: str = "SUN_FieldStrength") -> bpy.types.Material:
    """Create a material for plaquette field strength visualization.

    Maps field strength (plaquette trace deviation from identity) to color.
    Values near N_gauge (identity) -> green (weak field)
    Deviations -> red (strong field)
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (100, 0)

    # Attribute for field strength
    attr = nodes.new('ShaderNodeAttribute')
    attr.location = (-400, 0)
    attr.attribute_name = "field_strength"

    # For U(1): trace of identity is 1, plaquettes range from -1 to 1
    # Deviation from 1 indicates field strength
    subtract = nodes.new('ShaderNodeMath')
    subtract.location = (-250, 0)
    subtract.operation = 'SUBTRACT'
    subtract.inputs[1].default_value = 1.0  # Subtract identity trace

    absolute = nodes.new('ShaderNodeMath')
    absolute.location = (-100, 0)
    absolute.operation = 'ABSOLUTE'

    # Color ramp: 0 (weak) -> green, 2 (strong) -> red
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (50, -100)
    ramp.color_ramp.elements[0].color = (0.1, 0.8, 0.1, 1.0)  # Green
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].color = (0.9, 0.1, 0.1, 1.0)  # Red
    ramp.color_ramp.elements[1].position = 1.0

    # Clamp input
    clamp = nodes.new('ShaderNodeMath')
    clamp.location = (-50, -100)
    clamp.operation = 'MINIMUM'
    clamp.inputs[1].default_value = 2.0

    links.new(attr.outputs['Fac'], subtract.inputs[0])
    links.new(subtract.outputs['Value'], absolute.inputs[0])
    links.new(absolute.outputs['Value'], clamp.inputs[0])
    links.new(clamp.outputs['Value'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat


def apply_material_to_object(obj: bpy.types.Object, material_type: str = "density"):
    """Apply appropriate material to an object based on its data type."""
    if material_type == "density":
        mat = create_density_material()
    elif material_type == "phase":
        mat = create_phase_material()
    elif material_type == "field_strength":
        mat = create_field_strength_material()
    else:
        return

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
