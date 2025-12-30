"""
su_n Blender Addon - Gauge Theory Visualization

This addon imports simulation data from the su_n gauge theory library
and creates visualizations of lattice structures, fields, and gauge dynamics.

Supported visualizations:
- AMR block structure with refinement levels
- Scalar field density (|psi|^2) as point clouds or volumes
- Gauge link phases (U(1)) or matrices (SU(N))
- Plaquette traces (field strength)
- Animation of time evolution

File format: .sun (custom binary)
"""

bl_info = {
    "name": "su_n Gauge Theory Importer",
    "author": "su_n Project",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > SU(N) Lattice (.sun)",
    "description": "Import gauge theory simulation data for visualization",
    "category": "Import-Export",
}

import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

# Import our modules
from . import importers
from . import geometry


class SUN_OT_import(bpy.types.Operator, ImportHelper):
    """Import SU(N) lattice simulation data"""
    bl_idname = "import_scene.sun_lattice"
    bl_label = "Import SU(N) Lattice"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".sun"
    filter_glob: StringProperty(default="*.sun*", options={'HIDDEN'})

    # Import options
    import_density: BoolProperty(
        name="Import Density",
        description="Import |psi|^2 probability density as point cloud",
        default=True,
    )
    import_links: BoolProperty(
        name="Import Links",
        description="Import gauge link data as arrows/vectors",
        default=False,
    )
    import_plaquettes: BoolProperty(
        name="Import Plaquettes",
        description="Import plaquette traces as face colors",
        default=False,
    )
    density_threshold: FloatProperty(
        name="Density Threshold",
        description="Minimum density to show (filters low values)",
        default=0.001,
        min=0.0,
        max=1.0,
    )
    visualization_mode: EnumProperty(
        name="Visualization Mode",
        description="How to visualize the data",
        items=[
            ('POINTS', "Points", "Show sites as points with density attribute"),
            ('SPHERES', "Spheres", "Instance spheres at sites scaled by density"),
            ('VOLUME', "Volume", "Create volume from density (requires OpenVDB)"),
        ],
        default='POINTS',
    )

    def execute(self, context):
        # Parse the file
        try:
            data = importers.parse_sun_file(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to parse file: {e}")
            return {'CANCELLED'}

        # Create geometry based on options
        options = geometry.GeometryOptions(
            import_density=self.import_density,
            import_links=self.import_links,
            import_plaquettes=self.import_plaquettes,
            density_threshold=self.density_threshold,
            visualization_mode=self.visualization_mode,
        )

        try:
            geometry.create_geometry(context, data, options)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create geometry: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Imported {data.num_blocks} blocks")
        return {'FINISHED'}


class SUN_PT_import_panel(bpy.types.Panel):
    """Panel for SU(N) import settings in sidebar"""
    bl_label = "SU(N) Import"
    bl_idname = "SUN_PT_import_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SU(N)"

    def draw(self, context):
        layout = self.layout
        layout.operator("import_scene.sun_lattice", text="Import .sun File")


def menu_func_import(self, context):
    self.layout.operator(SUN_OT_import.bl_idname, text="SU(N) Lattice (.sun)")


classes = (
    SUN_OT_import,
    SUN_PT_import_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
