"""
Animation and caching support for time-evolving lattice data.

Handles:
- Frame-based file loading (.sun0001, .sun0002, etc.)
- Data caching to avoid redundant file reads
- Frame change handlers for Blender timeline
- Memory management for large sequences
"""

import bpy
from bpy.app.handlers import persistent
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from . import importers
from . import geometry


@dataclass
class CacheEntry:
    """Cached data for a single frame."""
    frame: int
    data: importers.LatticeData
    last_access: float = 0.0


@dataclass
class AnimationCache:
    """LRU cache for animation frame data."""
    base_path: str
    max_entries: int = 50
    entries: Dict[int, CacheEntry] = field(default_factory=dict)
    access_counter: float = 0.0

    def get(self, frame: int) -> Optional[importers.LatticeData]:
        """Get cached data for frame, or None if not cached."""
        if frame in self.entries:
            self.access_counter += 1
            self.entries[frame].last_access = self.access_counter
            return self.entries[frame].data
        return None

    def put(self, frame: int, data: importers.LatticeData):
        """Add data to cache, evicting old entries if needed."""
        self.access_counter += 1

        # Evict least recently used if at capacity
        while len(self.entries) >= self.max_entries:
            oldest_frame = min(self.entries.keys(),
                              key=lambda f: self.entries[f].last_access)
            del self.entries[oldest_frame]

        self.entries[frame] = CacheEntry(
            frame=frame,
            data=data,
            last_access=self.access_counter
        )

    def clear(self):
        """Clear all cached entries."""
        self.entries.clear()

    def get_frame_path(self, frame: int) -> str:
        """Get file path for a specific frame."""
        return f"{self.base_path}{frame:04d}"


# Global cache storage (keyed by collection name)
_animation_caches: Dict[str, AnimationCache] = {}


def get_cache(collection_name: str) -> Optional[AnimationCache]:
    """Get the animation cache for a collection."""
    return _animation_caches.get(collection_name)


def create_cache(collection_name: str, base_path: str, max_entries: int = 50) -> AnimationCache:
    """Create a new animation cache for a collection."""
    cache = AnimationCache(base_path=base_path, max_entries=max_entries)
    _animation_caches[collection_name] = cache
    return cache


def clear_cache(collection_name: str):
    """Clear the cache for a collection."""
    if collection_name in _animation_caches:
        _animation_caches[collection_name].clear()


def clear_all_caches():
    """Clear all animation caches."""
    for cache in _animation_caches.values():
        cache.clear()
    _animation_caches.clear()


def load_animation_sequence(
    context,
    base_path: str,
    options: geometry.GeometryOptions,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
    cache_size: int = 50
) -> str:
    """Load an animation sequence from numbered files.

    Args:
        context: Blender context
        base_path: Base path (e.g., "output.sun" for output.sun0001, etc.)
        options: Geometry creation options
        start_frame: First frame number to load
        end_frame: Last frame number (None = auto-detect)
        cache_size: Maximum frames to keep in cache

    Returns:
        Collection name for the loaded animation
    """
    # Find all frame files
    frame_files = importers.find_animation_files(base_path)
    if not frame_files:
        raise ValueError(f"No animation frames found for {base_path}")

    if end_frame is None:
        end_frame = len(frame_files)

    # Create collection
    collection_name = f"SUN_Anim_{Path(base_path).stem}"
    if collection_name in bpy.data.collections:
        # Remove existing
        bpy.data.collections.remove(bpy.data.collections[collection_name])

    collection = bpy.data.collections.new(collection_name)
    context.scene.collection.children.link(collection)

    # Create cache
    cache = create_cache(collection_name, base_path, cache_size)

    # Load first frame
    first_frame_path = cache.get_frame_path(start_frame)
    try:
        data = importers.parse_sun_file(first_frame_path)
        cache.put(start_frame, data)

        # Create geometry for first frame
        geometry.create_geometry(context, data, options)

        # Move objects to our collection
        for obj in list(context.scene.collection.objects):
            if obj.name.startswith("Block_"):
                context.scene.collection.objects.unlink(obj)
                collection.objects.link(obj)

    except Exception as e:
        raise ValueError(f"Failed to load first frame: {e}")

    # Set up frame range
    context.scene.frame_start = start_frame
    context.scene.frame_end = end_frame
    context.scene.frame_current = start_frame

    # Store animation info in collection for frame handler
    collection["sun_base_path"] = base_path
    collection["sun_start_frame"] = start_frame
    collection["sun_end_frame"] = end_frame

    return collection_name


def update_frame(collection_name: str, frame: int, context):
    """Update geometry for a specific frame.

    Called by frame change handler or manually.
    """
    cache = get_cache(collection_name)
    if cache is None:
        return

    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return

    # Try to get from cache
    data = cache.get(frame)

    # If not in cache, try to load from file
    if data is None:
        frame_path = cache.get_frame_path(frame)
        try:
            data = importers.parse_sun_file(frame_path)
            cache.put(frame, data)
        except FileNotFoundError:
            # Frame doesn't exist, keep current geometry
            return
        except Exception as e:
            print(f"Failed to load frame {frame}: {e}")
            return

    # Update geometry data in-place
    _update_geometry_data(collection, data)


def _update_geometry_data(collection, data: importers.LatticeData):
    """Update existing geometry with new frame data.

    Updates vertex attributes without recreating meshes.
    """
    for block_idx, block in enumerate(data.blocks):
        # Find corresponding mesh object
        obj_name = f"Block_{block_idx}_L{block.level}_points"
        obj = None

        for o in collection.objects:
            if o.name.startswith(f"Block_{block_idx}"):
                obj = o
                break

        if obj is None or obj.type != 'MESH':
            continue

        mesh = obj.data

        # Update density attribute if present
        if block.density is not None and "density" in mesh.attributes:
            density_attr = mesh.attributes["density"]
            for i, d in enumerate(block.density):
                if i < len(density_attr.data):
                    density_attr.data[i].value = d

        # Mark mesh as updated
        mesh.update()


@persistent
def frame_change_handler(scene):
    """Handler called when frame changes in timeline.

    Updates all animated SUN collections to show correct frame data.
    """
    frame = scene.frame_current

    # Find all SUN animation collections
    for collection in bpy.data.collections:
        if collection.name.startswith("SUN_Anim_"):
            if "sun_base_path" in collection:
                start = collection.get("sun_start_frame", 1)
                end = collection.get("sun_end_frame", 250)

                if start <= frame <= end:
                    update_frame(collection.name, frame, bpy.context)


def register_handlers():
    """Register frame change handler."""
    if frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(frame_change_handler)


def unregister_handlers():
    """Unregister frame change handler."""
    if frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(frame_change_handler)


# Operator for loading animations
class SUN_OT_load_animation(bpy.types.Operator):
    """Load an animation sequence of SU(N) lattice data"""
    bl_idname = "import_scene.sun_animation"
    bl_label = "Load SU(N) Animation"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(
        subtype='FILE_PATH',
        description="Base path for animation files"
    )
    cache_size: bpy.props.IntProperty(
        name="Cache Size",
        description="Maximum frames to keep in memory",
        default=50,
        min=1,
        max=500
    )

    def execute(self, context):
        options = geometry.GeometryOptions(
            import_density=True,
            import_links=False,
            density_threshold=0.001,
            visualization_mode='POINTS'
        )

        try:
            collection_name = load_animation_sequence(
                context,
                self.filepath,
                options,
                cache_size=self.cache_size
            )
            self.report({'INFO'}, f"Loaded animation: {collection_name}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def register():
    bpy.utils.register_class(SUN_OT_load_animation)
    register_handlers()


def unregister():
    unregister_handlers()
    clear_all_caches()
    bpy.utils.unregister_class(SUN_OT_load_animation)
