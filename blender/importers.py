"""
Binary format parser for .sun files.

File format (little-endian):
    Header (40 bytes):
        magic: 4 bytes ("SUN\0")
        version: u32
        num_blocks: u32
        block_size: u32
        n_gauge: u32
        n_field: u32
        base_spacing: f64
        time_slice: u32 (0xFFFFFFFF for all times)
        flags: u32 (bit 0: has_psi, bit 1: has_links, bit 2: has_plaquettes)

    Block Header (40 bytes per block):
        level: u8
        padding: 7 bytes
        origin: 4 x u64

    Per-block data (depends on flags):
        if has_psi: [volume] f64 (density at each site)
        if has_links: [volume * 4] f64 (phases for U(1)) or matrix elements
        if has_plaquettes: [volume * 6] f64 (traces for each orientation)
"""

import struct
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


# Flag constants
FLAG_HAS_PSI = 1 << 0
FLAG_HAS_LINKS = 1 << 1
FLAG_HAS_PLAQUETTES = 1 << 2


@dataclass
class Header:
    """File header containing global simulation parameters."""
    version: int
    num_blocks: int
    block_size: int
    n_gauge: int
    n_field: int
    base_spacing: float
    time_slice: int  # 0xFFFFFFFF means all times
    flags: int

    @property
    def has_psi(self) -> bool:
        return bool(self.flags & FLAG_HAS_PSI)

    @property
    def has_links(self) -> bool:
        return bool(self.flags & FLAG_HAS_LINKS)

    @property
    def has_plaquettes(self) -> bool:
        return bool(self.flags & FLAG_HAS_PLAQUETTES)

    @property
    def volume(self) -> int:
        """Number of sites per block (block_size^4 for 4D)."""
        return self.block_size ** 4


@dataclass
class Block:
    """A single AMR block with its data."""
    level: int
    origin: tuple  # (t, x, y, z) in level coordinates
    spacing: float  # Physical spacing = base_spacing / 2^level

    # Optional data arrays (present based on header flags)
    density: Optional[np.ndarray] = None  # [volume] f64
    link_phases: Optional[np.ndarray] = None  # [volume, 4] f64 for U(1)
    link_matrices: Optional[np.ndarray] = None  # [volume, 4, n_gauge, n_gauge, 2] for SU(N)
    plaquette_traces: Optional[np.ndarray] = None  # [volume, 6] f64


@dataclass
class LatticeData:
    """Complete parsed lattice data from a .sun file."""
    header: Header
    blocks: List[Block]

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def get_positions(self, block_idx: int) -> np.ndarray:
        """Get physical positions of all sites in a block.

        Returns:
            Array of shape [volume, 3] with (x, y, z) positions.
            Time coordinate is omitted for 3D visualization.
        """
        block = self.blocks[block_idx]
        block_size = self.header.block_size
        volume = self.header.volume
        spacing = block.spacing

        positions = np.zeros((volume, 3), dtype=np.float32)

        for site in range(volume):
            # Decode linear index to 4D coordinates
            t = site % block_size
            x = (site // block_size) % block_size
            y = (site // (block_size ** 2)) % block_size
            z = site // (block_size ** 3)

            # Physical position (skip time for 3D viz)
            positions[site, 0] = (block.origin[1] + x) * spacing
            positions[site, 1] = (block.origin[2] + y) * spacing
            positions[site, 2] = (block.origin[3] + z) * spacing

        return positions


def parse_sun_file(filepath: str) -> LatticeData:
    """Parse a .sun binary file.

    Args:
        filepath: Path to the .sun file

    Returns:
        LatticeData containing header and all blocks

    Raises:
        ValueError: If file format is invalid
    """
    with open(filepath, 'rb') as f:
        # Read and validate header
        header = _parse_header(f)

        # Read blocks
        blocks = []
        for _ in range(header.num_blocks):
            block = _parse_block(f, header)
            blocks.append(block)

        return LatticeData(header=header, blocks=blocks)


def _parse_header(f) -> Header:
    """Parse the 40-byte file header."""
    # Read header bytes
    header_bytes = f.read(40)
    if len(header_bytes) < 40:
        raise ValueError("File too short for header")

    # Unpack: 4s (magic) + 7I (u32s) + d (f64)
    magic, version, num_blocks, block_size, n_gauge, n_field, \
        base_spacing, time_slice, flags = struct.unpack('<4sIIIIIdII', header_bytes)

    # Validate magic
    if magic != b'SUN\x00':
        raise ValueError(f"Invalid magic: expected 'SUN\\0', got {magic!r}")

    if version != 1:
        raise ValueError(f"Unsupported version: {version}")

    return Header(
        version=version,
        num_blocks=num_blocks,
        block_size=block_size,
        n_gauge=n_gauge,
        n_field=n_field,
        base_spacing=base_spacing,
        time_slice=time_slice,
        flags=flags,
    )


def _parse_block(f, header: Header) -> Block:
    """Parse a single block with its data."""
    # Read block header (40 bytes)
    block_header = f.read(40)
    if len(block_header) < 40:
        raise ValueError("File too short for block header")

    # Unpack: B (u8 level) + 7x (padding) + 4Q (4 u64 origin)
    level, origin_t, origin_x, origin_y, origin_z = struct.unpack('<B7x4Q', block_header)

    spacing = header.base_spacing / (2 ** level)

    block = Block(
        level=level,
        origin=(origin_t, origin_x, origin_y, origin_z),
        spacing=spacing,
    )

    volume = header.volume

    # Read density if present
    if header.has_psi:
        density_bytes = f.read(volume * 8)
        block.density = np.frombuffer(density_bytes, dtype=np.float64)

    # Read link data if present
    if header.has_links:
        if header.n_gauge == 1:
            # U(1): phases only
            phases_bytes = f.read(volume * 4 * 8)
            block.link_phases = np.frombuffer(phases_bytes, dtype=np.float64).reshape(volume, 4)
        else:
            # SU(N): full matrices (re, im pairs)
            matrix_size = header.n_gauge ** 2 * 2
            matrices_bytes = f.read(volume * 4 * matrix_size * 8)
            matrices = np.frombuffer(matrices_bytes, dtype=np.float64)
            # Reshape to [volume, 4, n_gauge, n_gauge, 2] (last dim is re/im)
            block.link_matrices = matrices.reshape(volume, 4, header.n_gauge, header.n_gauge, 2)

    # Read plaquette data if present
    if header.has_plaquettes:
        plaq_bytes = f.read(volume * 6 * 8)
        block.plaquette_traces = np.frombuffer(plaq_bytes, dtype=np.float64).reshape(volume, 6)

    return block


def find_animation_files(base_path: str) -> List[str]:
    """Find all animation frame files matching a base path.

    Args:
        base_path: Base path like "output.sun" - will find output.sun0001, etc.

    Returns:
        Sorted list of file paths for each frame
    """
    base = Path(base_path)
    parent = base.parent
    stem = base.stem
    suffix = base.suffix

    # Look for files like stem + suffix + NNNN
    pattern = f"{stem}{suffix}*"
    files = sorted(parent.glob(pattern))

    # Filter to only numbered frames
    frame_files = []
    for f in files:
        # Check if the part after .sun is all digits
        extra = f.name[len(stem) + len(suffix):]
        if extra.isdigit():
            frame_files.append(str(f))

    return frame_files
