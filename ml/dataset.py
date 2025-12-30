import numpy as np
import torch
from torch.utils.data import Dataset
import struct
import os

class SunDataset(Dataset):
    """
    Dataset for loading 4D AMR blocks from .sun files.
    Returns tensors of shape (N_field * 2, 16, 16, 16, 16)
    """
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.sun')]
        self.block_data = []
        
        for f in self.files:
            with open(f, 'rb') as rb:
                header_data = rb.read(40)
                if len(header_data) < 40: continue
                
                magic, version, num_blocks, block_size, n_gauge, n_field, base_spacing, time_slice, flags = struct.unpack('<4sIIIIIdII', header_data)
                if magic != b'SUN\x00':
                    continue
                
                for i in range(num_blocks):
                    offset = rb.tell()
                    self.block_data.append({
                        'file': f,
                        'offset': offset,
                        'block_size': block_size,
                        'n_field': n_field,
                        'n_gauge': n_gauge,
                        'flags': flags
                    })
                    
                    # Skip block header (40 bytes)
                    rb.seek(40, 1)
                    
                    volume = block_size**4
                    # Skip density if present
                    if flags & 1: # FLAG_HAS_PSI
                        rb.seek(volume * 8, 1)
                    
                    # Skip full psi (we'll read it in __getitem__)
                    has_full = flags & 8 # FLAG_HAS_FULL_PSI
                    if has_full:
                        full_psi_start = rb.tell()
                        self.block_data[-1]['full_psi_offset'] = full_psi_start
                        rb.seek(volume * n_field * 2 * 8, 1)
                    
                    # Skip links
                    if flags & 2: # FLAG_HAS_LINKS
                        if n_gauge == 1:
                            rb.seek(volume * 4 * 8, 1)
                        else:
                            rb.seek(volume * 4 * (n_gauge**2 * 2) * 8, 1)
                    
                    # Skip plaquettes
                    if flags & 4: # FLAG_HAS_PLAQUETTES
                        rb.seek(volume * 6 * 8, 1)

    def __len__(self):
        return len(self.block_data)

    def __getitem__(self, idx):
        info = self.block_data[idx]
        if not (info['flags'] & 8):
            return None
            
        with open(info['file'], 'rb') as rb:
            rb.seek(0, 2)
            file_size = rb.tell()
            
            rb.seek(info['full_psi_offset'])
            
            s = info['block_size']
            volume = s**4
            n_field = info['n_field']
            
            expected_bytes = volume * n_field * 2 * 8
            data = rb.read(expected_bytes)
            
            if len(data) != expected_bytes:
                print(f"Error reading block {idx} from {info['file']}")
                print(f"Offset: {info['full_psi_offset']}")
                print(f"File size: {file_size}")
                print(f"Expected bytes: {expected_bytes}")
                print(f"Read bytes: {len(data)}")
                print(f"Short by: {expected_bytes - len(data)}")
            
            # Read as complex pairs (re, im)
            psi = np.frombuffer(data, dtype='<f8').copy()
            
            # Shape: (T, X, Y, Z, N_field, 2)
            psi = psi.reshape(s, s, s, s, n_field, 2)
            
            # Transpose to (N_field, 2, T, X, Y, Z) then reshape to (N_field*2, T, X, Y, Z)
            psi = psi.transpose(4, 5, 0, 1, 2, 3).reshape(n_field * 2, s, s, s, s)
            
            return torch.from_numpy(psi).float()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ds = SunDataset(sys.argv[1])
        print(f"Found {len(ds)} blocks")
        if len(ds) > 0:
            sample = ds[0]
            print(f"Sample shape: {sample.shape}")
