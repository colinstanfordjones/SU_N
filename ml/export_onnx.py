import torch
import torch.nn as nn
from model import VAE
import os

class EncoderOnly(nn.Module):
    """Wrapper to export only the encoder part of the VAE."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        mu, _ = self.vae.encode(x)
        return mu

class DecoderOnly(nn.Module):
    """Wrapper to export only the decoder part of the VAE."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z)

def export():
    n_field = 1
    latent_dim = 1024
    block_size = 16
    
    # 1. Initialize VAE
    vae = VAE(n_field=n_field, latent_dim=latent_dim)
    
    # Try to load existing weights if available
    vae_path = "ml/checkpoints/vae_final.pt"
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
        print(f"Loaded weights from {vae_path}")
    else:
        print("Using random weights for export test.")

    vae.eval()

    # 2. Export Encoder
    print("Exporting Encoder to ONNX...")
    encoder = EncoderOnly(vae)
    dummy_input = torch.randn(1, n_field * 2, block_size, block_size, block_size, block_size)
    
    torch.onnx.export(
        encoder,
        dummy_input,
        "ml/checkpoints/vae_encoder.onnx",
        export_params=True,
        opset_version=17, # opset 17+ has good support for 3D/4D ops
        do_constant_folding=True,
        input_names=['input_block'],
        output_names=['latent_vector'],
        dynamic_axes={'input_block': {0: 'batch_size'}, 'latent_vector': {0: 'batch_size'}}
    )
    print("Successfully exported Encoder to ml/checkpoints/vae_encoder.onnx")

    # 3. Export Decoder
    print("Exporting Decoder to ONNX...")
    decoder = DecoderOnly(vae)
    dummy_latent = torch.randn(1, latent_dim)
    
    torch.onnx.export(
        decoder,
        dummy_latent,
        "ml/checkpoints/vae_decoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['latent_vector'],
        output_names=['output_block'],
        dynamic_axes={'latent_vector': {0: 'batch_size'}, 'output_block': {0: 'batch_size'}}
    )
    print("Successfully exported Decoder to ml/checkpoints/vae_decoder.onnx")

if __name__ == "__main__":
    export()
