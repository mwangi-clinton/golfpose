import argparse
import torch
from mmpose.apis import init_model
from mmengine.config import Config

def export_onnx(config_path, checkpoint_path, output_path):
    print(f"Loading config from {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Initialize the MMPose model
    print("Initializing model...")
    model = init_model(config_path, checkpoint_path, device='cpu')
    model.eval()

    # Get input shape from config
    # In MMPose, image size is usually in pipeline or model configs
    if hasattr(cfg, 'codec') and hasattr(cfg.codec, 'input_size'):
        w, h = cfg.codec.input_size
    else:
        # Fallback to standard RTMPose resolutions if not explicitly found easily
        if '256x192' in config_path:
            w, h = 192, 256
        elif '128x96' in config_path:
            w, h = 96, 128
        else:
            w, h = 192, 256
            
    print(f"Using input size: [1, 3, {h}, {w}]")
    dummy_input = torch.randn(1, 3, h, w)

    # Wrap the model so that it only returns the tensor predictions (bypassing the MM tracking)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # In MMPose 1.x, mode='tensor' returns raw tensor outputs
            return self.model(x, mode='tensor')

    wrapped_model = ONNXWrapper(model)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Export complete! You can now use onnx2tflite or onnx-tf to convert this to TFLite.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MMPose model to ONNX")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--out", type=str, default="model.onnx", help="Output .onnx path")
    args = parser.parse_args()
    
    export_onnx(args.config, args.checkpoint, args.out)
