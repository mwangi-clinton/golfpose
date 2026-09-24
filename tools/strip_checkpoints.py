import os
import glob
import torch
import argparse

def strip_checkpoints(folder_path):
    # Find all .pth files in the folder
    checkpoints = glob.glob(os.path.join(folder_path, '*.pth'))
    
    if not checkpoints:
        print(f"No .pth files found in {folder_path}")
        return

    out_folder = os.path.join(folder_path, "published_models")
    os.makedirs(out_folder, exist_ok=True)

    for ckpt_path in checkpoints:
        # Skip already published models to avoid loops
        if "published" in ckpt_path:
            continue
            
        print(f"Processing: {ckpt_path}")
        
        # Load the heavy checkpoint (map_location='cpu' prevents GPU memory spikes)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # A dictionary to hold our stripped, lightweight checkpoint
        published_ckpt = {}
        
        # Different codebases save weights under different keys
        if 'state_dict' in checkpoint:
            # MMEngine (2D RTMPose) format
            published_ckpt['state_dict'] = checkpoint['state_dict']
            if 'meta' in checkpoint:
                published_ckpt['meta'] = checkpoint['meta'] # Keep config metadata
        elif 'model_pos' in checkpoint:
            # GolfPose 3D Lifter format
            published_ckpt['model_pos'] = checkpoint['model_pos']
            if 'args' in checkpoint:
                published_ckpt['args'] = checkpoint['args']
        else:
            # Maybe it is ALREADY just a state_dict
            published_ckpt = checkpoint
            
        # Save the lightweight model
        basename = os.path.basename(ckpt_path)
        out_name = basename.replace('.pth', '_published.pth')
        out_path = os.path.join(out_folder, out_name)
        
        torch.save(published_ckpt, out_path)
        
        # Compare sizes
        old_size = os.path.getsize(ckpt_path) / (1024 * 1024)
        new_size = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  -> Saved to {out_path}")
        print(f"  -> Size reduced from {old_size:.2f} MB to {new_size:.2f} MB\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip optimizer states from checkpoints to make them lightweight.")
    parser.add_argument("folder", type=str, help="Folder containing the .pth checkpoints")
    args = parser.parse_args()
    
    strip_checkpoints(args.folder)
