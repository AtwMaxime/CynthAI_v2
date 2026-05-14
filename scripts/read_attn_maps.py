"""Extract attention pattern data from cross_attn and attn PNG images."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from PIL import Image

ckpt_dir = Path("checkpoints/cheater")

# Read cross_attn images to extract actual attention values
print("=== CROSS-ATTENTION MAPS (from PNG pixel data) ===")
print("Each image is a heatmap: 13 action queries x 13 state tokens")
print()

cross_dir = ckpt_dir / "cross_attn"
targets = ["update000100.png", "update000500.png", "update001000.png", "update001400.png"]

for fn in targets:
    img_path = cross_dir / fn
    if not img_path.exists():
        continue

    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    h, w, _ = arr.shape
    update = fn.replace("update", "").replace(".png", "")

    # The image is 741x942 pixels with a color bar on the right
    # We need to extract the actual 13x13 heatmap
    # The heatmap is in the left portion with cell boundaries

    # Detect the heatmap region by finding where the plot content is
    # Convert to grayscale to find the heatmap
    gray = np.mean(arr[:,:,:3], axis=2)  # [h, w]

    # Find where the colorbar starts (vertical line of constant color)
    # Look at the right side for a vertical gradient
    col_std = gray.std(axis=0)  # std across rows for each column
    # The colorbar region typically has lower std (smoother gradient)

    # Actually, let's try to read the PDF version or use a different approach
    # Just output basic image stats
    print(f"{fn}: size={arr.shape}, unique_pixels_in_heatmap_area?")

    # Try to extract just the central content by cropping margins
    # Typical matplotlib: margins ~50px on each side
    # The heatmap itself is roughly 13*cell_size with labels

    # Find non-white rows/cols (content area)
    white_thresh = 250
    non_white_mask = gray < white_thresh
    rows_with_content = np.any(non_white_mask, axis=1)
    cols_with_content = np.any(non_white_mask, axis=0)

    if rows_with_content.any() and cols_with_content.any():
        y_min, y_max = np.where(rows_with_content)[0][[0, -1]]
        x_min, x_max = np.where(cols_with_content)[0][[0, -1]]
        content = arr[y_min:y_max+1, x_min:x_max+1]
        print(f"  Content region: ({x_min},{y_min})-({x_max},{y_max}) = {content.shape}")

        # Try to read the actual heatmap as a 13x13 grid
        # The heatmap uses yellow-orange-red colormap (typically YlOrRd)
        # Just look at the red channel intensity as a proxy for attention
        red_channel = content[:,:,0].astype(float)  # [h', w']
        # In YlOrRd: low=white(255,255,255), high=red(255,0,0), mid=orange(255,165,0)
        # Red channel is always high; green channel decreases
        green = content[:,:,1].astype(float)
        # low attention = white = high green (255)
        # high attention = red/orange = low green (0-165)
        attention_proxy = 255.0 - green  # higher = more attention

        print(f"  Attention proxy (255-green): shape={attention_proxy.shape}, "
              f"range=[{attention_proxy.min():.0f}, {attention_proxy.max():.0f}], "
              f"mean={attention_proxy.mean():.1f}")

        # Downsample to 13x13 if the content area is roughly square
        if abs(content.shape[0] - content.shape[1]) < content.shape[0] * 0.3:
            # Probably the heatmap
            # Downsample to 13x13 using average pooling
            import torch
            at = torch.from_numpy(attention_proxy).unsqueeze(0).unsqueeze(0)
            downsampled = torch.nn.functional.interpolate(at, size=(13, 13), mode='area').squeeze().numpy()

            # Normalize to sum to 1 per row (each query's attention should sum to 1)
            row_sums = downsampled.sum(axis=1, keepdims=True)
            downsampled = downsampled / (row_sums + 1e-8)

            print(f"  Downsampled 13x13 attention (row-normalized):")
            state_labels = ["O0","O1","O2","O3","O4","O5","P0","P1","P2","P3","P4","P5","FL"]
            action_labels = ["M1","M2","M3","M4","T1","T2","T3","T4","S0","S1","S2","S3","S4"]

            # Print matrix
            header = "         " + "".join(f"{s:>5}" for s in state_labels)
            print(f"  {header}")
            for q in range(13):
                row_vals = downsampled[q]
                top_idx = np.argsort(row_vals)[-3:][::-1]
                top_str = ",".join(f"{state_labels[t]}({row_vals[t]:.2f})" for t in top_idx)
                print(f"  {action_labels[q]:>4}: " + "".join(f"{row_vals[s]:.3f}" for s in range(13)) + f"  top: {top_str}")

            # Per-group analysis
            own = downsampled[:, 0:6].sum(axis=1)
            opp = downsampled[:, 6:12].sum(axis=1)
            fld = downsampled[:, 12:13].sum(axis=1)
            print(f"  Own: {own.mean():.3f}, Opp: {opp.mean():.3f}, Field: {fld.mean():.3f}")
            print(f"  Top-1 per query: {[state_labels[downsampled[q].argmax()] for q in range(13)]}")
    print()

# Also check self-attention grid images
print("=== SELF-ATTENTION GRID (from attn dir) ===")
attn_dir = ckpt_dir / "attn"
for fn in sorted(attn_dir.glob("*.png"))[:2]:  # just first 2
    img = Image.open(fn).convert("RGBA")
    arr = np.array(img)
    print(f"{fn.name}: shape={arr.shape}")

print()
print("Done.")