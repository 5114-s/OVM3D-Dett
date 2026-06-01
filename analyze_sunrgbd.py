import json
import numpy as np

# Load the file
path = "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl-sam3d/SUNRGBD_train.json"
with open(path, 'r') as f:
    data = json.load(f)

annotations = data.get("annotations", [])
print(f"Total annotations: {len(annotations)}")

# Analyze first 1000 valid annotations
valid_3d = [a for a in annotations if a.get("valid3D", False)][:1000]
print(f"Valid3D (first 1000): {len(valid_3d)}")

# Extract values
center_x, center_y, center_z = [], [], []
dim_l, dim_h, dim_w = [], [], []

for a in valid_3d:
    if "center_cam" in a and a["center_cam"]:
        c = a["center_cam"]
        center_x.append(c[0])
        center_y.append(c[1])
        center_z.append(c[2])
    if "dimensions" in a and a["dimensions"]:
        d = a["dimensions"]
        dim_l.append(d[0])
        dim_h.append(d[1])
        dim_w.append(d[2])

print(f"\nCenter_X: min={min(center_x):.4f}, max={max(center_x):.4f}, mean={np.mean(center_x):.4f}")
print(f"Center_Y: min={min(center_y):.4f}, max={max(center_y):.4f}, mean={np.mean(center_y):.4f}")
print(f"Center_Z: min={min(center_z):.4f}, max={max(center_z):.4f}, mean={np.mean(center_z):.4f}")
print(f"\nDim_L: min={min(dim_l):.4f}, max={max(dim_l):.4f}, mean={np.mean(dim_l):.4f}")
print(f"Dim_H: min={min(dim_h):.4f}, max={max(dim_h):.4f}, mean={np.mean(dim_h):.4f}")
print(f"Dim_W: min={min(dim_w):.4f}, max={max(dim_w):.4f}, mean={np.mean(dim_w):.4f}")

# Sample annotations
print("\n\nSAMPLE ANNOTATIONS:")
for i, a in enumerate(valid_3d[:3]):
    print(f"\nSample {i+1}:")
    print(f"  Category: {a.get('category_name')}")
    print(f"  center_cam: {a.get('center_cam')}")
    print(f"  dimensions: {a.get('dimensions')}")
    print(f"  R_cam: {a.get('R_cam')}")
    print(f"  bbox3D_cam (first 2 corners): {a.get('bbox3D_cam')[:2] if a.get('bbox3D_cam') else 'N/A'}")
