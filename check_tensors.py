import numpy as np
import os

print("="*70)
print("📊 CHECKING PROCESSED DATA")
print("="*70)

# Check intermediate processed files
processed_files = {
    "data/processed/train.npy": "Raw normalized train",
    "data/processed/val.npy": "Raw normalized val",
    "data/processed/test.npy": "Raw normalized test",
    "data/processed/normalized.npy": "All normalized data",
}

print("\n🔍 Intermediate files:")
for path, desc in processed_files.items():
    if os.path.exists(path):
        data = np.load(path)
        print(f"\n  {desc}: {path}")
        print(f"    Shape: {data.shape}")
        print(f"    Dtype: {data.dtype}")
        print(f"    Range: [{np.min(data):.6f}, {np.max(data):.6f}]")
        print(f"    Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}")
        print(f"    Unique values: {len(np.unique(data[:100]))}")
    else:
        print(f"\n  ❌ {desc}: NOT FOUND")

# Check tensor files
print("\n\n🔍 Tensor files (used for training):")
for split in ["train", "val", "test"]:
    X_path = f"data/processed/tensors/{split}_X.npy"
    y_path = f"data/processed/tensors/{split}_y.npy"
    
    if os.path.exists(X_path):
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"\n  {split.upper()}:")
        print(f"    X.shape: {X.shape}")
        print(f"    X range: [{np.min(X):.6f}, {np.max(X):.6f}]")
        print(f"    X unique values: {len(np.unique(X))}")
        print(f"    y.shape: {y.shape}")
        print(f"    y range: [{np.min(y):.6f}, {np.max(y):.6f}]")
        print(f"    y unique values: {len(np.unique(y))}")

print("\n" + "="*70)
