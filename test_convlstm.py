import numpy as np
import torch
import os
from models.convlstm import ConvLSTMModel
from training.metrics import rmse
import yaml


def test_convlstm_on_preprocessed_data():
    """
    Test ConvLSTM model on preprocessed spatial-temporal climate data
    """
    
    print("\n" + "="*60)
    print("🧪 CONVLSTM MODEL TEST ON PREPROCESSED DATA")
    print("="*60)
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load preprocessed test data
    print("\n📂 Loading preprocessed test data...")
    test_X_path = "data/processed/tensors/test_X.npy"
    test_y_path = "data/processed/tensors/test_y.npy"
    
    if not os.path.exists(test_X_path) or not os.path.exists(test_y_path):
        print("❌ Test data not found. Running preprocessing first...")
        print("   Try: python main.py --preprocess")
        return
    
    # Load data
    X_raw = np.load(test_X_path)  # (B, T, C, H, W)
    y_raw = np.load(test_y_path)  # (B, C, H, W)
    
    print(f"✓ X shape: {X_raw.shape}")
    print(f"✓ y shape: {y_raw.shape}")
    
    # Convert to tensors
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw, dtype=torch.float32)
    
    print(f"  X value range: [{np.min(X_raw):.4f}, {np.max(X_raw):.4f}]")
    print(f"  y value range: [{np.min(y_raw):.4f}, {np.max(y_raw):.4f}]")
    
    # Create model
    print(f"\n🧠 Creating ConvLSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMModel(config)
    model.to(device)
    model.eval()
    print(f"✓ Model on {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test on batches
    print(f"\n⚙️  Running inference on {len(X_tensor)} test samples...")
    batch_size = 8
    all_preds = []
    all_targets = []
    total_rmse = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[i:i+batch_size].to(device)
            y_batch = y_tensor[i:i+batch_size].to(device)
            
            # Forward pass
            try:
                preds = model(X_batch)  # (B, C, H, W)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                
                # Calculate RMSE for this batch
                batch_rmse = rmse(preds, y_batch).item()
                total_rmse += batch_rmse
                num_batches += 1
                
                if (i // batch_size) % max(1, (len(X_tensor) // batch_size) // 10) == 0:
                    print(f"  Batch {i//batch_size}/{len(X_tensor)//batch_size}: RMSE = {batch_rmse:.6f}")
                    
            except Exception as e:
                print(f"❌ Error on batch {i//batch_size}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if num_batches == 0:
        print("❌ No batches were processed!")
        return
    
    # Aggregate results
    all_preds = np.concatenate(all_preds, axis=0)  # (B, C, H, W)
    all_targets = np.concatenate(all_targets, axis=0)  # (B, C, H, W)
    
    mean_rmse = total_rmse / num_batches
    
    # Flatten for additional metrics
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    mae = np.mean(np.abs(preds_flat - targets_flat))
    std_error = np.std(preds_flat - targets_flat)
    
    # Correlation (safe)
    try:
        # Only compute on non-constant data
        pred_std = np.std(preds_flat)
        targ_std = np.std(targets_flat)
        if pred_std > 1e-6 and targ_std > 1e-6:
            correlation = np.corrcoef(preds_flat, targets_flat)[0, 1]
        else:
            correlation = 0.0
    except:
        correlation = 0.0
    
    # Per-spatial-location statistics
    spatial_rmse = np.sqrt(np.mean((all_preds - all_targets)**2, axis=0))  # (C, H, W)
    
    # Results
    print(f"\n" + "="*60)
    print("📊 TEST RESULTS (ConvLSTM Model)")
    print("="*60)
    print(f"Total samples tested:    {len(X_tensor)}")
    print(f"Sequence length:         {config['sequence_length']}")
    print(f"Spatial resolution:      {X_raw.shape[3]} x {X_raw.shape[4]} (Height x Width)")
    print(f"\n🎯 Global Metrics:")
    print(f"  RMSE (Mean):           {mean_rmse:.6f}")
    print(f"  MAE (Mean Abs Error):  {mae:.6f}")
    print(f"  Std Dev of errors:     {std_error:.6f}")
    print(f"  Correlation:           {correlation:.4f}")
    print(f"\n📊 Spatial Statistics (per grid cell):")
    print(f"  RMSE min:              {np.min(spatial_rmse):.6f}")
    print(f"  RMSE mean:             {np.mean(spatial_rmse):.6f}")
    print(f"  RMSE max:              {np.max(spatial_rmse):.6f}")
    print(f"  RMSE median:           {np.median(spatial_rmse):.6f}")
    print(f"\n📈 Value ranges:")
    print(f"  Predictions:           [{np.min(all_preds):.6f}, {np.max(all_preds):.6f}]")
    print(f"  Targets:               [{np.min(all_targets):.6f}, {np.max(all_targets):.6f}]")
    print(f"  Mean pred:             {np.mean(all_preds):.6f}")
    print(f"  Mean target:           {np.mean(all_targets):.6f}")
    print("="*60 + "\n")
    
    return {
        "rmse": mean_rmse,
        "mae": mae,
        "correlation": correlation,
        "num_samples": len(X_tensor),
        "spatial_rmse_mean": np.mean(spatial_rmse)
    }


if __name__ == "__main__":
    results = test_convlstm_on_preprocessed_data()
