import numpy as np
import torch
import os
from models.cnn_lstm import CNNLSTMModel
from training.metrics import rmse
import yaml


def test_lstm_on_preprocessed_data():
    """
    Direct test of LSTM model on preprocessed train/val/test split data
    """
    
    print("\n" + "="*60)
    print("🧪 DIRECT LSTM TEST ON PREPROCESSED DATA")
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
    X_raw = np.load(test_X_path)  # (B, T, H, W, 1)
    y_raw = np.load(test_y_path)  # (B, H, W, 1)
    
    print(f"✓ Raw X shape: {X_raw.shape}")
    print(f"✓ Raw y shape: {y_raw.shape}")
    
    # Data loader comment says (B, T, H, W, C) but checking data to see actual format
    # Let's try both interpretations
    # Check if height/width match our region (121, 141)
    if X_raw.shape[2] == 121 and X_raw.shape[3] == 141:
        # Shape is (B, T, H, W, C) - need to reshape to (B, T, C, H, W)
        X = np.transpose(X_raw, (0, 1, 4, 2, 3))
        y = np.transpose(y_raw, (0, 3, 1, 2))
        print(f"✓ Transposed X shape: {X.shape}")
        print(f"✓ Transposed y shape: {y.shape}")
    else:
        # Shape is already (B, T, C, H, W)
        X = X_raw
        y = y_raw
        print(f"✓ Data already in correct format")
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Create model
    print(f"\n🧠 Creating CNN-LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMModel(config)
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
                preds = model(X_batch)
                
                # Reshape for comparison
                preds_flat = preds.view(preds.shape[0], -1)
                y_flat = y_batch.view(y_batch.shape[0], -1)
                
                all_preds.append(preds_flat.cpu().numpy())
                all_targets.append(y_flat.cpu().numpy())
                
                batch_rmse = rmse(preds_flat, y_flat).item()
                total_rmse += batch_rmse
                num_batches += 1
                
                if (i // batch_size) % max(1, (len(X_tensor) // batch_size) // 10) == 0:
                    print(f"  Batch {i//batch_size}/{len(X_tensor)//batch_size}: RMSE = {batch_rmse:.6f}")
                    
            except Exception as e:
                print(f"❌ Error on batch {i//batch_size}: {e}")
                continue
    
    # Aggregate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mean_rmse = total_rmse / num_batches
    
    # Additional metrics
    mae = np.mean(np.abs(all_preds - all_targets))
    std_error = np.std(all_preds - all_targets)
    
    # Correlation (safe)
    try:
        correlation = np.corrcoef(all_preds.flatten(), all_targets.flatten())[0, 1]
    except:
        correlation = np.nan
    
    # Results
    print(f"\n" + "="*60)
    print("📊 TEST RESULTS (CNN-LSTM Model)")
    print("="*60)
    print(f"Total samples tested:    {len(X_tensor)}")
    print(f"Sequence length:         {config['sequence_length']}")
    print(f"Data shape per sample:   {X.shape[1:]}")
    print(f"\nMetrics:")
    print(f"  RMSE (Mean):           {mean_rmse:.6f}")
    print(f"  MAE (Mean Abs Error):  {mae:.6f}")
    print(f"  Std Dev of errors:     {std_error:.6f}")
    print(f"  Correlation:           {correlation:.4f}")
    print(f"\nValue ranges:")
    print(f"  Predictions:           [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    print(f"  Targets:               [{all_targets.min():.4f}, {all_targets.max():.4f}]")
    print(f"  Mean pred:             {np.mean(all_preds):.4f}")
    print(f"  Mean target:           {np.mean(all_targets):.4f}")
    print("="*60 + "\n")
    
    return {
        "rmse": mean_rmse,
        "mae": mae,
        "correlation": correlation,
        "num_samples": len(X_tensor)
    }


if __name__ == "__main__":
    results = test_lstm_on_preprocessed_data()
