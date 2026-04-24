import numpy as np
import torch
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from models.convlstm import ConvLSTMModel
from training.metrics import rmse
import yaml


class ClimateModelEvaluator:
    def __init__(self, model_path="experiments/latest/model.pth"):
        """Initialize evaluator with trained model"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)
        
        # Load normalization stats
        self.load_normalization_stats()
        
        # Load model
        self.model = self.load_model()
        
    def load_normalization_stats(self):
        """Load mean and std for denormalization"""
        try:
            with open("data/processed/mean_std.json") as f:
                stats = json.load(f)
            self.mean = float(stats.get("mean", 288.15))
            self.std = float(stats.get("std", 10.0))
            
            # Validate stats
            if np.isnan(self.mean) or np.isnan(self.std) or self.std == 0:
                self.mean = 288.15
                self.std = 10.0
                print(f"⚠️  Invalid stats in file, using defaults")
            
            print(f"✓ Normalization stats (mean={self.mean:.2f}, std={self.std:.2f})")
        except Exception as e:
            print(f"⚠️  Could not load normalization stats: {e}")
            self.mean = 288.15  # ~15°C in Kelvin
            self.std = 10.0
    
    def load_model(self):
        """Load trained model weights"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        model = ConvLSTMModel(self.config)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"✓ Model loaded from {self.model_path}")
        return model
    
    def denormalize(self, normalized_data):
        """Convert from normalized space back to original values"""
        return normalized_data * self.std + self.mean
    
    def denormalize_celsius(self, normalized_data):
        """Convert to Celsius"""
        kelvin = self.denormalize(normalized_data)
        celsius = kelvin - 273.15
        return celsius
    
    def evaluate(self, split="test"):
        """Evaluate model on test/val/train split"""
        print(f"\n{'='*70}")
        print(f"🧪 EVALUATING MODEL ON {split.upper()} SET")
        print(f"{'='*70}")
        
        # Load data
        X_path = f"data/processed/tensors/{split}_X.npy"
        y_path = f"data/processed/tensors/{split}_y.npy"
        
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Data files not found for {split}")
        
        X_norm = np.load(X_path)  # (B, T, C, H, W)
        y_norm = np.load(y_path)  # (B, C, H, W)
        
        print(f"Loaded {len(X_norm)} samples")
        
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32)
        
        # Run inference
        batch_size = 8
        all_preds_norm = []
        all_targets_norm = []
        
        print(f"\n🔮 Running inference...")
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                X_batch = X_tensor[i:i+batch_size].to(self.device)
                y_batch = y_tensor[i:i+batch_size].to(self.device)
                
                preds = self.model(X_batch)
                
                all_preds_norm.append(preds.cpu().numpy())
                all_targets_norm.append(y_batch.cpu().numpy())
                
                if (i // batch_size) % max(1, (len(X_tensor) // batch_size) // 5) == 0:
                    print(f"  Progress: {i//batch_size}/{len(X_tensor)//batch_size} batches")
        
        # Concatenate results
        preds_norm = np.concatenate(all_preds_norm, axis=0)  # (B, C, H, W)
        targets_norm = np.concatenate(all_targets_norm, axis=0)
        
        # Convert to Keras for proper shape
        B, C, H, W = preds_norm.shape
        
        # Denormalize
        preds_K = self.denormalize(preds_norm)
        targets_K = self.denormalize(targets_norm)
        
        preds_C = self.denormalize_celsius(preds_norm)
        targets_C = self.denormalize_celsius(targets_norm)
        
        # Compute metrics
        print(f"\n📊 Computing metrics...")
        
        # Flattened metrics
        preds_flat = preds_norm.flatten()
        targets_flat = targets_norm.flatten()
        
        rmse_norm = np.sqrt(np.mean((preds_flat - targets_flat)**2))
        mae_norm = np.mean(np.abs(preds_flat - targets_flat))
        
        preds_K_flat = preds_K.flatten()
        targets_K_flat = targets_K.flatten()
        
        rmse_K = np.sqrt(np.mean((preds_K_flat - targets_K_flat)**2))
        mae_K = np.mean(np.abs(preds_K_flat - targets_K_flat))
        
        preds_C_flat = preds_C.flatten()
        targets_C_flat = targets_C.flatten()
        
        rmse_C = np.sqrt(np.mean((preds_C_flat - targets_C_flat)**2))
        mae_C = np.mean(np.abs(preds_C_flat - targets_C_flat))
        
        # Correlation
        try:
            if np.std(preds_flat) > 1e-6 and np.std(targets_flat) > 1e-6:
                corr = np.corrcoef(preds_flat, targets_flat)[0, 1]
            else:
                corr = 0.0
        except:
            corr = 0.0
        
        # Spatial metrics
        spatial_rmse = np.sqrt(np.mean((preds_K - targets_K)**2, axis=0))  # (C, H, W)
        spatial_mae = np.mean(np.abs(preds_K - targets_K), axis=0)
        
        # Temporal metrics
        temporal_rmse = np.sqrt(np.mean((preds_K - targets_K)**2, axis=(2, 3)))  # (B, C)
        
        # Results dictionary
        results = {
            "split": split,
            "num_samples": len(X_norm),
            "spatial_shape": (H, W),
            "metrics": {
                "rmse_normalized": float(rmse_norm),
                "mae_normalized": float(mae_norm),
                "rmse_kelvin": float(rmse_K),
                "mae_kelvin": float(mae_K),
                "rmse_celsius": float(rmse_C),
                "mae_celsius": float(mae_C),
                "correlation": float(corr),
            },
            "spatial_metrics": {
                "spatial_rmse_min": float(np.min(spatial_rmse)),
                "spatial_rmse_mean": float(np.mean(spatial_rmse)),
                "spatial_rmse_max": float(np.max(spatial_rmse)),
                "spatial_rmse_std": float(np.std(spatial_rmse)),
            },
            "value_ranges": {
                "pred_min": float(np.min(targets_C_flat)),
                "pred_max": float(np.max(preds_C_flat)),
                "target_min": float(np.min(targets_C_flat)),
                "target_max": float(np.max(targets_C_flat)),
                "pred_mean": float(np.mean(preds_C_flat)),
                "target_mean": float(np.mean(targets_C_flat)),
            }
        }
        
        # Print results
        self.print_results(results)
        
        # Save results
        self.save_results(results)
        
        # Create visualizations
        self.create_visualizations(
            preds_norm, targets_norm, preds_C, targets_C,
            spatial_rmse, split
        )
        
        return results, preds_C, targets_C
    
    def print_results(self, results):
        """Print formatted results"""
        print(f"\n{'='*70}")
        print(f"📈 EVALUATION RESULTS - {results['split'].upper()}")
        print(f"{'='*70}")
        
        print(f"\n📊 Dataset:")
        print(f"  Samples:              {results['num_samples']}")
        print(f"  Spatial Resolution:   {results['spatial_shape'][0]} × {results['spatial_shape'][1]}")
        
        print(f"\n🎯 Error Metrics (Normalized Space):")
        print(f"  RMSE:                 {results['metrics']['rmse_normalized']:.6f}")
        print(f"  MAE:                  {results['metrics']['mae_normalized']:.6f}")
        
        print(f"\n🌡️  Error Metrics (Kelvin):")
        print(f"  RMSE:                 {results['metrics']['rmse_kelvin']:.4f} K")
        print(f"  MAE:                  {results['metrics']['mae_kelvin']:.4f} K")
        
        print(f"\n°C Error Metrics (Celsius):")
        print(f"  RMSE:                 {results['metrics']['rmse_celsius']:.4f} °C")
        print(f"  MAE:                  {results['metrics']['mae_celsius']:.4f} °C")
        
        print(f"\n📊 Spatial Analysis:")
        print(f"  Min RMSE (best grid): {results['spatial_metrics']['spatial_rmse_min']:.4f} K")
        print(f"  Mean RMSE:            {results['spatial_metrics']['spatial_rmse_mean']:.4f} K")
        print(f"  Max RMSE (worst):     {results['spatial_metrics']['spatial_rmse_max']:.4f} K")
        
        print(f"\n🔗 Correlation:")
        print(f"  Predicted vs Actual:  {results['metrics']['correlation']:.4f}")
        
        print(f"\n🌡️  Temperature Ranges:")
        print(f"  Predictions:          {results['value_ranges']['pred_min']:.2f}°C to {results['value_ranges']['pred_max']:.2f}°C")
        print(f"  Actual:               {results['value_ranges']['target_min']:.2f}°C to {results['value_ranges']['target_max']:.2f}°C")
        
        print(f"\n{'='*70}\n")
    
    def save_results(self, results):
        """Save results to JSON"""
        os.makedirs("outputs/evaluation", exist_ok=True)
        
        output_path = f"outputs/evaluation/{results['split']}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
    
    def create_visualizations(self, preds_norm, targets_norm, preds_C, targets_C, spatial_rmse, split):
        """Create evaluation visualizations"""
        os.makedirs("outputs/visualizations", exist_ok=True)
        
        # 1. Error distribution
        errors_C = preds_C - targets_C
        errors_C_flat = errors_C.flatten()
        
        # Handle NaN values
        valid_errors = errors_C_flat[~np.isnan(errors_C_flat)]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Model Evaluation - {split.upper()} Set", fontsize=16, fontweight='bold')
        
        if len(valid_errors) > 0:
            # Error histogram
            axes[0, 0].hist(valid_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 0].set_xlabel('Prediction Error (°C)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Error Distribution')
            mean_err = np.mean(valid_errors)
            axes[0, 0].axvline(mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.3f}°C')
            axes[0, 0].legend()
        
        # Prediction vs Actual scatter (use normalized data which is valid)
        preds_flat = preds_norm.flatten()
        targets_flat = targets_norm.flatten()
        valid_idx = ~(np.isnan(preds_flat) | np.isnan(targets_flat))
        
        if np.sum(valid_idx) > 0:
            sample_idx = np.random.choice(np.where(valid_idx)[0], min(1000, np.sum(valid_idx)), replace=False)
            axes[0, 1].scatter(targets_flat[sample_idx], preds_flat[sample_idx], alpha=0.5, s=10)
            min_val = min(targets_flat[valid_idx].min(), preds_flat[valid_idx].min())
            max_val = max(targets_flat[valid_idx].max(), preds_flat[valid_idx].max())
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
            axes[0, 1].set_xlabel('Actual (Normalized)')
            axes[0, 1].set_ylabel('Predicted (Normalized)')
            axes[0, 1].set_title('Prediction Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Spatial RMSE heatmap
        spatial_rmse_flat = spatial_rmse[0] if spatial_rmse.ndim == 3 else spatial_rmse
        valid_spatial = spatial_rmse_flat[~np.isnan(spatial_rmse_flat)]
        
        if len(valid_spatial) > 0:
            im = axes[1, 0].imshow(np.nan_to_num(spatial_rmse_flat), cmap='RdYlGn_r', aspect='auto')
            axes[1, 0].set_title('Spatial RMSE Distribution')
            axes[1, 0].set_xlabel('Longitude')
            axes[1, 0].set_ylabel('Latitude')
            plt.colorbar(im, ax=axes[1, 0], label='RMSE')
        
        # Error by location
        if spatial_rmse_flat.ndim == 2:
            mean_spatial = np.nanmean(spatial_rmse_flat, axis=0)
        else:
            mean_spatial = spatial_rmse_flat
        
        valid_spatial_idx = ~np.isnan(mean_spatial)
        if np.sum(valid_spatial_idx) > 0:
            axes[1, 1].plot(mean_spatial[valid_spatial_idx], linewidth=2, color='steelblue')
            axes[1, 1].set_xlabel('Longitude Index')
            axes[1, 1].set_ylabel('Mean RMSE')
            axes[1, 1].set_title('RMSE by Longitude')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = f"outputs/visualizations/{split}_evaluation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {fig_path}")
        plt.close()
        
        # 2. Sample predictions map
        sample_idx = 0
        pred_map = preds_C[sample_idx, 0]
        target_map = targets_C[sample_idx, 0]
        error_map = pred_map - target_map
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f"Sample Prediction - {split.upper()} (Sample {sample_idx})", fontsize=14, fontweight='bold')
        
        # Handle NaN values in maps
        vmin, vmax = np.nanpercentile(target_map, [2, 98])
        
        im1 = axes[0].imshow(np.nan_to_num(target_map), cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Actual Temperature')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0], label='Value')
        
        im2 = axes[1].imshow(np.nan_to_num(pred_map), cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted Temperature')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[1], label='Value')
        
        error_max = np.nanmax(np.abs(error_map))
        im3 = axes[2].imshow(np.nan_to_num(error_map), cmap='RdBu_r', aspect='auto', vmin=-error_max, vmax=error_max)
        axes[2].set_title('Prediction Error')
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        plt.colorbar(im3, ax=axes[2], label='Error')
        
        plt.tight_layout()
        map_path = f"outputs/visualizations/{split}_prediction_map.png"
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        print(f"✓ Map visualization saved to {map_path}")
        plt.close()


if __name__ == "__main__":
    evaluator = ClimateModelEvaluator()
    
    # Evaluate on all splits
    print("\n" + "="*60)
    print("EVALUATING ALL SPLITS")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        print(f"\n🔍 Evaluating {split.upper()} split...")
        results, preds, targets = evaluator.evaluate(split=split)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE - All splits evaluated!")
    print("="*60)
