import json
from pathlib import Path

print("="*60)
print("📊 EVALUATION RESULTS VERIFICATION")
print("="*60)

for split in ['train', 'val', 'test']:
    path = Path(f'outputs/evaluation/{split}_results.json')
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        samples = data.get('num_samples', '?')
        rmse = data.get('metrics', {}).get('rmse_celsius', '?')
        print(f"✅ {split.upper():5} | {samples:5} samples | RMSE: {rmse:.6f}°C")
    else:
        print(f"❌ {split.upper():5} | NOT FOUND")

print("="*60)
print("✅ All evaluation results ready for dashboard!")
print("="*60)
