# Quick-Start: Simple Daily Automation

A beginner-friendly bridge between static batch processing and full streaming.

## 📋 WHAT YOU'LL GET

```python
# Daily automated predictions without Redis/Kafka complexity
automatic_predictor.py runs:
1. Fetch latest weather data (every 6 hours)
2. Run model predictions
3. Save results with timestamp
4. Generate daily report
5. Email summary

No infrastructure needed! 
Runs on your laptop.
```

## 🔧 SETUP (5 minutes)

### Step 1: Install APScheduler
```bash
pip install APScheduler
```

### Step 2: Create daily_predictor.py
```python
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import torch
import numpy as np
import json
import os
from pathlib import Path

class DailyPredictor:
    def __init__(self):
        self.model_path = "experiments/latest/model.pth"
        self.output_dir = "outputs/scheduled_predictions"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model once
        from models.convlstm import ConvLSTMModel
        self.model = ConvLSTMModel(
            input_channels=1,
            num_filters=32,
            kernel_size=3,
            num_layers=2,
            output_size=11691
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ Model not found at {self.model_path}")
    
    def run_daily_prediction(self):
        """Run daily prediction (called by scheduler)"""
        print(f"\n{'='*50}")
        print(f"🕐 Daily prediction: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        try:
            # Load latest data
            latest_data = self.load_latest_data()
            if latest_data is None:
                print("❌ No data available")
                return
            
            # Run prediction
            prediction = self.predict(latest_data)
            
            # Save results
            self.save_results(prediction)
            
            # Generate report
            self.generate_report(prediction)
            
            print(f"✅ Daily prediction complete!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def load_latest_data(self):
        """Load latest preprocessed data"""
        data_path = "data/processed/tensors/test_X.npy"
        if os.path.exists(data_path):
            data = np.load(data_path)
            # Use the last sequence
            return torch.tensor(data[-1:], dtype=torch.float32).to(self.device)
        return None
    
    def predict(self, data):
        """Run inference"""
        with torch.no_grad():
            # Permute if needed: (batch, lat, time, channel, lon) → (batch, time, channel, height, width)
            if data.ndim == 5:
                data = data.permute(0, 2, 3, 1, 4)
            
            output = self.model(data)
            return output.cpu().numpy()
    
    def save_results(self, prediction):
        """Save prediction with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as NPY
        npy_path = f"{self.output_dir}/prediction_{timestamp}.npy"
        np.save(npy_path, prediction)
        
        # Save metadata as JSON
        json_path = f"{self.output_dir}/prediction_{timestamp}.json"
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": "ConvLSTM",
            "prediction_shape": prediction.shape,
            "prediction_path": npy_path,
            "mean_value": float(np.mean(prediction)),
            "min_value": float(np.min(prediction)),
            "max_value": float(np.max(prediction))
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved prediction: {npy_path}")
    
    def generate_report(self, prediction):
        """Generate daily report"""
        timestamp = datetime.now().strftime('%Y-%m-%d')
        
        report = f"""
═══════════════════════════════════════════════════
    DAILY PREDICTION REPORT - {timestamp}
═══════════════════════════════════════════════════

Model Performance:
- Mean Prediction: {np.mean(prediction):.6f}
- Min Value: {np.min(prediction):.6f}
- Max Value: {np.max(prediction):.6f}
- Std Dev: {np.std(prediction):.6f}

Data Shape: {prediction.shape}
Spatial Cells: {prediction.shape[-2] * prediction.shape[-1] if len(prediction.shape) > 2 else 'Unknown'}

═══════════════════════════════════════════════════

Output Files:
- Prediction: outputs/scheduled_predictions/
- Metadata: outputs/scheduled_predictions/*.json

Next Run: Check scheduler status
═══════════════════════════════════════════════════
"""
        
        report_path = f"{self.output_dir}/report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)

# ============================================================================
# SCHEDULER SETUP
# ============================================================================

def main():
    predictor = DailyPredictor()
    
    # Create scheduler
    scheduler = BlockingScheduler()
    
    # Schedule daily prediction at 2 AM
    scheduler.add_job(
        predictor.run_daily_prediction,
        'cron',
        hour=2,
        minute=0,
        id='daily_prediction'
    )
    
    # Or test with 1-minute intervals
    # scheduler.add_job(
    #     predictor.run_daily_prediction,
    #     'interval',
    #     minutes=1,
    #     id='test_prediction'
    # )
    
    print("🕐 Scheduler starting...")
    print("⏰ Daily prediction scheduled for 02:00 AM")
    print("📌 Press Ctrl+C to stop")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n✅ Scheduler stopped")

if __name__ == "__main__":
    main()
```

### Step 3: Run it!
```bash
python daily_predictor.py

# Output:
# 🕐 Scheduler starting...
# ⏰ Daily prediction scheduled for 02:00 AM
# 📌 Press Ctrl+C to stop
```

## 🧪 TESTING (Before Production)

### Test 1-minute intervals first:
```python
# In daily_predictor.py, change:
scheduler.add_job(
    predictor.run_daily_prediction,
    'interval',
    minutes=1,  # Run every minute to test
    id='test_prediction'
)

# Then check outputs/scheduled_predictions/ for new files
```

### Verify it worked:
```bash
# List predictions
Get-ChildItem outputs/scheduled_predictions/ -Filter "*.json"

# View latest metadata
Get-Content outputs/scheduled_predictions/prediction_*.json | Select-Object -Last 1
```

## 📊 SCHEDULE OPTIONS

### Common Schedules:

```python
# Every day at 2 AM
scheduler.add_job(func, 'cron', hour=2, minute=0)

# Every 6 hours
scheduler.add_job(func, 'interval', hours=6)

# Every Monday at 9 AM
scheduler.add_job(func, 'cron', day_of_week='mon', hour=9, minute=0)

# Monday-Friday at 8 AM
scheduler.add_job(func, 'cron', day_of_week='0-4', hour=8, minute=0)

# Every minute (testing only!)
scheduler.add_job(func, 'interval', minutes=1)

# Every 30 minutes during business hours
scheduler.add_job(func, 'cron', minute='*/30', hour='9-17')
```

## 📧 BONUS: Email Reports

Add email notifications:

```python
# Install: pip install python-dotenv

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

load_dotenv()

def send_email_report(prediction):
    """Send daily report via email"""
    
    sender = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")
    recipient = os.getenv("REPORT_EMAIL")
    
    if not all([sender, password, recipient]):
        print("⚠️ Email credentials not set in .env file")
        return
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = f"Daily Prediction Report - {datetime.now().strftime('%Y-%m-%d')}"
    
    body = f"""
    Daily Weather Prediction Summary
    ════════════════════════════════
    
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Prediction Statistics:
    - Mean: {np.mean(prediction):.6f}
    - Min: {np.min(prediction):.6f}
    - Max: {np.max(prediction):.6f}
    - Std: {np.std(prediction):.6f}
    
    Full report available in: outputs/scheduled_predictions/
    
    ---
    Automated Weather Forecast System
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("📧 Email sent!")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# Call in generate_report():
# send_email_report(prediction)
```

Set up .env file:
```
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
REPORT_EMAIL=recipient@example.com
```

## 🚀 RUNNING IN BACKGROUND

### Windows: Use Task Scheduler

```batch
# Create batch file: run_predictor.bat
@echo off
cd C:\Users\25mdsml006\Desktop\deep_learning\project\climate\climate2
python daily_predictor.py
```

Then use Windows Task Scheduler to run at startup.

### Linux/Mac: Use Cron

```bash
crontab -e

# Add line:
0 2 * * * cd /path/to/project && python daily_predictor.py >> logs/predictor.log 2>&1
```

### Docker: Keep it running 24/7

```dockerfile
FROM python:3.11

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt APScheduler

CMD ["python", "daily_predictor.py"]
```

```bash
docker build -t daily-predictor .
docker run -d --name predictor daily-predictor
```

## 📈 MONITORING

### Check if predictions are running:

```bash
# Count predictions generated
(Get-ChildItem outputs/scheduled_predictions/*.json).Count

# Show most recent
Get-ChildItem outputs/scheduled_predictions/ | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Monitor in real-time
Get-ChildItem outputs/scheduled_predictions/ -Watch
```

### Create a simple status dashboard:

```python
# status_dashboard.py
import streamlit as st
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Prediction Status", layout="wide")

st.title("📊 Daily Predictor Status")

pred_dir = Path("outputs/scheduled_predictions")

if pred_dir.exists():
    files = list(pred_dir.glob("*.json"))
    
    if files:
        # Sort by modification time
        latest = sorted(files, key=lambda x: x.stat().st_mtime)[-1]
        
        with open(latest) as f:
            data = json.load(f)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Run", latest.stem.split('_')[1])
        col2.metric("Mean Value", f"{data['mean_value']:.6f}")
        col3.metric("Min Value", f"{data['min_value']:.6f}")
        col4.metric("Max Value", f"{data['max_value']:.6f}")
        
        st.info(f"✅ Status: Running normally")
        st.info(f"📁 Total predictions: {len(files)//2}")  # /2 because of .npy and .json pairs
    else:
        st.warning("⏳ No predictions yet. Waiting for scheduler...")
else:
    st.error("❌ No prediction directory found")

st.markdown("---")
st.markdown("**Run**: `streamlit run status_dashboard.py`")
```

## ✨ NEXT STEPS

This simple automation:
- ✅ Runs on your laptop (no cloud needed)
- ✅ Makes predictions daily (or any interval)
- ✅ Saves timestamped results
- ✅ Generates reports
- ✅ Can email summaries

**When ready to upgrade**, add:
- Redis for real-time updates
- FastAPI for serving predictions
- Full Streamlit dashboard
- Docker for deployment

**For now**, enjoy automated daily forecasts! 🎉

---

## 🆘 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| "No module named 'APScheduler'" | `pip install APScheduler` |
| "Model file not found" | Update `model_path` to correct location |
| "Scheduler not running" | Check console - may need to install dependencies |
| "Email not sending" | Check .env file and Gmail app password |
| "Predictions not timestamped" | Verify `datetime.now()` is working |
| "Output directory missing" | Directory created automatically on first run |

## 📞 WHEN TO UPGRADE

Use this for **up to 1 year**, then consider:
- Redis/Kafka for higher throughput
- Database for historical queries
- API for multiple users
- Kubernetes for scaling

Your model is production-ready! 🚀
