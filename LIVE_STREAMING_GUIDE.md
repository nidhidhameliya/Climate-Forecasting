# 🌊 Live Streaming Climate Data Pipeline - Complete Guide

## 📋 Overview

Converting your trained model from **static batch processing** to **real-time streaming predictions**.

---

## 1️⃣ OPEN-SOURCE DATA SOURCES (Live Weather Data)

### A. NOAA (National Oceanic & Atmospheric Administration) - FREE ✅
```
📍 Source: https://www.ncei.noaa.gov/
📊 Data: Real-time weather, satellite, radar
🔄 Update Frequency: 6-hourly to hourly
💾 Format: NetCDF, GRIB, CSV
✅ License: Public Domain
```

**Best for Climate Data:**
- GFS (Global Forecast System) - 0.25° resolution
- HRRR (High-Resolution Rapid Refresh) - Hourly updates
- NDFD (National Digital Forecast Database)

### B. OpenWeather API - FREE TIER ✅
```
📍 Source: https://openweathermap.org/api
📊 Data: Current & forecast weather
🔄 Update: Hourly
💾 Format: JSON
✅ License: Open Data Commons
🆓 Free Tier: 1,000 calls/day
```

**Pro**: JSON is easy to parse, global coverage

### C. GFS/GRIB Data - FREE ✅
```
📍 Source: https://nomads.ncei.noaa.gov/
📊 Data: GFS forecasts globally
🔄 Update: 4x daily
💾 Format: GRIB2 (need conversion)
✅ License: Public Domain
```

### D. Copernicus Climate Data - FREE ✅
```
📍 Source: https://cds.climate.copernicus.eu/
📊 Data: Complete reanalysis datasets
🔄 Update: Daily/Monthly
💾 Format: NetCDF, GRIB
✅ License: Free for R&D
```

### E. Your Own Weather Stations
```
If you have IoT sensors:
- Arduino/Raspberry Pi data
- Weather station sensors
- Satellite ground stations
```

---

## 2️⃣ ARCHITECTURE FOR LIVE STREAMING

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│            (NOAA, OpenWeather, Sensors)                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              DATA INGESTION LAYER                       │
│   (Apache Kafka / Redis / RabbitMQ / Python Queue)     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         REAL-TIME PROCESSING LAYER                     │
│  (Apache Spark / Faust / Kafka Streams / Python)       │
│  - Validate data                                        │
│  - Fill missing values                                  │
│  - Normalize using stored statistics                    │
│  - Create sequences (rolling 7-day window)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              MODEL INFERENCE                            │
│    (PyTorch / TorchServe / Seldon / KServe)            │
│    - Load trained model                                 │
│    - Run prediction                                     │
│    - Return forecast                                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STORAGE & VISUALIZATION LAYER                  │
│  - TimescaleDB / InfluxDB (time-series)               │
│  - PostgreSQL (results)                                │
│  - Grafana / Dash (real-time dashboards)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              API & SERVING                             │
│     (FastAPI / Flask / Streamlit)                      │
│     - Query predictions                                │
│     - Web interface                                    │
│     - Mobile app backend                              │
└─────────────────────────────────────────────────────────┘
```

---

## 3️⃣ RECOMMENDED OPEN-SOURCE STACK

### Best Setup for Your Use Case: **Python Native** 🐍

```
┌──────────────────────────────────────────────────┐
│  Ingestion: APScheduler + Requests               │
│  Queue: Redis or RabbitMQ                        │
│  Processing: Celery (synchronous tasks)          │
│  Inference: PyTorch + FastAPI                    │
│  Storage: TimescaleDB (fork of PostgreSQL)       │
│  Monitoring: Prometheus + Grafana                │
│  Dashboard: Streamlit + Plotly                   │
└──────────────────────────────────────────────────┘
```

### Alternative: **Full Streaming** (if you need <100ms latency)

```
┌──────────────────────────────────────────────────┐
│  Ingestion: Kafka / Redis Streams                │
│  Processing: Apache Spark / Kafka Streams        │
│  Inference: TorchServe / Seldon Core            │
│  Storage: InfluxDB + Grafana                    │
│  Dashboard: Grafana / Kibana                    │
└──────────────────────────────────────────────────┘
```

---

## 4️⃣ IMPLEMENTATION: Python Native Approach (Recommended)

### Step 1: Install Required Packages

```bash
pip install \
  APScheduler \           # Schedule data fetching
  redis \                 # Message queue
  celery \                # Task management
  fastapi \               # Web API
  uvicorn \               # ASGI server
  sqlalchemy \            # ORM
  psycopg2-binary \       # PostgreSQL driver
  timescaledb \           # Time-series extension
  prometheus-client \     # Monitoring
  pyyaml \                # Config management
  requests \              # HTTP client
  numpy \                 # Data processing
  torch \                 # Model inference
  plotly \                # Real-time charts
  streamlit               # Dashboard
```

### Step 2: Data Ingestion Service

**File: `services/data_ingester.py`**

```python
import requests
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import redis

class WeatherDataIngester:
    """Fetch live weather data from NOAA/OpenWeatherMap"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.api_key = "YOUR_OPENWEATHER_KEY"
        self.scheduler = BackgroundScheduler()
    
    def fetch_openweather_data(self, lat: float, lon: float, country: str):
        """
        Fetch current weather from OpenWeather API (free tier)
        
        Args:
            lat: Latitude
            lon: Longitude
            country: Country code (e.g., 'IN' for India)
        
        Returns:
            dict: Temperature and metadata
        """
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Extract relevant fields
            weather_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'latitude': lat,
                'longitude': lon,
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'clouds': data['clouds']['all'],
                'description': data['weather'][0]['description']
            }
            
            return weather_data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def fetch_noaa_gfs_data(self, lat: float, lon: float):
        """
        Fetch GFS forecast from NOAA (free, no key needed)
        
        Use: https://nomads.ncei.noaa.gov/
        """
        # Simplified example - full implementation would parse GRIB2
        print(f"📥 Fetching GFS forecast for lat={lat}, lon={lon}")
        # Implementation depends on GRIB library
        pass
    
    def push_to_queue(self, data: dict, queue_name: str = "weather_data"):
        """Store data in Redis queue"""
        try:
            self.redis_client.rpush(
                queue_name,
                json.dumps(data)
            )
            print(f"✅ Data pushed to {queue_name}")
        except Exception as e:
            print(f"❌ Redis error: {e}")
    
    def start_scheduler(self, interval_seconds: int = 3600):
        """Start automatic data fetching"""
        self.scheduler.add_job(
            self.fetch_and_queue,
            'interval',
            seconds=interval_seconds,
            id='weather_fetch'
        )
        self.scheduler.start()
        print(f"✅ Scheduler started (interval: {interval_seconds}s)")
    
    def fetch_and_queue(self):
        """Fetch data from all stations and queue"""
        # Example grid points for India
        grid_points = [
            (28.6139, 77.2090, "Delhi"),
            (19.0760, 72.8777, "Mumbai"),
            (12.9716, 77.5946, "Bangalore"),
            (13.0827, 80.2707, "Chennai"),
            (31.5204, 74.3587, "Lahore"),
        ]
        
        for lat, lon, city in grid_points:
            data = self.fetch_openweather_data(lat, lon, city)
            if data:
                self.push_to_queue(data)


# Usage
if __name__ == "__main__":
    ingester = WeatherDataIngester()
    ingester.start_scheduler(interval_seconds=3600)  # Every hour
```

### Step 3: Real-Time Processing Service

**File: `services/data_processor.py`**

```python
import json
import numpy as np
import redis
from datetime import datetime, timedelta
from collections import deque
import pickle

class RealTimeProcessor:
    """Process streaming data into model-ready format"""
    
    def __init__(self, sequence_length: int = 7):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.sequence_length = sequence_length
        self.normalization_stats = self.load_normalization_stats()
        
        # Circular buffer for each grid cell to maintain rolling sequence
        self.sequence_buffer = {}
    
    def load_normalization_stats(self):
        """Load mean and std from training"""
        import json
        with open('data/processed/mean_std.json') as f:
            return json.load(f)
    
    def normalize_data(self, value: float) -> float:
        """Normalize using training statistics"""
        mean = self.normalization_stats.get('mean', 288.15)
        std = self.normalization_stats.get('std', 10.0)
        
        # Handle NaN
        if std == 0:
            std = 10.0
        
        return (value - mean) / (std + 1e-6)
    
    def denormalize_data(self, value: float) -> float:
        """Convert back to original scale"""
        mean = self.normalization_stats.get('mean', 288.15)
        std = self.normalization_stats.get('std', 10.0)
        
        if std == 0:
            std = 10.0
        
        return value * std + mean
    
    def process_weather_data(self, raw_data: dict) -> dict:
        """
        Process raw weather data into normalized format
        
        Input: {'temperature': 25.5, 'humidity': 60, ...}
        Output: {'temperature_normalized': 0.123, ...}
        """
        processed = {
            'timestamp': raw_data['timestamp'],
            'latitude': raw_data['latitude'],
            'longitude': raw_data['longitude'],
            'temperature_celsius': raw_data['temperature'],
            'temperature_kelvin': raw_data['temperature'] + 273.15,
            'temperature_normalized': self.normalize_data(
                raw_data['temperature'] + 273.15
            ),
            'humidity': raw_data['humidity'],
            'pressure': raw_data['pressure'],
            'wind_speed': raw_data['wind_speed']
        }
        
        return processed
    
    def create_sequences(self, grid_id: str, processed_data: dict) -> dict:
        """
        Create rolling 7-day sequences for model input
        
        Returns: (sequence_ready: bool, data: array if ready)
        """
        # Initialize buffer for this grid cell if needed
        if grid_id not in self.sequence_buffer:
            self.sequence_buffer[grid_id] = deque(maxlen=self.sequence_length)
        
        # Add normalized temperature to buffer
        self.sequence_buffer[grid_id].append(
            processed_data['temperature_normalized']
        )
        
        # Check if we have enough data
        if len(self.sequence_buffer[grid_id]) >= self.sequence_length:
            # Convert to numpy array in correct shape
            sequence = np.array(list(self.sequence_buffer[grid_id]))
            sequence = sequence.reshape(1, self.sequence_length, 1, 1, 1)  # (B, T, C, H, W)
            
            return {
                'ready': True,
                'data': sequence,
                'grid_id': grid_id,
                'timestamp': processed_data['timestamp']
            }
        else:
            return {
                'ready': False,
                'buffered': len(self.sequence_buffer[grid_id]),
                'needed': self.sequence_length
            }
    
    def process_queue(self):
        """Continuously process data from Redis queue"""
        while True:
            try:
                # Get data from queue (blocking, 1 second timeout)
                raw_data_json = self.redis_client.blpop(
                    'weather_data',
                    timeout=1
                )
                
                if raw_data_json:
                    raw_data = json.loads(raw_data_json[1])
                    
                    # Process the data
                    processed = self.process_weather_data(raw_data)
                    
                    # Create sequence
                    grid_id = f"{raw_data['latitude']}_{raw_data['longitude']}"
                    sequence_result = self.create_sequences(grid_id, processed)
                    
                    if sequence_result['ready']:
                        # Push to prediction queue
                        self.redis_client.rpush(
                            'prediction_queue',
                            json.dumps({
                                'grid_id': grid_id,
                                'data': sequence_result['data'].tolist(),
                                'timestamp': sequence_result['timestamp']
                            })
                        )
                        print(f"✅ Sequence ready for {grid_id}")
                    
            except Exception as e:
                print(f"❌ Processing error: {e}")


# Usage
if __name__ == "__main__":
    processor = RealTimeProcessor()
    processor.process_queue()
```

### Step 4: Model Inference Service

**File: `services/model_inference.py`**

```python
import torch
import json
import redis
import numpy as np
from datetime import datetime
from models.convlstm import ConvLSTMModel
import yaml

class ModelInferenceService:
    """Real-time model inference on streaming data"""
    
    def __init__(self, model_path: str = "experiments/latest/model.pth"):
        # Load config
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvLSTMModel(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Redis connection
        self.redis_client = redis.Redis(host='localhost', port=6379)
        
        # Load normalization stats
        import json
        with open('data/processed/mean_std.json') as f:
            stats = json.load(f)
        self.mean = stats.get('mean', 288.15)
        self.std = stats.get('std', 10.0)
        
        print(f"✅ Model loaded on {self.device}")
    
    def denormalize(self, norm_value: float) -> float:
        """Convert from normalized to Kelvin"""
        return norm_value * self.std + self.mean
    
    def denormalize_celsius(self, norm_value: float) -> float:
        """Convert to Celsius"""
        kelvin = self.denormalize(norm_value)
        return kelvin - 273.15
    
    def run_inference(self, input_data: np.ndarray) -> dict:
        """
        Run model inference
        
        Args:
            input_data: Shape (1, 7, 1, H, W)
        
        Returns:
            dict: Predictions and metadata
        """
        try:
            # Convert to tensor
            X = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            
            # Run inference
            with torch.no_grad():
                pred_normalized = self.model(X).cpu().numpy()
            
            # Denormalize
            pred_celsius = self.denormalize_celsius(pred_normalized[0, 0])
            
            return {
                'success': True,
                'prediction_normalized': float(pred_normalized[0, 0]),
                'prediction_celsius': float(pred_celsius),
                'prediction_kelvin': float(self.denormalize(pred_normalized[0, 0])),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'v1.0_convlstm'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def inference_loop(self):
        """Continuously run inference on queued predictions"""
        print("🚀 Starting inference loop...")
        
        while True:
            try:
                # Get data from prediction queue
                item = self.redis_client.blpop('prediction_queue', timeout=1)
                
                if item:
                    item_data = json.loads(item[1])
                    grid_id = item_data['grid_id']
                    input_array = np.array(item_data['data'])
                    
                    # Run inference
                    result = self.run_inference(input_array)
                    
                    if result['success']:
                        # Store result with grid_id
                        self.redis_client.setex(
                            f"prediction_{grid_id}",
                            3600,  # Expire in 1 hour
                            json.dumps(result)
                        )
                        
                        # Also push to results queue for persistence
                        self.redis_client.rpush(
                            'results_stream',
                            json.dumps({
                                'grid_id': grid_id,
                                **result
                            })
                        )
                        
                        print(f"✅ Inference complete ({grid_id}): {result['prediction_celsius']:.2f}°C")
                    else:
                        print(f"❌ Inference failed: {result['error']}")
                        
            except Exception as e:
                print(f"❌ Loop error: {e}")


# Usage
if __name__ == "__main__":
    service = ModelInferenceService()
    service.inference_loop()
```

### Step 5: Real-Time API Server

**File: `api_server.py`**

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import redis
import json
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Climate Forecast API", version="1.0")
redis_client = redis.Redis(host='localhost', port=6379)

class PredictionRequest(BaseModel):
    grid_id: str
    latitude: float
    longitude: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    redis_connected: bool

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and Redis status"""
    try:
        redis_client.ping()
        redis_ok = True
    except:
        redis_ok = False
    
    return HealthResponse(
        status="running" if redis_ok else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        redis_connected=redis_ok
    )

@app.get("/prediction/{grid_id}")
async def get_prediction(grid_id: str):
    """Get latest prediction for a grid cell"""
    result = redis_client.get(f"prediction_{grid_id}")
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for {grid_id}"
        )
    
    return json.loads(result)

@app.get("/predictions/all")
async def get_all_predictions():
    """Get all active predictions"""
    # Get all keys matching pattern
    keys = redis_client.keys("prediction_*")
    
    predictions = []
    for key in keys:
        data = redis_client.get(key)
        if data:
            predictions.append(json.loads(data))
    
    return {
        'count': len(predictions),
        'predictions': predictions,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.post("/trigger_prediction")
async def trigger_prediction(request: PredictionRequest):
    """Manually trigger prediction for a location"""
    
    # This would trigger the ingestion pipeline
    trigger_data = {
        'grid_id': request.grid_id,
        'latitude': request.latitude,
        'longitude': request.longitude,
        'timestamp': datetime.utcnow().isoformat(),
        'triggered': True
    }
    
    redis_client.rpush('weather_data', json.dumps(trigger_data))
    
    return {
        'status': 'triggered',
        'grid_id': request.grid_id,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.get("/metrics/predictions")
async def get_prediction_metrics():
    """Get prediction statistics"""
    keys = redis_client.keys("prediction_*")
    
    if not keys:
        return {
            'count': 0,
            'message': 'No predictions yet'
        }
    
    predictions = []
    for key in keys:
        data = redis_client.get(key)
        if data:
            predictions.append(json.loads(data))
    
    # Calculate statistics
    if predictions:
        temps = [p['prediction_celsius'] for p in predictions]
        return {
            'total_predictions': len(predictions),
            'min_temp': min(temps),
            'max_temp': max(temps),
            'mean_temp': sum(temps) / len(temps),
            'latest_update': max([p['timestamp'] for p in predictions])
        }

# Run with: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### Step 6: Real-Time Dashboard

**File: `streaming_dashboard.py`**

```python
import streamlit as st
import redis
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="🌍 Live Climate Predictions", layout="wide")

redis_client = redis.Redis(host='localhost', port=6379)

st.title("🌊 Real-Time Climate Forecasting Dashboard")
st.markdown("Live predictions from ConvLSTM model on streaming data")

# Sidebar
st.sidebar.title("⚙️ Settings")
refresh_interval = st.sidebar.slider("Refresh Interval (s)", 5, 60, 10)
show_history = st.sidebar.checkbox("Show History", value=True)

# Auto-refresh
if st.sidebar.button("Manual Refresh"):
    st.rerun()

placeholder = st.empty()

while True:
    with placeholder.container():
        # Get all predictions
        keys = redis_client.keys("prediction_*")
        
        if not keys:
            st.warning("⏳ Waiting for predictions... (Model needs 7 days of data)")
        else:
            # Fetch data
            predictions = []
            for key in keys:
                data = redis_client.get(key)
                if data:
                    pred = json.loads(data)
                    predictions.append(pred)
            
            # Split into columns
            col1, col2, col3 = st.columns(3)
            
            # Metrics
            temps = [p['prediction_celsius'] for p in predictions]
            with col1:
                st.metric("🌡️ Current Avg Temp", f"{sum(temps)/len(temps):.2f}°C")
            with col2:
                st.metric("🔥 Max Temp", f"{max(temps):.2f}°C")
            with col3:
                st.metric("❄️ Min Temp", f"{min(temps):.2f}°C")
            
            st.divider()
            
            # Map view
            st.subheader("🗺️ Spatial Distribution")
            
            # Create map data
            map_data = pd.DataFrame([
                {
                    'grid_id': p.get('grid_id', 'Unknown'),
                    'temperature': p['prediction_celsius'],
                    'timestamp': p['timestamp']
                }
                for p in predictions
            ])
            
            st.dataframe(map_data, use_container_width=True)
            
            # Time series (if history enabled)
            if show_history:
                st.subheader("📈 Prediction History")
                
                # Get history from Redis
                history_keys = redis_client.lrange('results_stream', -100, -1)
                
                if history_keys:
                    history = [json.loads(item) for item in history_keys]
                    
                    hist_df = pd.DataFrame([
                        {
                            'time': h['timestamp'],
                            'temperature': h['prediction_celsius'],
                            'location': h.get('grid_id', 'Unknown')
                        }
                        for h in history
                    ])
                    
                    # Plot
                    fig = go.Figure()
                    
                    for location in hist_df['location'].unique():
                        loc_data = hist_df[hist_df['location'] == location]
                        fig.add_trace(go.Scatter(
                            x=loc_data['time'],
                            y=loc_data['temperature'],
                            mode='lines+markers',
                            name=location
                        ))
                    
                    fig.update_layout(
                        title="Temperature Predictions Over Time",
                        xaxis_title="Time",
                        yaxis_title="Temperature (°C)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Status
        st.sidebar.divider()
        st.sidebar.subheader("📊 Service Status")
        
        try:
            redis_client.ping()
            st.sidebar.success("✅ Redis Connected")
        except:
            st.sidebar.error("❌ Redis Disconnected")
        
        # Next refresh
        st.sidebar.info(f"Next refresh: {refresh_interval}s")
    
    time.sleep(refresh_interval)
```

---

## 5️⃣ ORCHESTRATION: Docker Compose Setup

**File: `docker-compose.yml`**

```yaml
version: '3.8'

services:
  
  # Redis (message queue & cache)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  # PostgreSQL + TimescaleDB (time-series storage)
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: climate_db
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  # Data Ingestion Service
  ingester:
    build:
      context: .
      dockerfile: Dockerfile.ingester
    depends_on:
      redis:
        condition: service_healthy
    environment:
      REDIS_URL: redis://redis:6379
      OPENWEATHER_API_KEY: ${OPENWEATHER_KEY}
    volumes:
      - ./services:/app/services
    command: python services/data_ingester.py
    restart: unless-stopped
  
  # Data Processing Service
  processor:
    build:
      context: .
      dockerfile: Dockerfile.processor
    depends_on:
      redis:
        condition: service_healthy
    environment:
      REDIS_URL: redis://redis:6379
    volumes:
      - ./services:/app/services
      - ./data:/app/data
    command: python services/data_processor.py
    restart: unless-stopped
  
  # Model Inference Service
  inference:
    build:
      context: .
      dockerfile: Dockerfile.inference
    depends_on:
      redis:
        condition: service_healthy
    environment:
      REDIS_URL: redis://redis:6379
      TORCH_HOME: /tmp/torch
    volumes:
      - ./services:/app/services
      - ./models:/app/models
      - ./experiments:/app/experiments
    command: python services/model_inference.py
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # FastAPI Server
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    depends_on:
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    environment:
      REDIS_URL: redis://redis:6379
    volumes:
      - ./api_server.py:/app/api_server.py
    command: uvicorn api_server:app --host 0.0.0.0 --port 8000
    restart: unless-stopped
  
  # Prometheus (metrics)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
  
  # Grafana (visualization)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - timescaledb
      - prometheus
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  redis_data:
  pg_data:
  prometheus_data:
  grafana_data:
```

---

## 6️⃣ QUICK START

### Start Everything

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export OPENWEATHER_KEY="your_key_here"

# 3. Start all services
docker-compose up -d

# 4. View logs
docker-compose logs -f inference

# 5. Check API
curl http://localhost:8000/health

# 6. View dashboard
streamlit run streaming_dashboard.py

# 7. Visit Grafana
# http://localhost:3000 (admin/admin)
```

### Monitor

```bash
# View all services
docker-compose ps

# Check Redis
redis-cli -p 6379 keys "*"

# View queue length
redis-cli -p 6379 llen weather_data
redis-cli -p 6379 llen prediction_queue
redis-cli -p 6379 llen results_stream
```

---

## 7️⃣ DATA SOURCES SETUP

### Get Free OpenWeather API Key

```bash
# Visit: https://openweathermap.org/api
# Sign up → Get API Key
# Free tier: 1,000 calls/day

export OPENWEATHER_KEY="abc123xyz..."
```

### Use NOAA Data (No Key Needed!)

```python
# Download GFS forecast
# From: https://nomads.ncei.noaa.gov/

# Or use NOAA's OpenDAP server:
from netCDF4 import Dataset

url = "https://gfs-0p25-analysis-and-fc.s3.amazonaws.com/gfs0p25_2024031200_000p00_extracted.nc"
ds = Dataset(url)
temp = ds['tmp2m'][:]
```

### Use Your Own Sensors

```python
# Arduino/Raspberry Pi → MQTT
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    # Push to Redis
    redis_client.rpush('weather_data', json.dumps(data))

client = mqtt.Client()
client.on_message = on_message
client.connect("broker.mqtt.org", 1883, 60)
client.subscribe("weather/sensors/#")
client.loop_forever()
```

---

## ⚡ PRODUCTION CHECKLIST

- [ ] Use external secret manager (HashiCorp Vault)
- [ ] Implement model versioning
- [ ] Add circuit breaker for API calls
- [ ] Set up monitoring & alerting (Prometheus + Alert Manager)
- [ ] Implement data validation & schema checking
- [ ] Add rate limiting to API
- [ ] Use container orchestration (Kubernetes)
- [ ] Implement graceful shutdown
- [ ] Set up backup & disaster recovery
- [ ] Add audit logging
- [ ] Implement canary deployments
- [ ] Set up data quality checks

---

## 📚 OPEN SOURCE ALTERNATIVES

| Component | Tool | License | Notes |
|-----------|------|---------|-------|
| Message Queue | Apache Kafka | Apache 2.0 | High throughput |
| " | RabbitMQ | MPL 2.0 | Simple setup |
| " | Redis Streams | BSD | Built-in to Redis |
| Stream Processing | Apache Spark | Apache 2.0 | Big data |
| " | Kafka Streams | Apache 2.0 | Kafka-native |
| Model Serving | TorchServe | Apache 2.0 | PyTorch native |
| " | Seldon Core | Apache 2.0 | Kubernetes |
| Time-Series DB | TimescaleDB | Apache 2.0 | PostgreSQL-based |
| " | InfluxDB | AGPL | Pure time-series |
| Monitoring | Prometheus | Apache 2.0 | Metrics |
| Visualization | Grafana | AGPL | Beautiful dashboards |
| API | FastAPI | MIT | Modern, fast |
| Dashboard | Streamlit | Apache 2.0 | Data apps |

---

**All open source, all free to self-host!** ✨
