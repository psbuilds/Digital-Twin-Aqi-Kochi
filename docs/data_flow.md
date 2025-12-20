# Digital Twin AQI Kochi - Data Flow Documentation

## Table of Contents
- [1. Overview](#1-overview)
- [2. End-to-End Data Flow](#2-end-to-end-data-flow)
- [3. Detailed Data Flow Diagrams](#3-detailed-data-flow-diagrams)
- [4. Data Formats & Transformations](#4-data-formats--transformations)
- [5. Data Flow Scenarios](#5-data-flow-scenarios)
- [6. Data Quality & Validation](#6-data-quality--validation)
- [7. Performance Considerations](#7-performance-considerations)

---

## 1. Overview

This document describes how data flows through the Digital Twin AQI Kochi system, from sensor data generation to visualization and prediction. The system processes data through multiple stages, transforming it from raw sensor readings to actionable insights.

### 1.1 Data Flow Characteristics

- **Direction**: Unidirectional with feedback loops
- **Volume**: ~10-100 sensor readings per minute
- **Latency**: Near real-time (<5 seconds end-to-end)
- **Format**: JSON (NGSI-v2 compliant)
- **Persistence**: FIWARE Orion + CSV archives

### 1.2 Key Data Pipelines

1. **Real-time Pipeline**: Sensor → FIWARE → Dashboard
2. **ML Training Pipeline**: Historical Data → Feature Engineering → Model Training
3. **Prediction Pipeline**: Current Data + Model → Future Predictions
4. **Analytics Pipeline**: Historical Data → Aggregation → Insights

---

## 2. End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: DATA GENERATION
┌──────────────────────────┐
│  Sensor Simulator        │
│  (aqi_sensor_simulator)  │
│                          │
│  Every 60 seconds:       │
│  1. Generate pollutant   │
│     concentrations       │
│  2. Apply diurnal        │
│     patterns             │
│  3. Add noise            │
│  4. Add metadata         │
└──────────────────────────┘
            │
            │ Raw Sensor Data (Python Dict)
            │ {
            │   'sensor_id': 'AQI_001',
            │   'location': 'Ernakulam',
            │   'PM2.5': 45.5,
            │   'PM10': 120.0,
            │   'NO2': 35.2,
            │   'SO2': 15.8,
            │   'CO': 1.2,
            │   'O3': 60.0,
            │   'temperature': 28.5,
            │   'humidity': 75.0,
            │   'timestamp': '2024-12-20T10:30:00Z'
            │ }
            ▼
┌──────────────────────────────────────────────────────────────────────────────┐

STAGE 2: DATA VALIDATION & TRANSFORMATION
┌──────────────────────────┐
│  Context Ingestor        │
│  (context_ingestor.py)   │
│                          │
│  1. validate_data()      │
│     - Check ranges       │
│     - Verify timestamp   │
│     - Validate location  │
│                          │
│  2. Transform to NGSI-v2 │
│     - Add entity type    │
│     - Structure attrs    │
│     - Add metadata       │
└──────────────────────────┘
            │
            │ NGSI-v2 Entity (JSON)
            │ {
            │   "id": "AQI_Ernakulam_001",
            │   "type": "AirQualitySensor",
            │   "location": {
            │     "type": "geo:json",
            │     "value": {
            │       "type": "Point",
            │       "coordinates": [76.2999, 9.9816]
            │     }
            │   },
            │   "PM2_5": {
            │     "type": "Number",
            │     "value": 45.5,
            │     "metadata": {
            │       "timestamp": {
            │         "type": "DateTime",
            │         "value": "2024-12-20T10:30:00Z"
            │       },
            │       "unit": {"type": "Text", "value": "µg/m³"}
            │     }
            │   },
            │   ... (other pollutants)
            │ }
            ▼
┌──────────────────────────────────────────────────────────────────────────────┐

STAGE 3: CONTEXT STORAGE
┌──────────────────────────┐
│  Orion Client            │
│  (orion_client.py)       │
│                          │
│  update_entity()         │
│  ↓                       │
│  HTTP POST/PATCH         │
│  to Orion API            │
└──────────────────────────┘
            │
            │ HTTP Request
            │ POST http://localhost:1026/v2/entities
            │ Headers:
            │   Content-Type: application/json
            │   Fiware-Service: aqi_kochi
            │   Fiware-ServicePath: /sensors
            ▼
┌──────────────────────────┐
│  FIWARE Orion            │
│  Context Broker          │
│                          │
│  - Store entity          │
│  - Update timestamp      │
│  - Trigger subscriptions │
│  - Maintain history      │
└──────────────────────────┘
            │
            │ (Data persisted in Orion)
            │
            ├─────────────────────┬──────────────────────┬────────────────────┐
            │                     │                      │                    │
            ▼                     ▼                      ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐

STAGE 4: DATA PROCESSING (Parallel Paths)

PATH A: REAL-TIME AQI CALCULATION
┌──────────────────────────┐
│  AQI Logic Module        │
│  (current_aqi_rules.py)  │
│                          │
│  1. Extract pollutant    │
│     concentrations       │
│  2. Calculate sub-index  │
│     for each pollutant   │
│  3. Determine max        │
│     sub-index as AQI     │
└──────────────────────────┘
            │
            │ AQI Value (Number)
            │ aqi = 125
            ▼
┌──────────────────────────┐
│  Status Mapping          │
│  (status_mapping.py)     │
│                          │
│  1. Map AQI to category  │
│  2. Get health advice    │
│  3. Assign color code    │
└──────────────────────────┘
            │
            │ Status Object
            │ {
            │   'aqi': 125,
            │   'category': 'Unhealthy for Sensitive Groups',
            │   'color': '#FF9900',
            │   'health_implications': '...',
            │   'cautionary_statement': '...'
            │ }
            ▼
┌──────────────────────────┐
│  Update FIWARE Entity    │
│  (add AQI & status)      │
└──────────────────────────┘

PATH B: DATA EXTRACTION FOR ML
┌──────────────────────────┐
│  Data Extraction         │
│  (data_extraction.py)    │
│                          │
│  1. Query FIWARE for     │
│     historical data      │
│  2. Convert to DataFrame │
│  3. Clean data           │
│  4. Feature engineering  │
└──────────────────────────┘
            │
            │ Pandas DataFrame
            │ Columns: timestamp, PM2.5, PM10, NO2, SO2, CO, O3,
            │          temperature, humidity, hour, day_of_week,
            │          PM2.5_rolling_mean_3h, PM2.5_lag_1h, ...
            ▼
┌──────────────────────────┐
│  Model Training          │
│  (train_model.py)        │
│                          │
│  1. Train-test split     │
│  2. Train ML model       │
│  3. Evaluate performance │
│  4. Save model           │
└──────────────────────────┘
            │
            │ Trained Model (PKL)
            │ + Evaluation Metrics (CSV)
            ▼
┌──────────────────────────┐
│  Model Storage           │
│  models/aqi_predictor.pkl│
│  data/evaluation_results │
└──────────────────────────┘

PATH C: PREDICTION GENERATION
┌──────────────────────────┐
│  Predict Future AQI      │
│  (predict_future_aqi.py) │
│                          │
│  1. Load trained model   │
│  2. Get current data     │
│  3. Generate features    │
│     for future times     │
│  4. Make predictions     │
└──────────────────────────┘
            │
            │ Prediction Array
            │ [
            │   {'timestamp': '2024-12-20T11:00:00Z', 'aqi': 128, 'confidence': 0.92},
            │   {'timestamp': '2024-12-20T12:00:00Z', 'aqi': 132, 'confidence': 0.89},
            │   ...
            │ ]
            ▼
┌──────────────────────────┐
│  Store Predictions       │
│  (in FIWARE or cache)    │
└──────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐

STAGE 5: DATA VISUALIZATION
┌──────────────────────────┐
│  Dashboard               │
│  (dashboard/app.py)      │
│                          │
│  1. Query FIWARE for     │
│     current data         │
│  2. Fetch predictions    │
│  3. Retrieve historical  │
│     data                 │
│  4. Render visualizations│
└──────────────────────────┘
            │
            │ HTTP Response (HTML/JSON)
            ▼
┌──────────────────────────┐
│  User's Web Browser      │
│                          │
│  - Real-time AQI gauge   │
│  - Trend charts          │
│  - Prediction graphs     │
│  - Interactive map       │
└──────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐

FEEDBACK LOOPS:
1. Dashboard → FIWARE (periodic queries every 30s)
2. ML Training → Model Updates (daily/weekly retraining)
3. Predictions → Dashboard (real-time updates)
```

---

## 3. Detailed Data Flow Diagrams

### 3.1 Sensor Data Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  SENSOR DATA INGESTION FLOW                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ Timer Event      │
│ (Every 60s)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ SensorSimulator.generate_reading()       │
│                                          │
│ FOR each pollutant:                      │
│   base_value = config.base_values[p]    │
│   pattern = simulate_pattern(p, hour)   │
│   noise = random.gauss(0, noise_level)  │
│   value = base_value * pattern + noise  │
│                                          │
│ RETURN {                                 │
│   'sensor_id': self.sensor_id,          │
│   'location': self.location,            │
│   'PM2.5': value_pm25,                  │
│   'PM10': value_pm10,                   │
│   ... (other pollutants),               │
│   'timestamp': datetime.utcnow()        │
│ }                                        │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ context_ingestor.ingest_sensor_data()    │
│                                          │
│ STEP 1: Validate                         │
│   IF not validate_data(sensor_data):    │
│     LOG error                            │
│     RETURN False                         │
│                                          │
│ STEP 2: Transform to NGSI-v2             │
│   entity = {                             │
│     'id': f"AQI_{location}_{id}",       │
│     'type': 'AirQualitySensor',         │
│     'location': {                        │
│       'type': 'geo:json',               │
│       'value': get_coordinates(loc)     │
│     }                                    │
│   }                                      │
│   FOR each pollutant:                    │
│     entity[pollutant] = {               │
│       'type': 'Number',                 │
│       'value': sensor_data[pollutant],  │
│       'metadata': {                      │
│         'timestamp': {...},             │
│         'unit': {...}                   │
│       }                                  │
│     }                                    │
│                                          │
│ STEP 3: Send to Orion                   │
│   orion_client.update_entity(entity)    │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ orion_client.update_entity()             │
│                                          │
│ url = f"{orion_url}/v2/entities/{id}"   │
│ headers = {                              │
│   'Content-Type': 'application/json',   │
│   'Fiware-Service': service,            │
│   'Fiware-ServicePath': service_path    │
│ }                                        │
│                                          │
│ IF entity exists:                        │
│   response = requests.patch(url, ...)   │
│ ELSE:                                    │
│   response = requests.post(url, ...)    │
│                                          │
│ IF response.status_code not in [200,201]:│
│   RAISE Exception                        │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ FIWARE Orion Context Broker              │
│                                          │
│ 1. Receive HTTP request                  │
│ 2. Validate NGSI-v2 format               │
│ 3. Update entity in storage              │
│ 4. Update timestamp metadata             │
│ 5. Check subscriptions                   │
│ 6. Send notifications (if any)           │
│ 7. Return 200 OK                         │
└──────────────────────────────────────────┘
```

### 3.2 AQI Calculation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    AQI CALCULATION FLOW                          │
└─────────────────────────────────────────────────────────────────┘

INPUT: Pollutant Concentrations
{
  'PM2.5': 45.5,
  'PM10': 120.0,
  'NO2': 35.2,
  'SO2': 15.8,
  'CO': 1.2,
  'O3': 60.0
}
         │
         ▼
┌──────────────────────────────────────────┐
│ calculate_aqi(pollutants)                │
│                                          │
│ sub_indices = []                         │
│                                          │
│ FOR each pollutant in pollutants:        │
│   ├─► calculate_sub_index(pollutant)    │
│   │                                      │
│   │   STEP 1: Get breakpoint range      │
│   │   (C_low, C_high, I_low, I_high) =  │
│   │     get_breakpoint_range(           │
│   │       pollutant,                    │
│   │       concentration                 │
│   │     )                                │
│   │                                      │
│   │   STEP 2: Apply linear interpolation│
│   │   I = [(I_high - I_low) /           │
│   │        (C_high - C_low)] *          │
│   │       (C - C_low) + I_low           │
│   │                                      │
│   │   RETURN I                           │
│   │                                      │
│   └─► sub_indices.append(I)             │
│                                          │
│ overall_aqi = max(sub_indices)          │
│ RETURN overall_aqi                       │
└────────┬─────────────────────────────────┘
         │
         │ AQI = 125
         ▼
┌──────────────────────────────────────────┐
│ get_aqi_status(aqi)                      │
│                                          │
│ IF aqi <= 50:                            │
│   category = 'Good'                      │
│   color = '#00E400'                      │
│ ELIF aqi <= 100:                         │
│   category = 'Moderate'                  │
│   color = '#FFFF00'                      │
│ ELIF aqi <= 150:                         │
│   category = 'Unhealthy for Sensitive'  │
│   color = '#FF9900'                      │
│ ELIF aqi <= 200:                         │
│   category = 'Unhealthy'                 │
│   color = '#FF0000'                      │
│ ELIF aqi <= 300:                         │
│   category = 'Very Unhealthy'           │
│   color = '#8F3F97'                      │
│ ELSE:                                    │
│   category = 'Hazardous'                │
│   color = '#7E0023'                      │
│                                          │
│ health_implications = get_health_text() │
│ cautionary_statement = get_advice()     │
│                                          │
│ RETURN {                                 │
│   'aqi': aqi,                           │
│   'category': category,                 │
│   'color': color,                       │
│   'health_implications': ...,           │
│   'cautionary_statement': ...           │
│ }                                        │
└────────┬─────────────────────────────────┘
         │
         ▼
OUTPUT: AQI Status Object
{
  'aqi': 125,
  'category': 'Unhealthy for Sensitive Groups',
  'color': '#FF9900',
  'health_implications': 'Members of sensitive groups...',
  'cautionary_statement': 'People with respiratory...'
}
```

### 3.3 Machine Learning Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  MACHINE LEARNING PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: DATA EXTRACTION
┌──────────────────────────────────────────┐
│ extract_from_fiware(days=30)             │
│                                          │
│ 1. Calculate date range                  │
│    start = now() - timedelta(days=30)   │
│    end = now()                           │
│                                          │
│ 2. Query FIWARE                          │
│    entities = orion_client.query_entities(│
│      entity_type='AirQualitySensor',    │
│      filters={                           │
│        'timestamp': {'$gte': start,     │
│                      '$lte': end}       │
│      }                                   │
│    )                                     │
│                                          │
│ 3. Convert to DataFrame                  │
│    df = pd.DataFrame(entities)          │
│                                          │
│ RETURN df                                │
└────────┬─────────────────────────────────┘
         │
         │ Raw DataFrame (1000+ rows)
         ▼
┌──────────────────────────────────────────┐
│ clean_data(df)                           │
│                                          │
│ 1. Remove outliers                       │
│    df = df[df['PM2.5'] < 500]           │
│    df = df[df['PM2.5'] > 0]             │
│                                          │
│ 2. Handle missing values                 │
│    df = df.fillna(method='ffill')       │
│                                          │
│ 3. Remove duplicates                     │
│    df = df.drop_duplicates(             │
│      subset=['timestamp', 'location']   │
│    )                                     │
│                                          │
│ RETURN df                                │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ feature_engineering(df)                  │
│                                          │
│ 1. Time features                         │
│    df['hour'] = df['timestamp'].dt.hour │
│    df['day_of_week'] = ...              │
│    df['is_weekend'] = ...               │
│    df['month'] = ...                    │
│                                          │
│ 2. Rolling features                      │
│    df['PM2.5_rolling_mean_3h'] =        │
│      df['PM2.5'].rolling(3).mean()      │
│    df['PM2.5_rolling_std_3h'] =         │
│      df['PM2.5'].rolling(3).std()       │
│                                          │
│ 3. Lag features                          │
│    df['PM2.5_lag_1h'] =                 │
│      df['PM2.5'].shift(1)               │
│    df['PM2.5_lag_24h'] =                │
│      df['PM2.5'].shift(24)              │
│                                          │
│ 4. Interaction features                  │
│    df['temp_humidity_interaction'] =    │
│      df['temperature'] * df['humidity'] │
│                                          │
│ RETURN df                                │
└────────┬─────────────────────────────────┘
         │
         │ Feature-rich DataFrame
         ▼
PHASE 2: MODEL TRAINING
┌──────────────────────────────────────────┐
│ train_model(df)                          │
│                                          │
│ 1. Define features and target            │
│    features = ['PM2.5', 'PM10', 'NO2',  │
│                'temperature', 'humidity',│
│                'hour', 'day_of_week',   │
│                'PM2.5_rolling_mean_3h', │
│                'PM2.5_lag_1h', ...]     │
│    target = 'aqi'                       │
│                                          │
│ 2. Train-test split                      │
│    X_train, X_test, y_train, y_test =   │
│      train_test_split(                  │
│        df[features], df[target],        │
│        test_size=0.2,                   │
│        shuffle=False  # time series     │
│      )                                   │
│                                          │
│ 3. Initialize model                      │
│    model = RandomForestRegressor(       │
│      n_estimators=100,                  │
│      max_depth=10,                      │
│      random_state=42                    │
│    )                                     │
│                                          │
│ 4. Train model                           │
│    model.fit(X_train, y_train)          │
│                                          │
│ 5. Evaluate                              │
│    predictions = model.predict(X_test)  │
│    mae = mean_absolute_error(y_test, predictions)│
│    rmse = sqrt(mean_squared_error(...)) │
│    r2 = r2_score(y_test, predictions)   │
│                                          │
│ 6. Save model                            │
│    joblib.dump(model,                   │
│      'models/aqi_predictor.pkl')        │
│                                          │
│ 7. Save metrics                          │
│    metrics_df = pd.DataFrame({          │
│      'timestamp': [now()],              │
│      'model_name': ['RandomForest'],    │
│      'mae': [mae],                      │
│      'rmse': [rmse],                    │
│      'r2_score': [r2]                   │
│    })                                    │
│    metrics_df.to_csv(                   │
│      'data/evaluation_results.csv',     │
│      mode='append'                      │
│    )                                     │
└────────┬─────────────────────────────────┘
         │
         ▼
PHASE 3: PREDICTION
┌──────────────────────────────────────────┐
│ predict_aqi(hours=24)                    │
│                                          │
│ 1. Load model                            │
│    model = joblib.load(                 │
│      'models/aqi_predictor.pkl'         │
│    )                                     │
│                                          │
│ 2. Get current data                      │
│    current = orion_client.get_entity(   │
│      'AQI_Ernakulam_001'                │
│    )                                     │
│                                          │
│ 3. Generate future timestamps            │
│    future_times = [                     │
│      now() + timedelta(hours=i)         │
│      for i in range(1, hours+1)         │
│    ]                                     │
│                                          │
│ 4. Create feature matrix                 │
│    features = []                        │
│    FOR each time in future_times:       │
│      features.append({                  │
│        'PM2.5': current['PM2.5'],      │
│        'hour': time.hour,               │
│        'day_of_week': time.weekday(),  │
│        ... (other features)             │
│      })                                  │
│    X = pd.DataFrame(features)           │
│                                          │
│ 5. Make predictions                      │
│    predictions = model.predict(X)       │
│                                          │
│ 6. Format output                         │
│    results = [                          │
│      {                                   │
│        'timestamp': future_times[i],    │
│        'aqi': predictions[i],           │
│        'confidence': calculate_confidence()│
│      }                                   │
│      for i in range(len(predictions))   │
│    ]                                     │
│                                          │
│ RETURN results                           │
└──────────────────────────────────────────┘
```

### 3.4 Dashboard Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DASHBOARD DATA FLOW                          │
└─────────────────────────────────────────────────────────────────┘

USER REQUEST
   │
   │ HTTP GET http://localhost:5000/
   ▼
┌──────────────────────────────────────────┐
│ Flask Route: @app.route('/')             │
│                                          │
│ def index():                             │
│   # Fetch current data                   │
│   current_data = get_current_data()     │
│   # Fetch predictions                    │
│   predictions = get_predictions()       │
│   # Render template                      │
│   return render_template(               │
│     'dashboard.html',                   │
│     data=current_data,                  │
│     predictions=predictions             │
│   )                                      │
└────────┬─────────────────────────────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│ get_current_data()│  │ get_predictions()│
│                  │  │                  │
│ 1. Query FIWARE  │  │ 1. Load model    │
│    entities =    │  │    model = load()│
│    orion_client. │  │                  │
│    query_entities│  │ 2. Get current   │
│    (...)         │  │    data          │
│                  │  │                  │
│ 2. Calculate AQI │  │ 3. Predict       │
│    FOR entity:   │  │    predictions = │
│      aqi = calc_ │  │    predict_aqi() │
│      aqi(entity) │  │                  │
│      status =    │  │ 4. Format        │
│      get_status()│  │    RETURN list   │
│                  │  │                  │
│ 3. Format        │  │                  │
│    RETURN {      │  │                  │
│      'locations':│  │                  │
│      [...],      │  │                  │
│      'current_   │  │                  │
│      aqi': ...,  │  │                  │
│      'status':..}│  │                  │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│ render_template('dashboard.html', ...)   │
│                                          │
│ Jinja2 Template Processing:              │
│ - Insert current AQI values              │
│ - Generate charts (Plotly)               │
│ - Create map markers (Leaflet)           │
│ - Render prediction graphs               │
└────────┬─────────────────────────────────┘
         │
         │ HTML Response
         ▼
┌──────────────────────────────────────────┐
│ User's Browser                           │
│                                          │
│ 1. Render HTML                           │
│ 2. Execute JavaScript                    │
│    - Initialize charts                   │
│    - Setup auto-refresh                  │
│    - Load map                            │
│                                          │
│ 3. Auto-refresh (every 30s)              │
│    setInterval(() => {                   │
│      fetch('/api/current')               │
│        .then(data => updateUI(data))     │
│    }, 30000)                             │
└──────────────────────────────────────────┘
```

---

## 4. Data Formats & Transformations

### 4.1 Sensor Data Format (Raw)

```python
{
  "sensor_id": "AQI_001",
  "location": "Ernakulam",
  "latitude": 9.9816,
  "longitude": 76.2999,
  "PM2.5": 45.5,        # µg/m³
  "PM10": 120.0,        # µg/m³
  "NO2": 35.2,          # ppb
  "SO2": 15.8,          # ppb
  "CO": 1.2,            # ppm
  "O3": 60.0,           # ppb
  "temperature": 28.5,  # °C
  "humidity": 75.0,     # %
  "timestamp": "2024-12-20T10:30:00Z"
}
```

### 4.2 NGSI-v2 Entity Format (FIWARE)

```json
{
  "id": "AQI_Ernakulam_001",
  "type": "AirQualitySensor",
  "location": {
    "type": "geo:json",
    "value": {
      "type": "Point",
      "coordinates": [76.2999, 9.9816]
    }
  },
  "PM2_5": {
    "type": "Number",
    "value": 45.5,
    "metadata": {
      "timestamp": {
        "type": "DateTime",
        "value": "2024-12-20T10:30:00Z"
      },
      "unit": {
        "type": "Text",
        "value": "µg/m³"
      }
    }
  },
  "PM10": {
    "type": "Number",
    "value": 120.0,
    "metadata": {
      "timestamp": {"type": "DateTime", "value": "2024-12-20T10:30:00Z"},
      "unit": {"type": "Text", "value": "µg/m³"}
    }
  },
  "NO2": {
    "type": "Number",
    "value": 35.2,
    "metadata": {
      "timestamp": {"type": "DateTime", "value": "2024-12-20T10:30:00Z"},
      "unit": {"type": "Text", "value": "ppb"}
    }
  },
  "SO2": {
    "type": "Number",
    "value": 15.8,
    "metadata": {
      "timestamp": {"type": "DateTime", "value": "2024-12-20T10:30:00Z"},
      "unit": {"type": "Text", "value": "ppb"}
    }
  },
  "CO": {
    "type": "Number",
    "value": 1.2,
    "metadata": {
      "timestamp": {"type": "DateTime", "value": "2024-12-20T10:30:00Z"},
      "unit": {"type": "Text", "value": "ppm"}
    }
  },
  "O3": {
    "type": "Number",
    "value": 60.0,
    "metadata": {
      "timestamp": {"type": "DateTime", "value": "2024-12-20T10:30:00Z"},
      "unit": {"type": "Text", "value": "ppb"}
    }
  },
  "temperature": {
    "type": "Number",
    "value": 28.5,
    "metadata": {
      "unit": {"type": "Text", "value": "°C"}
    }
  },
  "humidity": {
    "type": "Number",
    "value": 75.0,
    "metadata": {
      "unit": {"type": "Text", "value": "%"}
    }
  },
  "aqi": {
    "type": "Number",
    "value": 125,
    "metadata": {
      "calculated_at": {"type": "DateTime", "value": "2024-12-20T10:30:05Z"}
    }
  },
  "status": {
    "type": "Text",
    "value": "Unhealthy for Sensitive Groups"
  }
}
```

### 4.3 ML Feature Matrix Format

```python
# Pandas DataFrame
   timestamp            PM2.5  PM10  NO2  SO2   CO   O3  temp  humidity  hour  day_of_week  PM2.5_rolling_mean_3h  PM2.5_lag_1h
0  2024-12-20 08:00:00  42.3   115   32   14    1.1  55  27.5  78        8     4            40.5                   38.2
1  2024-12-20 09:00:00  45.5   120   35   16    1.2  60  28.5  75        9     4            43.1                   42.3
2  2024-12-20 10:00:00  48.2   125   38   18    1.3  65  29.0  72        10    4            45.3                   45.5
...
```

### 4.4 Prediction Output Format

```python
{
  "predictions": [
    {
      "timestamp": "2024-12-20T11:00:00Z",
      "aqi": 128,
      "confidence": 0.92,
      "category": "Unhealthy for Sensitive Groups",
      "lower_bound": 120,
      "upper_bound": 136
    },
    {
      "timestamp": "2024-12-20T12:00:00Z",
      "aqi": 132,
      "confidence": 0.89,
      "category": "Unhealthy for Sensitive Groups",
      "lower_bound": 122,
      "upper_bound": 142
    }
  ],
  "model_version": "v1.2.3",
  "generated_at": "2024-12-20T10:30:00Z",
  "location": "Ernakulam"
}
```

### 4.5 Dashboard API Response Format

```json
{
  "current": {
    "location": "Ernakulam",
    "aqi": 125,
    "category": "Unhealthy for Sensitive Groups",
    "color": "#FF9900",
    "pollutants": {
      "PM2.5": 45.5,
      "PM10": 120.0,
      "NO2": 35.2,
      "SO2": 15.8,
      "CO": 1.2,
      "O3": 60.0
    },
    "weather": {
      "temperature": 28.5,
      "humidity": 75.0
    },
    "timestamp": "2024-12-20T10:30:00Z"
  },
  "predictions": [
    {"timestamp": "2024-12-20T11:00:00Z", "aqi": 128},
    {"timestamp": "2024-12-20T12:00:00Z", "aqi": 132}
  ],
  "historical": [
    {"timestamp": "2024-12-20T08:00:00Z", "aqi": 115},
    {"timestamp": "2024-12-20T09:00:00Z", "aqi": 120}
  ]
}
```

---

## 5. Data Flow Scenarios

### 5.1 Scenario 1: New Sensor Reading

**Trigger**: Timer event (every 60 seconds)

**Flow**:
1. Sensor simulator generates reading
2. Context ingestor validates data
3. Transform to NGSI-v2 format
4. Send to FIWARE Orion
5. Orion stores entity
6. AQI calculation triggered
7. Status mapping applied
8. Updated entity stored
9. Dashboard polls for updates
10. User sees new data

**Latency**: ~2-5 seconds end-to-end

### 5.2 Scenario 2: ML Model Training

**Trigger**: Manual execution or scheduled job (daily/weekly)

**Flow**:
1. Extract 30 days of historical data from FIWARE
2. Clean and preprocess data
3. Feature engineering
4. Train-test split
5. Train RandomForest model
6. Evaluate performance
7. Save model to disk
8. Log metrics to CSV
9. Model ready for predictions

**Duration**: ~5-30 minutes (depending on data volume)

### 5.3 Scenario 3: Prediction Request

**Trigger**: Dashboard loads or user requests predictions

**Flow**:
1. Load trained model from disk
2. Get current sensor data from FIWARE
3. Generate future timestamps (24 hours)
4. Create feature matrix
5. Model predicts AQI values
6. Calculate confidence intervals
7. Format predictions
8. Return to dashboard
9. Dashboard renders prediction chart

**Latency**: ~1-2 seconds

### 5.4 Scenario 4: Real-time Dashboard Update

**Trigger**: Auto-refresh timer (every 30 seconds)

**Flow**:
1. Browser sends AJAX request to `/api/current`
2. Server queries FIWARE for latest entities
3. Calculate AQI for each location
4. Format response as JSON
5. Send to browser
6. JavaScript updates UI elements
7. Charts re-render with new data

**Latency**: <1 second

---

## 6. Data Quality & Validation

### 6.1 Validation Rules

**Sensor Data Validation**:
```python
def validate_data(sensor_data):
    # Range checks
    if not (0 <= sensor_data['PM2.5'] <= 500):
        return False
    if not (0 <= sensor_data['PM10'] <= 1000):
        return False
    if not (0 <= sensor_data['NO2'] <= 200):
        return False
    if not (0 <= sensor_data['SO2'] <= 200):
        return False
    if not (0 <= sensor_data['CO'] <= 50):
        return False
    if not (0 <= sensor_data['O3'] <= 300):
        return False
    
    # Timestamp check
    if not is_valid_timestamp(sensor_data['timestamp']):
        return False
    
    # Location check
    if sensor_data['location'] not in VALID_LOCATIONS:
        return False
    
    return True
```

### 6.2 Data Cleaning Steps

1. **Outlier Removal**: Remove values beyond 3 standard deviations
2. **Missing Value Handling**: Forward fill for time series continuity
3. **Duplicate Removal**: Drop duplicate timestamps per location
4. **Consistency Checks**: Ensure temporal ordering

### 6.3 Data Quality Metrics

- **Completeness**: % of expected readings received
- **Accuracy**: Comparison with reference sensors (if available)
- **Timeliness**: Latency from generation to storage
- **Consistency**: Cross-pollutant correlation checks

---

## 7. Performance Considerations

### 7.1 Throughput

- **Sensor Data**: 10-100 readings/minute
- **FIWARE Queries**: 100-1000 queries/minute
- **Dashboard Requests**: 10-100 requests/minute
- **ML Predictions**: 1-10 predictions/minute

### 7.2 Latency Targets

| Operation | Target Latency |
|-----------|----------------|
| Sensor → FIWARE | <1 second |
| AQI Calculation | <100 ms |
| FIWARE Query | <200 ms |
| Dashboard Load | <2 seconds |
| ML Prediction | <1 second |

### 7.3 Optimization Strategies

1. **Caching**:
   - Cache FIWARE queries (30s TTL)
   - Cache ML predictions (5 min TTL)
   - Browser caching for static assets

2. **Batch Processing**:
   - Batch sensor data ingestion
   - Bulk FIWARE updates

3. **Asynchronous Operations**:
   - Background ML training
   - Async FIWARE queries

4. **Database Indexing**:
   - Index on entity_id, timestamp, location

---

## 8. Error Handling & Recovery

### 8.1 Error Scenarios

1. **FIWARE Unavailable**:
   - Buffer sensor data locally
   - Retry with exponential backoff
   - Alert administrator

2. **Invalid Sensor Data**:
   - Log error
   - Skip reading
   - Continue with next reading

3. **ML Model Failure**:
   - Fall back to historical average
   - Log error
   - Notify for retraining

4. **Dashboard Error**:
   - Display cached data
   - Show error message to user
   - Retry connection

### 8.2 Data Recovery

- **FIWARE Backup**: Periodic entity export to CSV
- **Model Versioning**: Keep last 3 model versions
- **Configuration Backup**: Version control for settings.yaml

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Maintained By**: Digital Twin AQI Kochi Team
