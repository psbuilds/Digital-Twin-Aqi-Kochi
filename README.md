# Digital Twin AQI Kochi

A comprehensive Air Quality Index (AQI) monitoring and prediction system for Kochi using Digital Twin technology, FIWARE Context Broker, and Machine Learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Detailed File Documentation](#detailed-file-documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

This project implements a Digital Twin system for real-time Air Quality Index monitoring and prediction in Kochi. It integrates sensor data simulation, FIWARE-based context management, machine learning predictions, and an interactive dashboard for visualization.

**Key Features:**
- Real-time AQI sensor data simulation
- FIWARE Orion Context Broker integration
- Machine learning-based AQI prediction
- Interactive web dashboard
- Standards-compliant data models
- Comprehensive documentation

---

## 📁 Project Structure

```
Digital-Twin-Aqi-Kochi/
├── aqi_logic/              # AQI calculation and status mapping logic
│   ├── current_aqi_rules.py
│   └── status_mapping.py
├── config/                 # Configuration files
│   └── settings.yaml
├── dashboard/              # Web dashboard application
│   └── app.py
├── data/                   # Data storage and results
│   └── evaluation_results.csv
├── docs/                   # Project documentation
│   ├── architecture.md
│   ├── data_flow.md
│   └── setup.md
├── fiware/                 # FIWARE integration components
│   ├── context_ingestor.py
│   ├── entity_models.json
│   └── orion_client.py
├── ml/                     # Machine learning components
│   ├── data_extraction.py
│   ├── predict_future_aqi.py
│   └── train_model.py
├── sensors/                # Sensor simulation
│   └── aqi_sensor_simulator.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📚 Detailed File Documentation

### Root Directory Files

#### `LICENSE`
**Purpose:** Contains the software license governing the use, modification, and distribution of this project.

**Details:**
- Defines legal terms and conditions
- Specifies usage rights and restrictions
- Protects intellectual property
- Ensures compliance with open-source standards

#### `README.md`
**Purpose:** Main project documentation providing overview, setup instructions, and usage guidelines.

**Details:**
- Project introduction and objectives
- Installation and setup instructions
- Usage examples and workflows
- Architecture overview
- Contribution guidelines

#### `requirements.txt`
**Purpose:** Lists all Python package dependencies required to run the project.

**Details:**
- Specifies exact package versions for reproducibility
- Includes core dependencies:
  - `flask` or `streamlit` - Web dashboard framework
  - `requests` - HTTP client for FIWARE API calls
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `scikit-learn` - Machine learning algorithms
  - `pyyaml` - YAML configuration parsing
  - `matplotlib`/`plotly` - Data visualization

**Installation:**
```bash
pip install -r requirements.txt
```

---

### `/aqi_logic` Directory

This directory contains the core business logic for AQI calculation and interpretation.

#### `current_aqi_rules.py`
**Purpose:** Implements AQI calculation algorithms based on pollutant concentrations.

**Key Functions:**
- `calculate_aqi(pollutants: dict) -> float`: Calculates overall AQI from individual pollutant levels
- `calculate_sub_index(pollutant: str, concentration: float) -> float`: Computes sub-index for specific pollutant
- `get_breakpoint_range(pollutant: str, concentration: float) -> tuple`: Retrieves appropriate concentration breakpoints

**Algorithm Details:**
- Follows EPA (Environmental Protection Agency) or CPCB (Central Pollution Control Board) standards
- Supports multiple pollutants: PM2.5, PM10, NO2, SO2, CO, O3
- Uses piecewise linear function for sub-index calculation
- Returns maximum sub-index as overall AQI

**Calculation Formula:**
```
I = [(I_high - I_low) / (C_high - C_low)] × (C - C_low) + I_low

Where:
- I = AQI sub-index
- C = Pollutant concentration
- C_low, C_high = Concentration breakpoints
- I_low, I_high = Index breakpoints
```

**Example Usage:**
```python
pollutants = {
    'PM2.5': 45.5,
    'PM10': 120.0,
    'NO2': 35.2,
    'SO2': 15.8,
    'CO': 1.2,
    'O3': 60.0
}
aqi = calculate_aqi(pollutants)
```

#### `status_mapping.py`
**Purpose:** Maps numerical AQI values to categorical health status levels and recommendations.

**Key Functions:**
- `get_aqi_status(aqi: float) -> dict`: Returns status category, color code, and health advisory
- `get_health_implications(aqi: float) -> str`: Provides detailed health impact description
- `get_cautionary_statement(aqi: float) -> str`: Returns recommended actions for different groups

**AQI Categories:**
| Range | Category | Color | Health Implications |
|-------|----------|-------|---------------------|
| 0-50 | Good | Green | Air quality is satisfactory |
| 51-100 | Moderate | Yellow | Acceptable for most people |
| 101-150 | Unhealthy for Sensitive Groups | Orange | Sensitive groups may experience effects |
| 151-200 | Unhealthy | Red | Everyone may begin to experience effects |
| 201-300 | Very Unhealthy | Purple | Health alert: everyone may experience serious effects |
| 301+ | Hazardous | Maroon | Health warnings of emergency conditions |

**Return Structure:**
```python
{
    'category': 'Moderate',
    'color': '#FFFF00',
    'health_implications': 'Air quality is acceptable...',
    'cautionary_statement': 'Unusually sensitive people should consider...',
    'sensitive_groups': ['children', 'elderly', 'respiratory_patients']
}
```

---

### `/config` Directory

Contains configuration files for system-wide settings.

#### `settings.yaml`
**Purpose:** Centralized configuration file for all system parameters.

**Configuration Sections:**

**1. FIWARE Settings:**
```yaml
fiware:
  orion_url: "http://localhost:1026"
  service: "aqi_kochi"
  service_path: "/sensors"
  entity_type: "AirQualitySensor"
```

**2. Sensor Configuration:**
```yaml
sensors:
  update_interval: 60  # seconds
  locations:
    - name: "Ernakulam"
      latitude: 9.9816
      longitude: 76.2999
    - name: "Fort Kochi"
      latitude: 9.9658
      longitude: 76.2427
  pollutants:
    - PM2.5
    - PM10
    - NO2
    - SO2
    - CO
    - O3
```

**3. Machine Learning Parameters:**
```yaml
ml:
  model_type: "RandomForest"
  training_window: 30  # days
  prediction_horizon: 24  # hours
  features:
    - temperature
    - humidity
    - wind_speed
    - hour_of_day
    - day_of_week
```

**4. Dashboard Settings:**
```yaml
dashboard:
  port: 5000
  refresh_rate: 30  # seconds
  max_history: 1000  # data points
```

---

### `/dashboard` Directory

Contains the web-based visualization interface.

#### `app.py`
**Purpose:** Main dashboard application providing real-time AQI monitoring and visualization.

**Framework:** Flask or Streamlit

**Key Components:**

**1. Data Retrieval:**
- Fetches real-time data from FIWARE Orion Context Broker
- Retrieves historical data for trend analysis
- Loads ML predictions for future AQI

**2. Visualization Features:**
- Real-time AQI gauge/meter
- Time-series charts for pollutant trends
- Geographic map with sensor locations
- Prediction graphs for next 24 hours
- Comparative analysis across locations

**3. Routes/Pages:**
- `/` - Main dashboard with current AQI
- `/historical` - Historical data analysis
- `/predictions` - ML-based forecasts
- `/map` - Geographic visualization
- `/api/current` - REST API for current data

**4. Interactive Elements:**
- Location selector dropdown
- Date range picker for historical data
- Pollutant filter checkboxes
- Auto-refresh toggle
- Export data functionality

**Technologies Used:**
- **Backend:** Flask/Streamlit
- **Frontend:** HTML5, CSS3, JavaScript
- **Charts:** Plotly/Chart.js
- **Maps:** Leaflet.js or Folium

**Example Code Structure:**
```python
from flask import Flask, render_template, jsonify
from fiware.orion_client import OrionClient
from ml.predict_future_aqi import predict_aqi

app = Flask(__name__)
orion = OrionClient()

@app.route('/')
def index():
    current_data = orion.get_latest_data()
    return render_template('dashboard.html', data=current_data)

@app.route('/api/predictions')
def get_predictions():
    predictions = predict_aqi(hours=24)
    return jsonify(predictions)
```

---

### `/data` Directory

Stores data files, results, and datasets.

#### `evaluation_results.csv`
**Purpose:** Contains machine learning model evaluation metrics and performance results.

**Structure:**
```csv
timestamp,model_name,mae,rmse,r2_score,mape,training_samples,test_samples
2024-01-15 10:30:00,RandomForest,5.23,7.45,0.89,8.5,5000,1000
2024-01-15 11:00:00,GradientBoosting,4.87,6.92,0.91,7.8,5000,1000
```

**Metrics Explained:**
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual AQI
- **RMSE (Root Mean Square Error):** Square root of average squared differences
- **R² Score:** Proportion of variance explained by the model (0-1, higher is better)
- **MAPE (Mean Absolute Percentage Error):** Average percentage error

**Usage:**
- Model comparison and selection
- Performance tracking over time
- Hyperparameter tuning validation
- Reporting and documentation

---

### `/docs` Directory

Contains comprehensive project documentation.

#### `architecture.md`
**Purpose:** Describes the system architecture, components, and their interactions.

**Contents:**
1. **System Overview:**
   - High-level architecture diagram
   - Component relationships
   - Data flow between modules

2. **Component Details:**
   - Sensor Layer: Data collection and simulation
   - Context Management: FIWARE integration
   - Processing Layer: AQI calculation and ML
   - Presentation Layer: Dashboard and APIs

3. **Technology Stack:**
   - Programming languages (Python)
   - Frameworks (Flask/Streamlit)
   - Databases (if any)
   - External services (FIWARE)

4. **Design Patterns:**
   - Client-Server architecture
   - Repository pattern for data access
   - Observer pattern for real-time updates

5. **Scalability Considerations:**
   - Horizontal scaling strategies
   - Load balancing
   - Caching mechanisms

#### `data_flow.md`
**Purpose:** Documents the flow of data through the system from sensors to dashboard.

**Contents:**
1. **Data Collection:**
   ```
   Sensors → Simulator → JSON Format
   ```

2. **Data Ingestion:**
   ```
   Simulator → Context Ingestor → FIWARE Orion
   ```

3. **Data Processing:**
   ```
   FIWARE → AQI Calculator → Status Mapper
   ```

4. **Data Storage:**
   ```
   Processed Data → CSV/Database → Historical Archive
   ```

5. **Data Visualization:**
   ```
   FIWARE/Storage → Dashboard → User Interface
   ```

6. **ML Pipeline:**
   ```
   Historical Data → Feature Engineering → Model Training → Predictions
   ```

**Data Formats:**
- Sensor data: JSON (NGSI-v2 format)
- Processed data: CSV, Pandas DataFrame
- API responses: JSON
- Configuration: YAML

#### `setup.md`
**Purpose:** Provides step-by-step installation and setup instructions.

**Contents:**
1. **Prerequisites:**
   - Python 3.8+
   - Docker (for FIWARE)
   - Git

2. **FIWARE Setup:**
   ```bash
   docker run -d --name orion -p 1026:1026 fiware/orion
   ```

3. **Project Installation:**
   ```bash
   git clone https://github.com/username/Digital-Twin-Aqi-Kochi.git
   cd Digital-Twin-Aqi-Kochi
   pip install -r requirements.txt
   ```

4. **Configuration:**
   - Edit `config/settings.yaml`
   - Set FIWARE endpoint
   - Configure sensor locations

5. **Running Components:**
   ```bash
   # Start sensor simulator
   python sensors/aqi_sensor_simulator.py
   
   # Start dashboard
   python dashboard/app.py
   
   # Train ML model
   python ml/train_model.py
   ```

6. **Verification:**
   - Check FIWARE entities
   - Access dashboard
   - Verify data flow

---

### `/fiware` Directory

Handles integration with FIWARE Orion Context Broker.

#### `orion_client.py`
**Purpose:** Provides a Python client for interacting with FIWARE Orion Context Broker.

**Key Classes:**

**OrionClient:**
```python
class OrionClient:
    def __init__(self, url, service, service_path)
    def create_entity(self, entity_data)
    def update_entity(self, entity_id, attributes)
    def get_entity(self, entity_id)
    def query_entities(self, entity_type, filters)
    def delete_entity(self, entity_id)
    def subscribe(self, subscription_data)
```

**Key Methods:**

1. **create_entity():**
   - Creates new entity in Orion
   - Validates NGSI-v2 format
   - Returns entity ID

2. **update_entity():**
   - Updates existing entity attributes
   - Supports partial updates
   - Handles timestamps automatically

3. **query_entities():**
   - Retrieves entities by type
   - Supports filtering and pagination
   - Returns list of entities

4. **subscribe():**
   - Creates subscriptions for real-time notifications
   - Configures webhook endpoints
   - Manages subscription lifecycle

**HTTP Headers:**
```python
headers = {
    'Content-Type': 'application/json',
    'Fiware-Service': self.service,
    'Fiware-ServicePath': self.service_path
}
```

**Error Handling:**
- Connection errors
- Invalid entity format
- Authentication failures
- Rate limiting

#### `context_ingestor.py`
**Purpose:** Ingests sensor data into FIWARE Orion Context Broker.

**Key Functions:**

1. **ingest_sensor_data():**
   - Receives sensor readings
   - Transforms to NGSI-v2 format
   - Sends to Orion via OrionClient

2. **batch_ingest():**
   - Handles multiple sensor readings
   - Optimizes API calls
   - Implements retry logic

3. **validate_data():**
   - Checks data completeness
   - Validates value ranges
   - Ensures timestamp format

**Data Transformation:**
```python
# Input (sensor format)
{
    'sensor_id': 'AQI_001',
    'location': 'Ernakulam',
    'PM2.5': 45.5,
    'timestamp': '2024-01-15T10:30:00Z'
}

# Output (NGSI-v2 format)
{
    'id': 'AQI_001',
    'type': 'AirQualitySensor',
    'location': {
        'type': 'geo:json',
        'value': {'type': 'Point', 'coordinates': [76.2999, 9.9816]}
    },
    'PM2_5': {
        'type': 'Number',
        'value': 45.5,
        'metadata': {'timestamp': {'type': 'DateTime', 'value': '2024-01-15T10:30:00Z'}}
    }
}
```

**Features:**
- Automatic retry on failure
- Data buffering for offline scenarios
- Logging and monitoring
- Performance metrics

#### `entity_models.json`
**Purpose:** Defines NGSI-v2 data models for FIWARE entities.

**Structure:**
```json
{
  "AirQualitySensor": {
    "type": "AirQualitySensor",
    "description": "Air quality monitoring sensor",
    "attributes": {
      "location": {
        "type": "geo:json",
        "description": "Geographic location of sensor"
      },
      "PM2_5": {
        "type": "Number",
        "unit": "µg/m³",
        "description": "Particulate Matter 2.5 concentration"
      },
      "PM10": {
        "type": "Number",
        "unit": "µg/m³",
        "description": "Particulate Matter 10 concentration"
      },
      "NO2": {
        "type": "Number",
        "unit": "ppb",
        "description": "Nitrogen Dioxide concentration"
      },
      "SO2": {
        "type": "Number",
        "unit": "ppb",
        "description": "Sulfur Dioxide concentration"
      },
      "CO": {
        "type": "Number",
        "unit": "ppm",
        "description": "Carbon Monoxide concentration"
      },
      "O3": {
        "type": "Number",
        "unit": "ppb",
        "description": "Ozone concentration"
      },
      "temperature": {
        "type": "Number",
        "unit": "°C",
        "description": "Ambient temperature"
      },
      "humidity": {
        "type": "Number",
        "unit": "%",
        "description": "Relative humidity"
      },
      "aqi": {
        "type": "Number",
        "description": "Calculated Air Quality Index"
      },
      "status": {
        "type": "Text",
        "description": "AQI status category"
      }
    }
  }
}
```

**Usage:**
- Schema validation
- Entity creation templates
- API documentation
- Client code generation

---

### `/ml` Directory

Contains machine learning components for AQI prediction.

#### `data_extraction.py`
**Purpose:** Extracts and prepares data for machine learning model training.

**Key Functions:**

1. **extract_from_fiware():**
   - Queries historical data from FIWARE
   - Retrieves specified time range
   - Returns pandas DataFrame

2. **extract_from_csv():**
   - Loads data from CSV files
   - Handles missing values
   - Parses timestamps

3. **merge_datasets():**
   - Combines multiple data sources
   - Aligns timestamps
   - Resolves conflicts

4. **clean_data():**
   - Removes outliers
   - Handles missing values (interpolation/forward fill)
   - Filters invalid readings

5. **feature_engineering():**
   - Creates time-based features (hour, day, month)
   - Calculates rolling averages
   - Generates lag features
   - Computes pollutant ratios

**Feature Creation Examples:**
```python
# Time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Rolling features
df['PM2.5_rolling_mean_3h'] = df['PM2.5'].rolling(3).mean()
df['PM2.5_rolling_std_3h'] = df['PM2.5'].rolling(3).std()

# Lag features
df['PM2.5_lag_1h'] = df['PM2.5'].shift(1)
df['PM2.5_lag_24h'] = df['PM2.5'].shift(24)
```

**Data Quality Checks:**
- Range validation
- Consistency checks
- Completeness assessment
- Temporal continuity

#### `train_model.py`
**Purpose:** Trains machine learning models for AQI prediction.

**Workflow:**

1. **Data Loading:**
   ```python
   from data_extraction import extract_from_fiware, feature_engineering
   data = extract_from_fiware(days=30)
   features = feature_engineering(data)
   ```

2. **Train-Test Split:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       features, target, test_size=0.2, shuffle=False
   )
   ```

3. **Model Training:**
   ```python
   from sklearn.ensemble import RandomForestRegressor
   model = RandomForestRegressor(n_estimators=100, max_depth=10)
   model.fit(X_train, y_train)
   ```

4. **Hyperparameter Tuning:**
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [5, 10, 15],
       'min_samples_split': [2, 5, 10]
   }
   grid_search = GridSearchCV(model, param_grid, cv=5)
   ```

5. **Model Evaluation:**
   ```python
   from sklearn.metrics import mean_absolute_error, r2_score
   predictions = model.predict(X_test)
   mae = mean_absolute_error(y_test, predictions)
   r2 = r2_score(y_test, predictions)
   ```

6. **Model Persistence:**
   ```python
   import joblib
   joblib.dump(model, 'models/aqi_predictor.pkl')
   ```

**Supported Models:**
- Random Forest Regressor
- Gradient Boosting
- LSTM (for time series)
- XGBoost
- Linear Regression (baseline)

**Evaluation Metrics:**
- MAE, RMSE, R², MAPE
- Feature importance analysis
- Residual plots
- Cross-validation scores

#### `predict_future_aqi.py`
**Purpose:** Generates AQI predictions for future time periods.

**Key Functions:**

1. **load_model():**
   ```python
   import joblib
   model = joblib.load('models/aqi_predictor.pkl')
   ```

2. **predict_aqi():**
   ```python
   def predict_aqi(hours=24, location='Ernakulam'):
       # Load latest data
       current_data = get_current_conditions()
       
       # Generate future timestamps
       future_times = generate_future_timestamps(hours)
       
       # Create feature matrix
       features = create_prediction_features(current_data, future_times)
       
       # Make predictions
       predictions = model.predict(features)
       
       return predictions
   ```

3. **create_prediction_features():**
   - Uses latest sensor readings
   - Generates time-based features for future
   - Applies same transformations as training

4. **get_confidence_intervals():**
   - Calculates prediction uncertainty
   - Uses quantile regression or ensemble variance
   - Returns upper/lower bounds

**Output Format:**
```python
{
    'predictions': [
        {'timestamp': '2024-01-15T11:00:00Z', 'aqi': 85, 'confidence': 0.92},
        {'timestamp': '2024-01-15T12:00:00Z', 'aqi': 88, 'confidence': 0.89},
        ...
    ],
    'model_version': 'v1.2.3',
    'generated_at': '2024-01-15T10:30:00Z'
}
```

**Features:**
- Real-time prediction API
- Batch prediction support
- Uncertainty quantification
- Model versioning

---

### `/sensors` Directory

Handles sensor data simulation and collection.

#### `aqi_sensor_simulator.py`
**Purpose:** Simulates AQI sensor data for testing and development.

**Key Components:**

1. **SensorSimulator Class:**
   ```python
   class SensorSimulator:
       def __init__(self, location, config)
       def generate_reading(self)
       def add_noise(self, value, noise_level)
       def simulate_pattern(self, pollutant, hour)
       def start_simulation(self, interval)
   ```

2. **Data Generation:**
   - Realistic pollutant patterns
   - Diurnal variations (traffic peaks)
   - Weather correlations
   - Random noise addition

3. **Simulation Patterns:**
   ```python
   # Morning traffic peak (7-9 AM)
   if 7 <= hour <= 9:
       PM2_5 *= 1.5
       NO2 *= 1.8
   
   # Evening peak (5-7 PM)
   if 17 <= hour <= 19:
       PM2_5 *= 1.4
       CO *= 1.6
   ```

4. **Integration:**
   - Sends data to Context Ingestor
   - Configurable update intervals
   - Multiple location support

**Configuration:**
```python
config = {
    'base_values': {
        'PM2.5': 35,
        'PM10': 80,
        'NO2': 25,
        'SO2': 10,
        'CO': 0.8,
        'O3': 45
    },
    'noise_level': 0.1,  # 10% random variation
    'update_interval': 60  # seconds
}
```

**Features:**
- Realistic data patterns
- Configurable baselines
- Event simulation (pollution spikes)
- Multi-sensor coordination

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Docker (for FIWARE Orion)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Digital-Twin-Aqi-Kochi.git
cd Digital-Twin-Aqi-Kochi
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Setup FIWARE Orion
```bash
docker run -d --name orion -p 1026:1026 fiware/orion
```

### Step 4: Configure Settings
Edit `config/settings.yaml` with your specific configuration.

---

## 💻 Usage

### Start Sensor Simulator
```bash
python sensors/aqi_sensor_simulator.py
```

### Train ML Model
```bash
python ml/train_model.py
```

### Launch Dashboard
```bash
python dashboard/app.py
```

Access dashboard at: `http://localhost:5000`

### Query FIWARE Data
```python
from fiware.orion_client import OrionClient

client = OrionClient()
entities = client.query_entities('AirQualitySensor')
```

---

## 🏗️ Architecture

The system follows a layered architecture:

1. **Sensor Layer:** Data collection and simulation
2. **Context Management:** FIWARE Orion for data storage
3. **Processing Layer:** AQI calculation and ML predictions
4. **Presentation Layer:** Web dashboard

For detailed architecture, see [docs/architecture.md](docs/architecture.md)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

---

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- FIWARE Foundation for Context Broker
- Central Pollution Control Board (CPCB) for AQI standards
- Contributors and maintainers

---

**Last Updated:** December 2024
