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


**Example Usage:**


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


---

### `/config` Directory

Contains configuration files for system-wide settings.

#### `settings.yaml`
**Purpose:** Centralized configuration file for all system parameters.

**Configuration Sections:**

**1. FIWARE Settings:**


**2. Sensor Configuration:**


**3. Machine Learning Parameters:**


**4. Dashboard Settings:**


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


---

### `/data` Directory

Stores data files, results, and datasets.

#### `evaluation_results.csv`
**Purpose:** Contains machine learning model evaluation metrics and performance results.

**Structure:**


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
   

2. **Data Ingestion:**
   

3. **Data Processing:**
   

4. **Data Storage:**
   

5. **Data Visualization:**
   

6. **ML Pipeline:**
   

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
   

3. **Project Installation:**
   

4. **Configuration:**
   - Edit `config/settings.yaml`
   - Set FIWARE endpoint
   - Configure sensor locations

5. **Running Components:**
   

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


**Features:**
- Automatic retry on failure
- Data buffering for offline scenarios
- Logging and monitoring
- Performance metrics

#### `entity_models.json`
**Purpose:** Defines NGSI-v2 data models for FIWARE entities.

**Structure:**


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


**Data Quality Checks:**
- Range validation
- Consistency checks
- Completeness assessment
- Temporal continuity

#### `train_model.py`
**Purpose:** Trains machine learning models for AQI prediction.

**Workflow:**

1. **Data Loading:**
   

2. **Train-Test Split:**
   

3. **Model Training:**
   

4. **Hyperparameter Tuning:**
   

5. **Model Evaluation:**
   

6. **Model Persistence:**
   

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
   

2. **predict_aqi():**
   

3. **create_prediction_features():**
   - Uses latest sensor readings
   - Generates time-based features for future
   - Applies same transformations as training

4. **get_confidence_intervals():**
   - Calculates prediction uncertainty
   - Uses quantile regression or ensemble variance
   - Returns upper/lower bounds

**Output Format:**


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
   

2. **Data Generation:**
   - Realistic pollutant patterns
   - Diurnal variations (traffic peaks)
   - Weather correlations
   - Random noise addition

3. **Simulation Patterns:**
   

4. **Integration:**
   - Sends data to Context Ingestor
   - Configurable update intervals
   - Multiple location support

**Configuration:**


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


### Step 2: Install Dependencies


### Step 3: Setup FIWARE Orion


### Step 4: Configure Settings
Edit `config/settings.yaml` with your specific configuration.

---

## 💻 Usage

### Start Sensor Simulator


### Train ML Model


### Launch Dashboard


Access dashboard at: `http://localhost:5000`

### Query FIWARE Data


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
