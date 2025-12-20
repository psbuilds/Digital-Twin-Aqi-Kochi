# Digital Twin AQI Kochi - System Architecture

## Table of Contents
- [1. System Overview](#1-system-overview)
- [2. Architecture Diagram](#2-architecture-diagram)
- [3. Component Architecture](#3-component-architecture)
- [4. Technology Stack](#4-technology-stack)
- [5. Design Patterns](#5-design-patterns)
- [6. Deployment Architecture](#6-deployment-architecture)
- [7. Security Architecture](#7-security-architecture)
- [8. Scalability & Performance](#8-scalability--performance)

---

## 1. System Overview

The Digital Twin AQI Kochi system is a comprehensive air quality monitoring and prediction platform built on a **layered architecture** pattern. The system integrates real-time sensor data simulation, context-aware data management using FIWARE, machine learning-based predictions, and interactive visualization.

### 1.1 Architectural Principles

- **Separation of Concerns**: Each layer has distinct responsibilities
- **Modularity**: Components are loosely coupled and independently deployable
- **Scalability**: Horizontal scaling support for high-volume data
- **Interoperability**: Standards-compliant (NGSI-v2) data models
- **Extensibility**: Plugin architecture for new sensors and models

### 1.2 System Characteristics

- **Type**: Distributed IoT System with Digital Twin capabilities
- **Pattern**: Layered Architecture + Microservices
- **Communication**: RESTful APIs (HTTP/JSON)
- **Data Format**: NGSI-v2 (Next Generation Service Interface)
- **Real-time**: Event-driven with periodic updates

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Web Dashboard (Flask/Streamlit)                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │ Real-time    │  │ Historical   │  │  Prediction  │  │  Map View  │ │ │
│  │  │ AQI Display  │  │  Analytics   │  │   Charts     │  │  (Leaflet) │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │ │
│  │                                                                          │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │              REST API Endpoints (/api/*)                         │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ HTTP/JSON
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                                     │
│  ┌──────────────────────────┐         ┌──────────────────────────────────┐ │
│  │   AQI Logic Module       │         │   Machine Learning Module        │ │
│  │  ┌──────────────────┐    │         │  ┌────────────────────────────┐  │ │
│  │  │ current_aqi_     │    │         │  │  data_extraction.py        │  │ │
│  │  │ rules.py         │    │         │  │  - Extract from FIWARE     │  │ │
│  │  │ - calculate_aqi()│    │         │  │  - Feature engineering     │  │ │
│  │  │ - sub_index()    │    │         │  │  - Data cleaning           │  │ │
│  │  └──────────────────┘    │         │  └────────────────────────────┘  │ │
│  │  ┌──────────────────┐    │         │  ┌────────────────────────────┐  │ │
│  │  │ status_mapping.py│    │         │  │  train_model.py            │  │ │
│  │  │ - get_status()   │    │         │  │  - RandomForest/XGBoost    │  │ │
│  │  │ - health_advice()│    │         │  │  - Hyperparameter tuning   │  │ │
│  │  └──────────────────┘    │         │  │  - Model evaluation        │  │ │
│  └──────────────────────────┘         │  └────────────────────────────┘  │ │
│                                        │  ┌────────────────────────────┐  │ │
│                                        │  │  predict_future_aqi.py     │  │ │
│                                        │  │  - Load trained model      │  │ │
│                                        │  │  - Generate predictions    │  │ │
│                                        │  │  - Confidence intervals    │  │ │
│                                        │  └────────────────────────────┘  │ │
│                                        └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ Query/Update
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT MANAGEMENT LAYER (FIWARE)                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    FIWARE Orion Context Broker                          │ │
│  │                         (Docker Container)                              │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Entity Storage (NGSI-v2 Format)                                 │  │ │
│  │  │  - AirQualitySensor entities                                     │  │ │
│  │  │  - Real-time context data                                        │  │ │
│  │  │  - Historical data (with persistence)                            │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                          │ │
│  │  REST API: http://localhost:1026/v2/entities                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────┐         ┌──────────────────────────────────┐ │
│  │   orion_client.py        │         │   context_ingestor.py            │ │
│  │  - create_entity()       │         │  - ingest_sensor_data()          │ │
│  │  - update_entity()       │         │  - batch_ingest()                │ │
│  │  - query_entities()      │         │  - validate_data()               │ │
│  │  - subscribe()           │         │  - transform_to_ngsi()           │ │
│  └──────────────────────────┘         └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ Sensor Data (JSON)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SENSOR LAYER                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              aqi_sensor_simulator.py (SensorSimulator)                  │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Simulated Sensors (Multiple Locations)                          │  │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │  │ │
│  │  │  │ Ernakulam   │  │ Fort Kochi  │  │  Location N │              │  │ │
│  │  │  │ Sensor      │  │ Sensor      │  │  Sensor     │              │  │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘              │  │ │
│  │  │                                                                   │  │ │
│  │  │  Pollutant Measurements:                                         │  │ │
│  │  │  • PM2.5 (µg/m³)    • PM10 (µg/m³)    • NO2 (ppb)               │  │ │
│  │  │  • SO2 (ppb)        • CO (ppm)         • O3 (ppb)                │  │ │
│  │  │  • Temperature (°C) • Humidity (%)                               │  │ │
│  │  │                                                                   │  │ │
│  │  │  Features:                                                        │  │ │
│  │  │  - Realistic diurnal patterns (traffic peaks)                    │  │ │
│  │  │  - Weather correlations                                          │  │ │
│  │  │  - Random noise injection                                        │  │ │
│  │  │  - Configurable update intervals (default: 60s)                  │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION LAYER                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        config/settings.yaml                             │ │
│  │  - FIWARE connection settings                                           │ │
│  │  - Sensor configurations (locations, pollutants)                        │ │
│  │  - ML model parameters                                                  │ │
│  │  - Dashboard settings                                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    fiware/entity_models.json                            │ │
│  │  - NGSI-v2 entity schemas                                               │ │
│  │  - Data model definitions                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA STORAGE LAYER                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  data/evaluation_results.csv                                            │ │
│  │  - Model performance metrics (MAE, RMSE, R², MAPE)                      │ │
│  │  - Training/testing results                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Trained ML Models (Serialized)                                         │ │
│  │  - models/aqi_predictor.pkl (joblib format)                             │ │
│  │  - Feature scalers and transformers                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Sensor Layer

**Purpose**: Simulate real-world AQI sensors and generate realistic environmental data.

**Components**:
- `aqi_sensor_simulator.py`: Main simulator class

**Key Responsibilities**:
- Generate pollutant concentration data (PM2.5, PM10, NO2, SO2, CO, O3)
- Simulate realistic patterns (diurnal cycles, traffic peaks)
- Add environmental factors (temperature, humidity)
- Inject controlled noise for realism
- Support multiple geographic locations

**Interfaces**:
- **Output**: JSON sensor readings → Context Ingestor
- **Configuration**: settings.yaml → Sensor parameters

**Design Pattern**: Factory Pattern (creates sensor instances per location)

---

### 3.2 Context Management Layer (FIWARE)

**Purpose**: Manage real-time context information using FIWARE standards.

#### 3.2.1 FIWARE Orion Context Broker

**Technology**: Docker container running FIWARE Orion
**Port**: 1026
**Protocol**: HTTP REST API (NGSI-v2)

**Capabilities**:
- Entity CRUD operations
- Temporal queries
- Subscriptions and notifications
- Multi-tenancy (fiware-service headers)

**Data Model**: NGSI-v2 compliant entities
```json
{
  "id": "AQI_Ernakulam_001",
  "type": "AirQualitySensor",
  "location": {
    "type": "geo:json",
    "value": {"type": "Point", "coordinates": [76.2999, 9.9816]}
  },
  "PM2_5": {"type": "Number", "value": 45.5},
  "aqi": {"type": "Number", "value": 125},
  "status": {"type": "Text", "value": "Unhealthy for Sensitive Groups"}
}
```

#### 3.2.2 Orion Client (`orion_client.py`)

**Purpose**: Python wrapper for Orion Context Broker API

**Key Methods**:
- `create_entity(entity_data)`: Create new sensor entity
- `update_entity(entity_id, attributes)`: Update sensor readings
- `get_entity(entity_id)`: Retrieve specific entity
- `query_entities(entity_type, filters)`: Query with filters
- `subscribe(subscription_data)`: Create subscriptions

**Design Pattern**: Adapter Pattern (wraps FIWARE API)

#### 3.2.3 Context Ingestor (`context_ingestor.py`)

**Purpose**: Transform and ingest sensor data into FIWARE

**Key Functions**:
- `ingest_sensor_data()`: Single sensor reading ingestion
- `batch_ingest()`: Bulk data ingestion
- `validate_data()`: Data quality checks
- Transform sensor format → NGSI-v2 format

**Design Pattern**: Transformer Pattern

---

### 3.3 Processing Layer

#### 3.3.1 AQI Logic Module

**Components**:
1. **current_aqi_rules.py**
   - Implements EPA/CPCB AQI calculation algorithms
   - Piecewise linear interpolation for sub-indices
   - Supports 6 pollutants (PM2.5, PM10, NO2, SO2, CO, O3)
   
2. **status_mapping.py**
   - Maps AQI values to health categories
   - Provides health implications and advisories
   - Color coding for visualization

**AQI Calculation Formula**:
```
I = [(I_high - I_low) / (C_high - C_low)] × (C - C_low) + I_low

Overall AQI = max(sub_index_PM2.5, sub_index_PM10, ..., sub_index_O3)
```

**Design Pattern**: Strategy Pattern (different calculation strategies)

#### 3.3.2 Machine Learning Module

**Components**:

1. **data_extraction.py**
   - Extract historical data from FIWARE
   - Feature engineering (time features, rolling averages, lag features)
   - Data cleaning and preprocessing
   - Train-test split preparation

2. **train_model.py**
   - Model training pipeline
   - Supported algorithms: RandomForest, XGBoost, GradientBoosting, LSTM
   - Hyperparameter tuning (GridSearchCV)
   - Model evaluation (MAE, RMSE, R², MAPE)
   - Model persistence (joblib)

3. **predict_future_aqi.py**
   - Load trained models
   - Generate future predictions (default: 24 hours)
   - Confidence interval calculation
   - Real-time prediction API

**ML Pipeline**:
```
Historical Data → Feature Engineering → Model Training → Evaluation → Deployment
                                                              ↓
                                                    Current Data → Predictions
```

**Design Pattern**: Pipeline Pattern

---

### 3.4 Presentation Layer

**Component**: `dashboard/app.py`

**Framework**: Flask or Streamlit

**Features**:
- Real-time AQI dashboard
- Historical trend analysis
- Prediction visualization
- Interactive maps (Leaflet.js)
- REST API endpoints

**Routes**:
- `/` - Main dashboard
- `/historical` - Historical analytics
- `/predictions` - ML forecasts
- `/map` - Geographic view
- `/api/current` - Current data API
- `/api/predictions` - Predictions API

**Visualization Libraries**:
- Plotly/Chart.js for charts
- Leaflet.js/Folium for maps
- Bootstrap/Material-UI for UI

**Design Pattern**: MVC (Model-View-Controller)

---

## 4. Technology Stack

### 4.1 Core Technologies

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Backend** | Python 3.8+ | Core programming language |
| **Web Framework** | Flask/Streamlit | Dashboard and API |
| **Context Broker** | FIWARE Orion | Context management |
| **Containerization** | Docker | FIWARE deployment |
| **ML Framework** | scikit-learn, XGBoost | Predictive modeling |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Visualization** | Plotly, Matplotlib | Charts and graphs |
| **Mapping** | Leaflet.js, Folium | Geographic visualization |

### 4.2 Python Dependencies

```
flask>=2.0.0 or streamlit>=1.20.0
requests>=2.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
pyyaml>=6.0
plotly>=5.13.0
matplotlib>=3.6.0
folium>=0.14.0
joblib>=1.2.0
```

### 4.3 External Services

- **FIWARE Orion Context Broker**: v3.7.0+
- **Docker**: v20.10+

---

## 5. Design Patterns

### 5.1 Architectural Patterns

1. **Layered Architecture**
   - Clear separation between sensor, context, processing, and presentation layers
   - Each layer depends only on the layer below

2. **Client-Server**
   - Dashboard (client) ↔ FIWARE/API (server)
   - Sensor simulator (client) ↔ Context Broker (server)

3. **Event-Driven**
   - Sensor updates trigger context updates
   - FIWARE subscriptions for real-time notifications

### 5.2 Design Patterns

1. **Adapter Pattern**
   - `orion_client.py` adapts FIWARE API to Python interface

2. **Factory Pattern**
   - Sensor simulator creates sensor instances per location

3. **Strategy Pattern**
   - Different AQI calculation strategies (EPA vs CPCB)
   - Multiple ML algorithms (RandomForest, XGBoost, LSTM)

4. **Repository Pattern**
   - Data access abstraction for FIWARE queries

5. **Pipeline Pattern**
   - ML data processing pipeline
   - Sensor data → Ingestion → Processing → Storage

6. **Observer Pattern**
   - FIWARE subscriptions notify dashboard of changes

---

## 6. Deployment Architecture

### 6.1 Development Environment

```
┌─────────────────────────────────────────────┐
│         Developer Machine (Windows)          │
│  ┌────────────────────────────────────────┐ │
│  │  Python Virtual Environment            │ │
│  │  - Flask/Streamlit app                 │ │
│  │  - Sensor simulator                    │ │
│  │  - ML training scripts                 │ │
│  └────────────────────────────────────────┘ │
│                     ↕                        │
│  ┌────────────────────────────────────────┐ │
│  │  Docker Container                      │ │
│  │  - FIWARE Orion (port 1026)            │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### 6.2 Production Deployment (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Dashboard    │  │  Dashboard    │  │  Dashboard    │
│  Instance 1   │  │  Instance 2   │  │  Instance N   │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────┐
        │   FIWARE Orion Cluster              │
        │   (with MongoDB persistence)        │
        └─────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────┐
        │   Sensor Simulator Cluster          │
        │   (distributed across locations)    │
        └─────────────────────────────────────┘
```

### 6.3 Container Orchestration (Kubernetes)

```yaml
# Deployment components:
- FIWARE Orion StatefulSet (3 replicas)
- MongoDB StatefulSet (3 replicas)
- Dashboard Deployment (auto-scaling)
- Sensor Simulator DaemonSet
- ML Training CronJob
```

---

## 7. Security Architecture

### 7.1 Authentication & Authorization

- **FIWARE**: Multi-tenancy via `Fiware-Service` headers
- **Dashboard**: User authentication (OAuth 2.0 recommended)
- **API**: Token-based authentication (JWT)

### 7.2 Data Security

- **In-Transit**: HTTPS/TLS for all communications
- **At-Rest**: Encrypted storage for sensitive data
- **API Keys**: Secure storage in environment variables

### 7.3 Network Security

```
┌─────────────────────────────────────────┐
│         Firewall / Security Group        │
│  - Port 443 (HTTPS) - Public            │
│  - Port 1026 (Orion) - Internal only    │
│  - Port 27017 (MongoDB) - Internal only │
└─────────────────────────────────────────┘
```

---

## 8. Scalability & Performance

### 8.1 Horizontal Scaling

**Scalable Components**:
- Dashboard instances (stateless)
- Sensor simulators (distributed)
- FIWARE Orion (cluster mode)

**Non-Scalable Components**:
- ML training (batch process)

### 8.2 Performance Optimizations

1. **Caching**:
   - Redis cache for frequent queries
   - Dashboard data caching (30s TTL)

2. **Database Indexing**:
   - MongoDB indices on entity_id, type, timestamp

3. **Batch Processing**:
   - Bulk sensor data ingestion
   - Batch predictions

4. **Asynchronous Processing**:
   - Background ML training
   - Async API calls

### 8.3 Monitoring & Observability

**Metrics**:
- Sensor data ingestion rate
- API response times
- ML prediction accuracy
- FIWARE entity count

**Logging**:
- Structured logging (JSON format)
- Centralized log aggregation (ELK stack)

**Alerting**:
- Sensor data anomalies
- API downtime
- ML model drift

---

## 9. Future Enhancements

### 9.1 Planned Features

1. **Real Sensor Integration**
   - Replace simulator with actual IoT sensors
   - LoRaWAN/MQTT protocol support

2. **Advanced ML Models**
   - Deep learning (LSTM, Transformer)
   - Ensemble methods
   - AutoML for hyperparameter optimization

3. **Mobile Application**
   - React Native mobile app
   - Push notifications for alerts

4. **Data Analytics**
   - Historical trend analysis
   - Comparative city analysis
   - Pollution source identification

### 9.2 Scalability Roadmap

- Kubernetes deployment
- Multi-region support
- Edge computing for sensor processing
- Real-time streaming (Apache Kafka)

---

## 10. References

- [FIWARE Orion Documentation](https://fiware-orion.readthedocs.io/)
- [NGSI-v2 Specification](https://fiware.github.io/specifications/ngsiv2/stable/)
- [EPA AQI Guidelines](https://www.airnow.gov/aqi/)
- [CPCB AQI Standards](https://cpcb.nic.in/)

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Maintained By**: Digital Twin AQI Kochi Team
