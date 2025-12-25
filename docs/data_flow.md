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



---

## 3. Detailed Data Flow Diagrams

### 3.1 Sensor Data Ingestion Flow



### 3.2 AQI Calculation Flow



### 3.3 Machine Learning Pipeline Flow



### 3.4 Dashboard Data Flow



---

## 4. Data Formats & Transformations

### 4.1 Sensor Data Format (Raw)



### 4.2 NGSI-v2 Entity Format (FIWARE)



### 4.3 ML Feature Matrix Format



### 4.4 Prediction Output Format



### 4.5 Dashboard API Response Format



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
