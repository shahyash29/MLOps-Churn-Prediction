# MLOps Churn Prediction

A complete MLOps pipeline for customer churn prediction using the Telco dataset. This project demonstrates end-to-end ML lifecycle management with MLflow, containerized deployment, monitoring, and data drift detection.

## Architecture

```
├── data/
│   ├── raw/              # Original telco.csv dataset
│   └── processed/        # Cleaned train/test splits
├── features/             # Data preprocessing & feature engineering
├── training/             # ML model training scripts
├── serving/              # FastAPI prediction service
├── monitor/              # Data drift monitoring
├── artifacts/            # Model artifacts & monitoring reports
└── mlruns/              # MLflow experiment tracking
```

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose

### 1. Setup Environment
```bash
git clone <repo-url>
cd mlops-churn
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python -m features.build
```

### 3. Train Models
```bash
python training/train_mlflow.py
```

### 4. Start Services
```bash
docker-compose up -d
```

### 5. Access Services
- **MLflow UI**: http://localhost:5000
- **Prediction API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## Models

The pipeline supports multiple algorithms:

- **Logistic Regression**: Baseline model with class balancing
- **XGBoost**: Gradient boosting with hyperparameter tuning

Models are automatically tracked in MLflow with:
- Metrics: Accuracy, ROC-AUC
- Artifacts: Model files, plots, classification reports
- Versioning: Automatic model registration

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:8001/health
```

### Prediction
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "SeniorCitizen": 0,
      "tenure": 12,
      "MonthlyCharges": 50.0,
      "TotalCharges": 600.0,
      "Contract": "Month-to-month",
      "PaymentMethod": "Electronic check"
    }
  }'
```

### Metrics (Prometheus)
```bash
curl http://localhost:8001/metrics
```

## Monitoring

### Data Drift Detection
```bash
python monitor/drift_report.py
```

The system automatically:
- Logs all predictions to `logs/inference/events.jsonl`
- Monitors feature distributions using Evidently
- Generates HTML drift reports in `artifacts/monitoring/`

### Key Monitoring Features
- Real-time prediction logging
- Statistical drift detection
- Visual drift reports
- Prometheus metrics integration

## 🐳 Docker Services

### MLflow Server
- **Image**: `ghcr.io/mlflow/mlflow:latest`
- **Port**: 5000
- **Storage**: SQLite backend with file artifact store

### Training Service
- **Build**: `Dockerfile.train`
- **Purpose**: Model training with MLflow tracking
- **Volumes**: Data and artifacts mounted

### API Service
- **Build**: `Dockerfile.api`
- **Port**: 8001
- **Purpose**: Real-time prediction serving
- **Environment**: Model staging configuration

## 🔄 ML Pipeline

1. **Data Ingestion**: Raw telco dataset processing
2. **Feature Engineering**: Numeric coercion, missing value handling
3. **Model Training**: Multiple algorithms with hyperparameter tracking
4. **Model Registration**: Automatic versioning in MLflow
5. **Model Serving**: REST API with health checks
6. **Monitoring**: Drift detection and performance tracking

## Key Files

- `features/build.py`: Data preprocessing and train/test splitting
- `training/train_mlflow.py`: MLflow-integrated model training
- `serving/app.py`: FastAPI prediction service
- `monitor/drift_report.py`: Evidently-based drift monitoring
- `docker-compose.yml`: Multi-service orchestration

## Dependencies

### Core ML Stack
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms and preprocessing
- **xgboost**: Gradient boosting
- **mlflow**: Experiment tracking and model registry

### API & Monitoring
- **FastAPI**: REST API framework
- **evidently**: Data drift detection
- **prometheus-client**: Metrics collection

### Deployment
- **Docker**: Containerization
- **gunicorn**: Production WSGI server

## Features

- End-to-end MLops pipeline
- Automated experiment tracking
- Model versioning and registry
- Containerized deployment
- Real-time prediction API
- Data drift monitoring
- Prometheus metrics
- Health checks and logging

## Next Steps

- [ ] Add CI/CD pipeline
- [ ] Implement A/B testing
- [ ] Add model performance monitoring
- [ ] Integrate with cloud platforms
- [ ] Add automated retraining triggers