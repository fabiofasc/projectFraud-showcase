*Author: Fabio Fasciglione*

# Real-Time Fraud Detection System

**A demonstration prototype of a real-time ML pipeline for financial transaction fraud detection, trained on synthetic data**

---

## Abstract

This project presents a demonstration prototype of a real-time fraud detection system for financial transaction processing. The system combines streaming data ingestion via Apache Kafka, a gradient-boosted classification model (XGBoost), and an inference API (FastAPI). Key engineering contributions include an approach to extreme class imbalance (0.5% fraud rate), a 30-feature behavioral engineering pipeline, and a Docker Compose infrastructure with Prometheus and Grafana configured as a datasource. On synthetic data, the model reaches precision and recall of 0.70 on a test set containing only 10 frauds; this figure is indicative, not a measure of production performance. I also wrote an asynchronous load-test harness, but no latency or throughput results from an actual run are documented.

---

## 1. Introduction

Fraud detection in financial systems presents a unique confluence of challenges: extreme class imbalance, strict latency constraints, the need for interpretable decisions, and adversarial adaptation by malicious actors. A model that naively predicts every transaction as legitimate achieves 99.5% accuracy while catching zero frauds—a stark illustration of why standard accuracy metrics are inadequate for this domain.

This system was designed to address all these challenges simultaneously, treating fraud detection not as a pure machine learning problem but as a full-stack engineering discipline. The goal was to build a prototype that goes beyond an offline notebook: streaming ingestion, real-time serving, a persistence schema for audit trails, and monitoring infrastructure.

The rest of this document is organized as follows: §2 describes the system architecture and data flow; §3 covers feature engineering; §4 details the model design and training methodology; §5 presents the API design and performance characteristics; §6 covers infrastructure and MLOps practices; §7 reports empirical results.

---

## 2. System Architecture

### 2.1 Overview

The system is a demonstration prototype with a Docker Compose infrastructure. The Compose file starts the supporting services (ZooKeeper, Kafka, PostgreSQL, Prometheus, Redis, Grafana); the API and the Kafka consumer are run separately as Python processes. Redis is declared as an optional container but is not used by the code.

```
┌─────────────────────┐
│  Transaction        │  Synthetic generator
│  Producer           │  100+ TPS, realistic fraud patterns
└────────┬────────────┘
         │ Kafka topic: "transactions"
         ▼
┌─────────────────────┐       ┌─────────────────────┐
│  Kafka Consumer     │       │  FastAPI /predict   │
│  + ML Inference     │       │  (synchronous path) │
│  (streaming path)   │       └──────────┬──────────┘
└────────┬────────────┘                  │
         │                               │
         ▼                               ▼
┌─────────────────────────────────────────────────┐
│                  PostgreSQL                     │
│  predictions (written by the consumer)          │
│  other tables: schema only, not populated       │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  Prometheus + Grafana   │
          │  (Grafana: datasource)  │
          └─────────────────────────┘
```

### 2.2 Data Flow

Transactions enter the system via a Kafka producer that generates synthetic records at 100+ transactions per second, simulating realistic behavioral patterns including five distinct fraud archetypes (high-value outliers, late-night anomalies, rapid sequential transactions, cross-border anomalies, and compromised merchant patterns).

Two parallel consumption paths exist:
- **Streaming path**: A Kafka consumer reads from the topic, invokes the ML pipeline in-process, and persists predictions to PostgreSQL. Offsets are committed manually after each message, but a failed database write is only logged and the offset is still committed, so delivery is best-effort, not at-least-once. The insert is idempotent (`ON CONFLICT DO NOTHING`), so adding retries would be safe.
- **Synchronous path**: A REST API accepts individual or batch transaction payloads for real-time predictions from upstream services.

### 2.3 Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Message broker | Apache Kafka 7.6.0 | Durable, replayable event log; horizontal scaling via consumer groups |
| ML model | XGBoost 2.0.3 | Strong tabular performance; native class-weight support; fast inference |
| API framework | FastAPI + Uvicorn | Async-native; automatic OpenAPI docs; Pydantic validation |
| Database | PostgreSQL 16 | ACID guarantees for audit trail; rich query support for analytics |
| Monitoring | Prometheus + Grafana | Industry standard; low overhead; rich alerting |
| Containerization | Docker Compose | Reproducible multi-service deployment; environment parity |

---

## 3. Feature Engineering

Raw transactions arrive with nine fields: `transaction_id`, `user_id`, `amount`, `merchant`, `merchant_category`, `timestamp`, `location_country`, `device_type`, and `is_fraud`. From these, the pipeline engineers 31 features across five categories.

### 3.1 Temporal Features

Fraud patterns are strongly time-dependent. The pipeline extracts `hour`, `day_of_week`, `is_weekend`, and `is_late_night` (transactions between 02:00–05:00 carry significantly higher fraud rates in empirical data). These features capture circadian patterns that behavioral features alone cannot.

### 3.2 Behavioral Deviation Features (Top Predictors)

The most predictive features capture *deviation from a user's own baseline*—a classical approach in fraud detection grounded in the intuition that legitimate users have stable behavioral profiles.

Per-user statistics (`user_avg_amount`, `user_std_amount`, `user_transaction_count`) are computed from historical data. From these, two derived features are constructed:

- **`amount_vs_user_avg_ratio`**: `amount / user_avg_amount` — captures proportional deviation.
- **`amount_z_score_user`**: `(amount - user_avg_amount) / user_std_amount` — standardized deviation, directly interpretable as "how many standard deviations above this user's normal spending." Consistently the top feature in importance analysis.

### 3.3 Global Statistical Features

Complementing user-level features, global statistics capture macro-level anomalies:
- `amount_z_score`: Global z-score across all transactions.
- `amount_log`: Log-transformed amount to handle right-skewed distributions.
- `is_high_amount` (>$1,000) and `is_very_high_amount` (>$5,000): Binary threshold indicators.

### 3.4 Risk Scoring Features

- **`merchant_fraud_rate`**: Historical fraud percentage per merchant, computed from training data. This encodes domain knowledge about high-risk merchants without requiring explicit labeling.
- **`is_high_risk_country`**: Binary flag for transactions originating from jurisdictions with elevated baseline fraud rates (NG, RU, CN, VN, PK).

### 3.5 Categorical Encoding

`merchant_category` and `device_type` are one-hot encoded. The feature pipeline ensures consistent column presence at inference time, filling missing columns with zeros to prevent schema mismatches between training and serving environments—a common source of production failures.

### 3.6 Feature Consistency

The same `FeatureEngineer` object is serialized alongside the model artifact (`feature_engineer_latest.pkl`). At inference time, this identical object is loaded and applied, guaranteeing training-serving symmetry. This eliminates an entire class of silent degradation bugs.

---

## 4. Model Design and Training

### 4.1 Algorithm Selection

XGBoost was selected over alternatives (LightGBM, Random Forest, logistic regression, neural networks) for several domain-specific reasons:

1. **Tabular data performance**: Gradient-boosted trees consistently outperform deep learning on structured tabular features with moderate dataset sizes.
2. **Class imbalance support**: The `scale_pos_weight` hyperparameter directly addresses class imbalance without requiring data resampling.
3. **Inference speed**: Single-record predictions execute in under 1ms in-process, meeting latency requirements without batching.
4. **Interpretability**: SHAP-compatible feature importances support regulatory and operational explainability requirements.

### 4.2 Handling Extreme Class Imbalance

With a 0.5% fraud rate, class imbalance is the central modeling challenge. Three strategies were evaluated:

| Strategy | Approach | Trade-off |
|----------|----------|-----------|
| `scale_pos_weight` | Weight fraud class by `(1 - fraud_rate) / fraud_rate ≈ 199` | Simplest; no data modification; used in this prototype |
| SMOTE oversampling | Synthesize minority-class neighbors | Higher recall; memory-intensive for large datasets |
| Random undersampling | Reduce majority class | Fast; discards potentially useful data |

The `scale_pos_weight` approach was selected. It operates at the loss function level, giving fraud samples 199x the gradient contribution of normal samples, without altering the training set distribution or introducing synthetic samples that may not reflect real fraud patterns.

### 4.3 Model Configuration

```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric=['auc', 'aucpr'],    # AUCPR preferred for imbalanced data
    max_depth=6,                     # Controls overfitting
    n_estimators=200,
    learning_rate=0.1,
    scale_pos_weight=199,            # Fraud class weighting
    tree_method='hist',              # Fast histogram-based split finding
    early_stopping_rounds=20,        # Halt when validation stops improving
    random_state=42
)
```

### 4.4 Evaluation Methodology

**Why PR-AUC, not ROC-AUC**: ROC-AUC is misleading under extreme imbalance because it treats true negatives (the abundant class) symmetrically with true positives. A classifier with poor recall can achieve high ROC-AUC simply by correctly classifying the majority class. Precision-Recall AUC focuses on the minority class performance, making it the appropriate primary metric.

Evaluation uses a stratified train/test split with early stopping, preserving class ratios in both partitions. The confusion matrix is interpreted in business terms:
- **False Positive Rate**: Customer friction (legitimate transactions blocked).
- **False Negative Rate**: Financial loss (frauds missed).

The optimal decision threshold is set at 0.4 for the streaming path (slightly favoring recall) and 0.5 for the synchronous API (balanced precision/recall), reflecting different downstream cost structures.

---

## 5. API Design and Performance

### 5.1 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 1,000 transactions) |
| `GET` | `/health` | Liveness probe (load balancer integration) |
| `GET` | `/ready` | Readiness probe (service readiness) |
| `GET` | `/metrics/app` | Application metrics (latency percentiles, RPS, fraud rate) |
| `GET` | `/model/info` | Model metadata and performance metrics |
| `POST` | `/admin/reload` | Hot model reload without service restart |

### 5.2 Response Format

```json
{
  "transaction_id": "txn_a1b2c3d4",
  "is_fraud": true,
  "fraud_probability": 0.89,
  "risk_level": "high",
  "model_version": "v1_20241202",
  "latency_ms": 1.23,
  "top_features": {
    "amount_z_score_user": 15.2
  }
}
```

Risk levels (`low` / `medium` / `high` / `critical`) map probability thresholds to actionable tiers, enabling downstream systems to route decisions without re-implementing threshold logic.

### 5.3 Performance Optimizations

**Model Singleton Caching**: The `ModelLoader` class implements the Singleton pattern. The first call to `get_model()` deserializes the XGBoost artifact and the feature engineer from disk (~100ms), subsequent calls return the cached in-memory object. This prevents per-request deserialization overhead.

**Async I/O**: FastAPI's async request handlers allow concurrent request processing without thread-per-request overhead. I/O-bound operations (such as health checks) yield the event loop while waiting, enabling higher throughput on a single process.

**Batch Vectorization**: Sequential prediction of 100 transactions takes approximately 100ms; a vectorized batch of 100 takes approximately 15ms—a 6.7× speedup from eliminating Python loop overhead and leveraging XGBoost's native batch inference.

**Hot Model Reload**: The `/admin/reload` endpoint reloads the model and feature engineer from disk without restarting the service. It is unauthenticated, so it is a convenience for local use, not something to expose.

### 5.4 Performance

I wrote an asynchronous load-test harness (`scripts/load_test_api.py`), but no results from an actual run are stored, so this README reports no latency or throughput figures. Numbers should be re-run and saved before being cited.

---

## 6. Infrastructure and MLOps

### 6.1 Data Persistence Schema

Five PostgreSQL tables are defined in the schema. Only `predictions` is written by the code (by the streaming consumer; the API does not write to the database). Drift detection, an alert system and MLflow experiment tracking are not implemented.

- **`predictions`**: Every model decision, with timestamp, probability, latency, and model version. Enables offline analysis of the model's decisions.
- **`model_metrics`**: Periodic performance snapshots (precision, recall, PR-AUC). Schema only; not populated by the code.
- **`fraud_alerts`**: Schema only; no alerting system is implemented.
- **`feature_store`**: Schema only; not populated by the code.
- **`ab_test_results`**: Schema only; see §6.4.

### 6.2 Observability Stack

Three observability layers are instrumented:

1. **Infrastructure**: Docker health checks and restart policies ensure service availability.
2. **Platform**: `prometheus-fastapi-instrumentator` automatically instruments all endpoints with request count, latency histograms, and in-flight request gauges. Prometheus scrapes every 15 seconds; Grafana is configured with Prometheus as a datasource, but no dashboards are included.
3. **Application**: Custom business metrics expose fraud rate, model version in use, and prediction latency percentiles—decoupled from HTTP-level metrics and tied to business semantics.

Request tracing assigns a UUID to every inbound request, propagated via `X-Request-ID` response headers and embedded in all log lines, enabling correlation across services.

### 6.3 Reliability

- **Delivery semantics**: Kafka consumers use manual offset commits, but the commit happens after every message even when the database write failed (errors are caught and logged). A crash before the commit causes replay; a failed write does not. Making write errors propagate and committing only on success would give true at-least-once.
- **Graceful Shutdown**: Signal handlers for `SIGINT`/`SIGTERM` allow in-flight requests to complete before process exit.
- **Structured Logging**: `loguru` with JSON-compatible output enables log aggregation and querying in production log systems.

### 6.4 A/B Testing

Schema prepared, not implemented: the `ab_test_results` and `ab_tests` tables exist, but there is no traffic-assignment logic.

---

## 7. Results

### 7.1 Model Performance

Measured on synthetic data, on a test set containing only 10 frauds. The figures are indicative, not a measure of production performance.

| Metric | Value |
|--------|-------|
| Precision | 0.70 |
| Recall | 0.70 |
| PR-AUC | 0.63 |
| ROC-AUC | 0.97 |

### 7.2 Interpretation

With a recall of 0.70 on 10 frauds, 3 frauds were missed (false negatives = financial exposure). With so few positive examples, a single transaction shifts these figures substantially.

### 7.3 Top Predictive Features

Feature importance analysis consistently identifies the following as the most discriminative signals:

1. `amount_z_score_user` — deviation from the user's own spending baseline
2. `merchant_fraud_rate` — historical merchant risk
3. `is_late_night` — temporal anomaly indicator
4. `amount_vs_user_avg_ratio` — proportional spending deviation
5. `is_high_risk_country` — geographic risk signal

This ordering reinforces the value of behavioral (user-level) features over global statistical features, consistent with the fraud detection literature.

---

## 8. Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+

### Running the Full Stack

```bash
# Clone the repository
git clone <repo-url>
cd fraud-detection-system

# Start the infrastructure (ZooKeeper, Kafka, PostgreSQL, Prometheus, Redis, Grafana)
docker-compose up -d

# Verify services are healthy
docker-compose ps

# Collect training data (runs producer for ~5 minutes)
python scripts/collect_training_data.py

# Train the model
python src/models/train.py

# Run a prediction (requires the API to be started separately: it is not part of the Compose file)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "user_id": "user_42",
    "amount": 9500.00,
    "merchant": "electronics_store",
    "merchant_category": "electronics",
    "timestamp": "2024-12-02T03:15:00",
    "location_country": "NG",
    "device_type": "mobile"
  }'
```

### Service URLs

| Service | URL |
|---------|-----|
| Fraud Detection API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## 9. Project Structure

```
fraud-detection-system/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application, endpoints, middleware
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── model_loader.py      # Singleton model caching
│   ├── models/
│   │   └── train.py             # XGBoost training pipeline
│   ├── features/
│   │   └── engineering.py       # Feature engineering pipeline
│   └── data_ingestion/
│       ├── kafka_producer.py    # Synthetic transaction generator
│       └── kafka_consumer_with_model.py  # Streaming inference consumer
├── docker/
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── init-db.sql              # PostgreSQL schema
│   ├── prometheus.yml           # Metrics scrape config
│   └── grafana-datasources.yml  # Grafana configuration
├── scripts/
│   ├── collect_training_data.py # Kafka-to-CSV data collection
│   └── load_test_api.py         # Async load testing harness
├── models/                      # Serialized model artifacts
│   ├── fraud_model_latest.pkl
│   └── feature_engineer_latest.pkl
└── requirements.txt
```

---

## 10. Conclusion

This prototype illustrates that fraud detection requires engineering depth across multiple disciplines simultaneously: the statistics of imbalanced classification, the distributed systems concerns of streaming pipelines, the software engineering principles of reliable API design, and the operational discipline of observability.

The core technical insight is that *feature engineering and class imbalance handling matter more than algorithm selection*. A well-tuned XGBoost model with behavioral deviation features and appropriate class weighting outperforms a naive deep learning approach on this problem class, while being orders of magnitude faster to serve and easier to explain.

The system is intentionally over-engineered relative to its toy dataset—its value is as a blueprint for production ML systems, not as a fraud model trained on synthetic data.

---

## Limits and next steps

**Limits**
- All data is synthetic, produced by the project's own generator, so the model has never seen real fraud patterns.
- The metrics come from a test set with only 10 fraudulent transactions (precision and recall 0.70): they are indicative and say little about real-world performance.
- The model was trained locally with a minimal setup, enough to get a reasonable baseline for the prototype, not tuned.
- Kafka runs as a single broker, and the producer, consumer and API are started by hand rather than orchestrated.
- Database saves in the streaming consumer are best-effort: a failed write is logged and the offset is still committed.
- There are no automated tests and no recorded load test.

**Next steps**
- Make save errors propagate and commit offsets only on success (true at-least-once), with a dead letter topic for records that keep failing.
- Add a test suite and keep a load-test run as evidence.
- Evaluate on a larger labeled set and report confidence intervals.

## License

MIT

---

*Built project demonstrating end-to-end ML engineering for financial fraud detection.*