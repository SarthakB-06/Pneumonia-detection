# Clinical Pneumonia Detection AI & MLOps Pipeline

An end-to-end, production-grade Deep Learning and MLOps pipeline designed to analyze chest X-ray images and classify them as **NORMAL** or **PNEUMONIA**. 

This repository tracks the entire lifecycle of a medical AI system: from raw data exploration and addressing severe class imbalances in Jupyter Notebooks, to training a high-recall transfer learning model (ResNet50V2), wrapping it in a high-performance **FastAPI** backend, building a polished **Streamlit** user interface, containerizing with **Docker Compose**, and orchestrating deployment via **Kubernetes**.

---

## 📊 Core Performance Metrics (Test Set)

In medical diagnostics, missing a sick patient (False Negative) is catastrophic. By optimizing our training loop for clinical utility, the model achieves the following metrics on completely unseen test data:

| Metric | Score | Clinical Significance |
| :--- | :--- | :--- |
| **Test Recall** | **93.85%** | **Primary Metric:** Out of 100 actual pneumonia cases, the AI successfully catches ~94 of them. |
| **Test Precision** | **88.83%** | When the AI flags an X-ray as Pneumonia, it is correct 88.8% of the time. |
| **Test Accuracy** | **88.78%** | Overall correct classification rate across healthy and diseased lungs. |
| **Test Loss** | **0.2889** | Binary Crossentropy loss indicating highly stable probability calibrations. |

---

## 🏗️ System Architecture & Features

1. **Mathematical Imbalance Correction:** The dataset exhibits a severe 3:1 majority-to-minority class imbalance (3,876 Pneumonia vs 1,342 Normal images). Instead of duplicating data or using destructive synthetic expansions, the pipeline programmatically computes dynamic class weights (Normal: 1.94, Pneumonia: 0.67) to penalize minority class errors heavily during backpropagation.
2. **Idempotent Data Engineering:** Includes a custom validation split engine that safely reallocates 10% of training data into a statistically significant validation set while maintaining strict directory isolation and structural idempotency.
3. **Inference Parity Enforcement:** Ensures the exact dynamic data pipelines used during training (224x224 resizing, $[0, 1]$ float scale normalization) are mirrored step-for-step inside the live web server, preventing skew during real-world inference.
4. **MLOps Compatibility Layer:** Integrates environment overrides (`TF_USE_LEGACY_KERAS=1` and `tf_keras`) to bridge Keras 2/3 functional serialization architectures inside production container environments.

---

## 📁 Repository Structure

```text
pneumonia-detection-ops/
├── api/
│   ├── Dockerfile             # Multi-stage production build for FastAPI server
│   └── main.py                # FastAPI engine handling image serialization & predictions
├── frontend/
│   ├── Dockerfile             # Production build for interactive client app
│   └── app.py                 # Streamlit web UI communicating with internal DNS
├── k8s/
│   ├── api-deployment.yaml    # Kubernetes Deployment & internal Headless Service for backend
│   └── frontend-deployment.yaml # Kubernetes Deployment & External LoadBalancer for frontend
├── notebooks/
│   └── 01_data_exploration.ipynb # EDA, data auditing, and validation split prototyping
├── saved_models/
│   └── best_pneumonia_model.keras # Highly optimized, serialized model weights
├── src/
│   ├── config.py              # Centralized infrastructure and hyperparameter configurations
│   ├── data_loader.py         # Real-time data pipeline with dynamic image augmentations
│   ├── model.py               # Fine-tuned ResNet50V2 model definition
│   └── train.py               # Main model training loop with automated callbacks
├── docker-compose.yml         # Local orchestration multi-container blueprint
└── requirements.txt           # Unified, version-locked dependency matrix

```
🚀 Deployment Instructions
1. Local Monolithic Execution
To configure your local environment, install dependencies, run data allocation, and start training:

# Install version-locked dependencies
pip install -r requirements.txt

# Run the data engineering script to rebalance the train/val directories
python src/setup_data.py

# Launch training loop (EarlyStopping and ModelCheckpoint auto-engaged)
python src/train.py
```



```
2. Multi-Container Orchestration (Docker Compose)
To run the full stack (FastAPI Backend + Streamlit Frontend) locally within an isolated container network:
# Stop any local instances utilizing ports 8000 or 8501, then execute:
docker-compose up --build

Once initialized, access the interactive portal at http://localhost:8501 and the automated API documentation at http://localhost:8000/docs.
```

```
3. Production Cloud Orchestration (Kubernetes)
To deploy the version-locked containers to a scalable Kubernetes cluster using internal DNS lookups for microservice routing:

# Ensure Kubernetes is enabled in Docker Desktop (or your cloud context)
# Apply all manifest blueprints inside the k8s directory
kubectl apply -f k8s/

# Monitor deployment progress until all Pods change to 'Running'
kubectl get pods -w

# Discover the external port mapping for the service
kubectl get services

Navigate to http://localhost:8501 to test the system running entirely under enterprise orchestration.
```



