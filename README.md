# Eksperimen MSML - Breast Cancer Classification

Repository ini berisi submission proyek akhir Membangun Sistem Machine Learning untuk klasifikasi Breast Cancer Wisconsin.

## Struktur Project

```text
.
├── breast_cancer_raw/
│   └── breast_cancer_raw.csv
├── breast_cancer_preprocessing/
│   ├── train_preprocessed.csv
│   └── test_preprocessed.csv
├── preprocessing/
│   ├── Eksperimen_Jaelani_Surya_Saputra.ipynb
│   ├── automate_Jaelani_Surya_Saputra.py
│   └── breast_cancer_preprocessing/
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   └── artifacts/
├── Monitoring dan Logging/
│   ├── 2.prometheus.yml
│   ├── 3.prometheus_exporter.py
│   └── 7.inference.py
├── Eksperimen_SML_Jaelani-Surya-Saputra.txt
└── Workflow-CI.txt
```

## Cara Menjalankan

Install dependency:

```bash
pip install -r Membangun_model/requirements.txt
```

Jalankan preprocessing:

```bash
python preprocessing/automate_Jaelani_Surya_Saputra.py
```

Training model baseline:

```bash
python Membangun_model/modelling.py
```

Training dengan hyperparameter tuning:

```bash
python Membangun_model/modelling_tuning.py
```

Inference:

```bash
python "Monitoring dan Logging/7.inference.py" --rows 5
```

Exporter Prometheus:

```bash
python "Monitoring dan Logging/3.prometheus_exporter.py"
```

Endpoint metrics tersedia di:

```text
http://localhost:8000/metrics
```

## Hasil Model

Baseline Random Forest menghasilkan metrik berikut pada data test:

```text
accuracy : 0.9474
precision: 0.9583
recall   : 0.9583
f1_score : 0.9583
roc_auc  : 0.9937
```

## Catatan Submission

File screenshot di folder `Membangun_model` dan `Monitoring dan Logging` perlu diganti dengan screenshot aktual dari MLflow, Prometheus, dan Grafana sebelum dikumpulkan.
