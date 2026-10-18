# Defect Detection Pipeline for Operational Monitoring

## Problem Overview

In industrial and operational environments, visual inspection is a critical step for ensuring product quality and system reliability. Components or surfaces may develop defects such as scratches, cracks, deformations, or contaminations that indicate abnormal operating conditions or manufacturing faults.

Traditionally, defect detection relies on manual visual inspection, which is time-consuming, subjective, and difficult to scale. As production volumes increase and quality requirements become stricter, automated image-based inspection systems are increasingly necessary.

This project addresses the problem of **automated defect detection from images**, framing it as a **binary classification task**:

* **Normal**: Image corresponds to an expected, non-defective operational state.
* **Defective**: Image contains visual anomalies indicating a deviation from normal conditions.

The objective is to design an end-to-end machine learning pipeline that supports reliable operational monitoring. The system prioritizes detecting defective items, accepting some false alarms to reduce the risk of undetected faults.

---

## Operational Considerations

* **False positives** (normal classified as defective) may lead to unnecessary inspections, increased costs, and reduced trust in the system.
* **False negatives** (defective classified as normal) may allow faulty components to pass undetected, potentially causing safety risks, system failures, or downstream losses.

To reflect these constraints, the system is evaluated using Recall at a fixed False Positive Rate (FPR ≤ 5%).
This metric directly captures the operational requirement of maximizing defect detection (high recall) while explicitly constraining unnecessary inspections to an acceptable level.

Rather than optimizing global metrics that treat all errors symmetrically, this evaluation strategy aligns the model’s performance with real-world decision thresholds used in industrial inspection pipelines.

For many real-world decisions (e.g., accept/reject, trigger alert/continue operation), a binary outcome is sufficient and often preferred. The system is therefore designed with deployment and operational decision-making in mind.

---

## Dataset Description

This project uses a subset of the **MVTec Anomaly Detection (MVTec AD)** dataset.

MVTec AD is a widely used industrial inspection dataset containing high-resolution images of manufactured objects and surfaces under controlled conditions. It includes:

* Multiple object and texture categories
* Clean *normal* samples
* Diverse real-world defect types such as cracks, scratches, contaminations, and structural anomalies

Although originally designed for anomaly detection, the dataset is adapted here to a **binary classification setting** by grouping all anomalous samples under a single *defective* label. This mirrors real operational scenarios where the primary goal is to detect whether intervention is required, rather than to classify the specific defect type.

**Reasons for choosing MVTec AD:**

* Realistic industrial imagery
* High relevance to operational monitoring tasks
* Commonly used as a benchmark in both academic and applied research
* Avoids toy or synthetic datasets unsuitable for real-world inspection problems

Details about data preprocessing, splits, and category selection are documented in `data/README.md`.

---

## High-Level Pipeline

The system follows a modular, end-to-end machine learning workflow:

```
        Raw Images
             │
             ▼
    Data Ingestion & Validation
             │
             ▼
        Preprocessing
             │
             ▼
  Feature Extraction (frozen CNN)
             │
             ▼
   Anomaly Scoring + Calibrated
   Threshold (Normal vs Defective)
             │
             ▼
        Model Evaluation
   (Recall @ FPR ≤ 5%, AUROC)
```

This structure keeps data handling, scoring, calibration and evaluation separate, so each stage can be tested and swapped independently.

---

## Modeling Approach

A **ResNet18 pretrained on ImageNet** is used as a **frozen feature extractor**: the classification head is removed and each image is mapped to a **512-dimensional embedding**. No network weights are trained.

A normality model is fitted on embeddings of **normal training images only**, one model per category. Three scoring methods are compared on the same embeddings (higher score = more anomalous):

| Method | Fit on normal embeddings | Anomaly score |
|---|---|---|
| `l2_prototype` | Mean embedding (prototype) | L2 distance to the prototype |
| `mahalanobis` | Mean and covariance (+ ε·I ridge) | Squared Mahalanobis distance |
| `knn` | Store all embeddings | Mean L2 distance to the k nearest |

---

## Evaluation Protocol

The data is split so that calibration never needs defective images, matching deployments where defects are rare or unavailable when the system is set up:

| Split | Contents | Used for |
|---|---|---|
| train | normal only (80% of MVTec `train/good`) | Fit the normality model |
| val | normal only (20% of MVTec `train/good`) | Calibrate the decision threshold |
| test | MVTec `test/` (normal + defective) | Report performance |

1. **Calibration.** The threshold is the 95th percentile of validation scores, so at most 5% of normal images are flagged. False positive rate depends only on normal images, so no defects are needed for this step.
2. **Evaluation.** On the test split, at that fixed threshold: **recall** (defects caught), **realized FPR** and precision. Threshold-free **AUROC** and **AUPR** are also reported.
3. **Oracle reference.** The best recall achievable on test with FPR ≤ 5% if the threshold were chosen with test labels. This is not deployable; the gap to the calibrated recall measures the cost of calibrating on normal data only.

---

## Quickstart

```bash
pip install -r requirements.txt          # or requirements-dev.txt for tests and linting

# 1. Download MVTec AD and extract it to data/raw/mvtec_ad/ (see data/README.md)
# 2. Build the dataset index
python data/index_dataset.py

# 3. Fit, calibrate and evaluate every category × scoring method in configs/train.yaml
python -m src.training.run --config configs/train.yaml

# Tests (no dataset needed)
pytest
```

Outputs go to `outputs/`: `results.csv` (one row per category × method), per-image test scores, and the fitted model and threshold for each pair.

---

## Project Structure

```
defect-detection-pipeline/
├── configs/            # train.yaml (pipeline), data.yaml (dataset indexing)
├── data/               # index_dataset.py and dataset documentation
├── scripts/            # data sanity checks
├── src/
│   ├── data/           # dataset and transforms
│   ├── models/         # frozen ResNet18 feature extractor
│   ├── scoring/        # l2_prototype, mahalanobis, knn scorers
│   ├── evaluation/     # calibration, metrics, plots
│   └── training/       # run.py: fit → calibrate → evaluate
├── tests/              # unit tests and an end-to-end smoke test
├── requirements.txt
└── README.md
```

---

## Project Goals

* Build a realistic image-based defect detection pipeline
* Emphasize operational decision-making constraints
* Demonstrate clean structure and reproducibility
* Serve as a reference implementation for industrial visual inspection systems

---

## References & Sources

The design choices and dataset selection in this project are informed by the following sources:

1. **MVTec Anomaly Detection Dataset**
   Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C.
   *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*
   Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

2. Industrial visual inspection and anomaly detection surveys:

   * Chandola, V., Banerjee, A., & Kumar, V. (2009).
     *Anomaly detection: A survey*. ACM Computing Surveys.
   * Pimentel et al. (2014).
     *A review of novelty detection*. Signal Processing.

3. Industry practices in ML system design and deployment:

   * Google Cloud ML Engineering best practices
   * “Hidden Technical Debt in Machine Learning Systems” — Sculley et al., NIPS 2015
