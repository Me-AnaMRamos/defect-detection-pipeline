

# Defect Detection Pipeline for Operational Monitoring

## Problem Overview

In industrial and operational environments, visual inspection is a critical step for ensuring product quality and system reliability. Components or surfaces may develop defects such as scratches, cracks, deformations, or contaminations that indicate abnormal operating conditions or manufacturing faults.

Traditionally, defect detection relies on manual visual inspection, which is time-consuming, subjective, and difficult to scale. As production volumes increase and quality requirements become stricter, automated image-based inspection systems are increasingly necessary.

This project addresses the problem of **automated defect detection from images**, framing it as a **binary classification task**:

* **Normal**: Image corresponds to an expected, non-defective operational state.
* **Defective**: Image contains visual anomalies indicating a deviation from normal conditions.

The objective is to design an end-to-end machine learning pipeline that supports reliable operational monitoring while explicitly considering the trade-offs between false positives and false negatives.

---

## Operational Considerations

* **False positives** (normal classified as defective) may lead to unnecessary inspections, increased costs, and reduced trust in the system.
* **False negatives** (defective classified as normal) may allow faulty components to pass undetected, potentially causing safety risks, system failures, or downstream losses.

For many real-world operational decisions (e.g., accept/reject, trigger alert/continue operation), a binary decision is sufficient and often preferred. Therefore, the system is designed with deployment and decision-making constraints in mind, rather than purely academic performance metrics.

---

## Dataset Description

This project uses a subset of the **MVTec Anomaly Detection (MVTec AD)** dataset.

MVTec AD is a widely used industrial inspection dataset containing high-resolution images of manufactured objects and surfaces under controlled conditions. It includes:

* Multiple object and texture categories
* Clean *normal* samples
* Diverse real-world defect types such as cracks, scratches, contaminations, and structural anomalies

Although originally designed for anomaly detection, the dataset is adapted here to a **binary classification setting** by grouping all anomalous samples under a single *defective* label. This mirrors real operational scenarios where the primary goal is to detect whether intervention is required, rather than to classify the specific defect type.

**Key reasons for choosing MVTec AD:**

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
     Preprocessing & Augmentation
             │
             ▼
      Feature Learning (CNN)
             │
             ▼
      Binary Classification
   (Normal vs Defective)
             │
             ▼
        Model Evaluation
   (Precision, Recall, Trade-offs)
             │
             ▼
        Inference Service
        (API / Container)
```

This structure allows clear separation between data handling, model training, evaluation, and deployment, enabling maintainability and future extensions.

---

## Project Structure

The repository is organized to reflect best practices for machine learning systems intended for production environments:

```
defect-detection-pipeline/
├── data/          # Dataset handling and documentation
├── notebooks/     # Exploration and experiments
├── src/           # Core pipeline logic
├── api/           # Inference service
├── docker/        # Containerization for deployment
├── configs/       # Configuration files
├── tests/         # Unit and integration tests
├── requirements.txt
└── README.md
```

---

## Project Goals

* Build a realistic image-based defect detection pipeline
* Emphasize operational decision-making constraints
* Demonstrate clean structure, reproducibility, and deployment readiness
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
