#  RF-DETR for Real-Time Pothole Detection on Edge Devices

*A Capstone Project*

---

##  Abstract

Road infrastructure monitoring is a critical challenge in many developing regions, where manual inspection is inefficient and costly. This project presents a real-time pothole detection system based on a **Robust Feature Detection Transformer (RF-DETR)** model. The system is designed to operate efficiently in **low-resource environments** and upon completion, on **edge devices such as smartphones and dashcams**.

The model was trained on a custom pothole dataset and evaluated using both quantitative metrics and qualitative analysis. Results show strong detection capability (mAP@50 ≈ 0.86, F1 ≈ 0.83), with limitations observed in detecting small-scale potholes. The system demonstrates practical applicability for deployment in real-world road monitoring scenarios.

---

##  Introduction

### Background

Poor road infrastructure significantly impacts transportation safety, economic productivity, and public health. Potholes are among the most common road defects and pose serious risks to vehicles and pedestrians.

Traditional inspection methods:

* Are manual and time-consuming
* Lack scalability
* Are prone to human error

### Motivation

With the rise of **computer vision and edge AI**, automated pothole detection systems can:

* Enable continuous monitoring
* Reduce operational costs
* Improve road maintenance response times

### Objective

To develop a **real-time, energy-efficient pothole detection system** using RF-DETR that:

* Achieves high detection accuracy
* Operates under limited computational resources
* Generalizes to real-world environments

---

##  Methodology

### Model Architecture

This project uses **RF-DETR (Robust Feature Detection Transformer)**, a transformer-based object detection model that:

* Leverages attention mechanisms for global feature understanding
* Improves robustness compared to CNN-based detectors
* Supports efficient training and inference

### Training Setup

* **Epochs:** 30
* **Batch Size:** 4 (effective batch size = 16 via gradient accumulation)
* **Learning Rate:** 1e-4
* **Optimizer:** Adam-based (default RF-DETR configuration)
* **EMA (Exponential Moving Average):** Enabled for improved generalization

### Dataset

* Custom pothole dataset in COCO format (https://universe.roboflow.com/muhammadmoin-arxtl/potholes-detection-jbnou/dataset/1)
* Single-class detection task (Pothole)
* Split into training, validation, and test sets

---

##  Results and Evaluation

### Overall Performance

| Metric    | Value |
| --------- | ----- |
| mAP@50:95 | 0.589 |
| mAP@50    | 0.864 |
| F1 Score  | 0.831 |
| Precision | 0.823 |
| Recall    | 0.839 |

### Test Set Performance

| Metric             | Value |
| ------------------ | ----- |
| mAP@50:95          | 0.473 |
| mAP@50             | 0.678 |
| AP (Large Objects) | 0.647 |
| AP (Small Objects) | 0.026 |

### Key Observations

* Strong detection performance for **medium and large potholes**
* Significant drop in performance for **small potholes**
* EMA weights improved model stability and performance

---

##  Training Analysis

### Loss Behavior

* Training loss decreased steadily
* Validation loss plateaued early (~epoch 5–8)

This indicates **early overfitting**

### mAP Trends

* Rapid convergence within first 10–15 epochs
* Performance stabilized afterward

### Interpretation

The model quickly learns discriminative features but reaches a performance ceiling due to dataset limitations.

---

##  Error Analysis

### False Positives

Observed in:

* Road patches
* Shadows
* Surface discoloration
* Cracks resembling potholes

 Cause: reliance on **texture-based features**

---

### False Negatives

Observed in:

* Small potholes
* Low-contrast defects
* Water-filled or irregular potholes

 Cause:

* Poor scale sensitivity
* Limited representation of subtle features

---

### Insight

The model performs well on **visually distinct potholes** but struggles with:

* ambiguous textures
* small-scale defects

---

##  Video Inference

The model was tested on real-world road videos containing dense pothole distributions.

### Observations

* Consistent detection across frames
* Robust to motion and perspective changes
* Minor flickering on rendered videos. (can be reduced via confidence filtering)

### Significance

Demonstrates readiness for:

* Dashcam-based monitoring
* Mobile deployment
* Real-time applications

---

##  Limitations

* Poor detection of small potholes
* Sensitivity to lighting and shadows
* Limited dataset diversity
* Slight overfitting observed

---

##  Future Work

* Expand dataset with diverse conditions (local road conditions)
* Improve small object detection (multi-scale training)
* Apply advanced augmentation techniques
* Deploy on edge devices (smartphones, IoT cameras)
* Integrate GPS for geospatial pothole mapping

---

##  Deployment Potential

This system can be deployed in:

* Smart city infrastructure
* Road maintenance agencies
* Fleet monitoring systems
* Crowdsourced road condition platforms

---

##  Conclusion

This project demonstrates that **transformer-based object detection (RF-DETR)** is a viable approach for pothole detection in real-world environments. Despite limitations in small object detection, the model achieves strong overall performance and shows clear potential for deployment in resource-constrained settings.

---

##  References

* Carion et al., *End-to-End Object Detection with Transformers (DETR)*
* RF-DETR Documentation
* COCO Dataset Evaluation Metrics
* OpenCV & Supervision Libraries

---

##  Author

**Jude Wandera**
Capstone Project — 2026

---
