# Machine Learning-Based Polysomnography Data Analysis for ADHD Diagnosis

**Author**: Amirhossein Eskorouchi  
**Affiliation**: Department of Industrial and Systems Engineering, Mississippi State University  
**Conference**: 40th Southern Biomedical Engineering Conference (SBEC), 2024

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Methodology](#methodology)
4. [Model Performance](#model-performance)
5. [Feature Importance](#feature-importance)
6. [Conclusion](#conclusion)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)
9. [License](#license)

---

## Project Overview

This project uses machine learning models to analyze **polysomnography (PSG)** data, focusing on identifying **sleep stage-based biomarkers** for **ADHD diagnosis**. Our dataset includes physiological recordings such as EEG, EOG, EMG, and ECG data across five sleep stages. A **Random Forest classifier** was employed, and **nested cross-validation** ensured robust model performance.

---

## Data Description

The dataset includes PSG recordings from **48 individuals** (25 ADHD and 23 non-ADHD), each with ~8 hours of sleep data. Data was recorded across the following **sleep stages**:
- **Wake**
- **Sleep Stage 1**
- **Sleep Stage 2**
- **Sleep Stage 3-4**
- **REM (Rapid Eye Movement)**

A total of **20,294 epochs for ADHD** and **19,968 epochs for non-ADHD** individuals were analyzed.

---

## Methodology

1. **Data Preprocessing**:
   - EEG data was preprocessed using the **Reference Electrode Standardization Technique (REST)**.
   - A **band-pass filter** (1 Hz to 100 Hz) and a **notch filter** (60 Hz) were applied to remove noise.

2. **Feature Extraction**:
   - Correlation matrices were computed between 17 PSG channels to capture interactions between different physiological systems.
   - These matrices were transformed into graphs, and we calculated the **average shortest path length** as a measure of network efficiency across all sleep stages.

3. **Machine Learning**:
   - We used a **Random Forest classifier** with nested cross-validation for model evaluation and **GridSearchCV** for hyperparameter tuning.

---

## Model Performance

The model achieved the following average results across 5-fold cross-validation:
- **Accuracy**: 72%
- **Precision**: 71%
- **Recall**: 85%
- **F1-Score**: 76%

These metrics highlight the model's balanced performance in detecting ADHD cases based on PSG data.

---

## Feature Importance

**Permutation analysis** was applied to assess the contribution of each sleep stage. The most important features for ADHD classification were:
- **Sleep Stage 3-4** (Importance ≈ 0.14)
- **Sleep Stage 1** (Importance ≈ 0.12)
- **Wake** (Importance ≈ 0.09)

Shuffling these features resulted in significant drops in model performance, underscoring their relevance in ADHD diagnosis.

---

## Conclusion

The study demonstrated that **Sleep Stage 1** and **Sleep Stage 3-4** are critical markers for ADHD diagnosis. The model's ability to accurately classify ADHD cases suggests that sleep stage transitions can be valuable biomarkers. Future work will focus on expanding the dataset and incorporating other physiological signals such as heart rate variability.

---

## Requirements

To run this project, the following Python libraries are required:
- `numpy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `networkx`

You can install all dependencies via:
```bash
pip install -r requirements.txt
