# ZERVE AI Datathon — Health Insurance Claim Prediction

## Overview
Presented at IIT Bombay as Finalists.
-
This repository contains our machine learning solution developed for the **ZERVE AI Datathon**, where the goal was to predict the **probability of a customer filing a significant health insurance claim** using an anonymized dataset with 50 engineered features.

The solution focuses on building a **robust ensemble pipeline** optimized for the **Normalized Gini Coefficient**, with special attention to **class imbalance**, **generalization**, and **ranking quality**.

---

## Problem Summary

Health insurance providers need reliable risk models to identify high-risk customers likely to file claims. The task was to generate **probability predictions (0–1)** for each row in the test dataset, evaluated using the **Normalized Gini Coefficient**.

The dataset is **imbalanced**, with claim events being relatively rare.

---

## Core Approach

We adopted an ensemble-driven strategy combining multiple gradient boosting models with disciplined cross-validation:

* **XGBoost**, **LightGBM**, and **CatBoost** were trained independently
* **Stratified K-Fold Cross-Validation (5 folds)** was used throughout
* **Out-of-Fold (OOF) predictions** were generated to prevent data leakage

Each model produced probabilistic outputs that were later combined via blending.

---

## Handling Imbalanced Data

Class imbalance was addressed using **model-native techniques**:

* LightGBM: `scale_pos_weight`
* CatBoost: `auto_class_weights = Balanced`
* Stratified sampling to preserve class ratios across folds

This ensured stable learning and improved ranking of rare positive cases.

---

## Two-Stage Modeling Strategy

We initially experimented with a **soft two-stage approach**, where base model probabilities were fed into an additional training stage. While promising, this setup showed instability.

The final and more effective strategy was:

* Generate **leakage-free OOF predictions** from each base model using K-Folds
* Blend these predictions using a **regularized linear meta-model**

This approach improved generalization, reduced overfitting, and delivered more consistent performance.

---

## Evaluation Strategy

Model performance was validated using **OOF AUC**, which directly translates to **Normalized Gini** (Gini = 2 × AUC − 1). This ensured alignment with the competition metric and reliable estimation of test performance.

---

## Final Output

The final output is a **single CSV file** containing predicted probabilities for each row in the test dataset, representing the likelihood of a customer filing a health insurance claim.


The solution was fully implemented and executed within the **Zerve AI Canvas** environment.

Link:https://app.zerve.ai/notebook/968ba3ef-05fc-4e3e-93b4-2577c039ead9?left_sidebar=executor_images

