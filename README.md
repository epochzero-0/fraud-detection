## Overview

This project tackles a real-world fraud detection problem using a dataset of over **6.36 million financial transactions**. The goal is to proactively identify fraudulent activity using machine learning - with high precision, minimal false positives, and clear interpretability.

## Problem Statement

Develop a machine learning pipeline to:
- Detect fraudulent transactions with high accuracy
- Explain the driving factors behind each prediction
- Optimize real-world fraud prevention workflows (e.g., Top-N review)

## Solution Approach

- **Model Used**: XGBoost (`tree_method='hist'` for scale)
- **Threshold Tuning**: F1-optimized decision boundary (≈ 0.91 F1 score)
- **Interpretability**: SHAP values for global feature importance
- **Scalability**: Trained and evaluated on the full 6.36M dataset

## Final Model Performance

| Metric            | Value          |
|-------------------|----------------|
| Precision (Fraud) | 93%            |
| Recall (Fraud)    | 89%            |
| F1 Score          | 0.91           |
| ROC-AUC           | 0.99986        |
| False Positives   | 108            |
| False Negatives   | 177            |

### Top-N Fraud Ranking Result

- **Top 1000 highest-probability predictions** contained **1000 actual frauds**
- **100% fraud detection yield** in review queue (vs. 0.13% baseline fraud rate)
- ~**775x improvement** over random selection

## SHAP Insights

SHAP analysis confirmed that the model relies on sensible, business-aligned indicators:
- High `amount`
- Sudden drops in `newbalanceOrig`
- Specific transaction types like `TRANSFER` and `CASH_OUT`

## Key Takeaways

- Model is **ready for real-time deployment**
- Offers **explainability + high precision**
- Scales to millions of transactions
- Effective for fraud operations with **limited human review bandwidth**
- [Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) used
