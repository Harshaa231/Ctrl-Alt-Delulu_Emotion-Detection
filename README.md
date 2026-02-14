# Emotion Detection using BERT (Pure PyTorch)

## Project Overview
This project performs emotion classification using a fine-tuned BERT model trained with pure PyTorch.

### The implementation includes:
Exploratory Data Analysis (EDA)
BERT fine-tuning
Evaluation metrics
Confusion matrix
Inference pipeline

All implementation is contained in a single Jupyter notebook.

## Dataset
Source: Hugging Face datasets library
Dataset: shreyaspullehf/emotion_dataset_100k
Task: Multi-class emotion classification

## Pipeline
1. Exploratory Data Analysis
Loaded dataset using Hugging Face datasets
Visualized class distribution
Checked class imbalance

2. Model Fine-tuning
Pretrained BERT encoder
Fine-tuned using PyTorch only
Custom training loop

3. Evaluation
Accuracy
Precision (weighted)
Recall (weighted)
F1 Score (weighted)
Confusion Matrix

4. Inference Pipeline
Function:
predict_text(text: str)
Returns predicted emotion and confidence.

## Results
Test Accuracy ≈ 95–96%
Weighted F1 ≈ 0.96

## How to Run
Open the notebook:
emotion_detection_bert.ipynb
Run all cells.

## Dependencies
PyTorch
Transformers
Hugging Face datasets
NumPy
Scikit-learn
Matplotlib
Seaborn
