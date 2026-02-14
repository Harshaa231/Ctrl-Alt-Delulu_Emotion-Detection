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

## Assumptions
- The dataset contains two columns: text (sentence) and label (emotion).
- Labels are automatically mapped to numeric IDs.
- Dataset did not provide validation/test splits, so an 80-10-10 split was created.
- Maximum sequence length was limited to 128 tokens.
- Pretrained BERT weights provide meaningful semantic representations.

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

## Model Architecture
- Pretrained BERT encoder (bert-base-uncased)
- Dropout layer for regularization
- Fully connected classification layer

The [CLS] token representation was used for classification.

## Training Details
- Optimizer: AdamW
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 2
- Loss Function: CrossEntropyLoss
- Training done using GPU when available.

## Results
Test Accuracy ≈ 95–96%
Weighted F1 ≈ 0.96

## Observations
- The model achieved ~95–96% test accuracy.
- Training and validation accuracy remained close, indicating minimal overfitting.
- Performance was strong across most classes.
- Lower performance was observed for "sadness" and "loneliness", suggesting semantic overlap between these emotions.
- BERT provided strong contextual understanding of emotional text.

## Inference Pipeline
A function predict_text(text) was implemented that:
- Takes raw text input
- Tokenizes using BERT tokenizer
- Predicts emotion label
- Returns confidence score
The pipeline was tested on custom examples.


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
