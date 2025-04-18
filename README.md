# Infection Classification and Event Detection System

This repository implements an end-to-end machine learning pipeline for classifying infections using structured patient data, time-series vitals, and event-based text sequences. It integrates LSTM-based deep learning with a domain-aware event generator to support predictive modeling in medical scenarios, such as sepsis detection.

---

## 📌 Project Highlights

- 🔍 **Event-Driven Data Modeling**  
  Converts time-series patient vitals into structured event chains based on clinically significant changes.

- 🧠 **LSTM-Based Classification Model**  
  A deep learning model (with Embedding + Bidirectional LSTM) trained on textual event sequences and demographic features.

- ⚙️ **TPU/GPU Compatible**  
  Automatically detects and utilizes TPU for faster training if available.

- 🧪 **Real-Time Testing & Evaluation**  
  Includes confusion matrix, classification report, and prediction on new sample cases.

---

## 🔧 Execution Flow

### 1. **Infection Classification with Deep Learning**

- Load and preprocess the dataset (`infection_dataset_final_balanced.csv`)
- Tokenize the `Sequence` feature representing clinical events
- Combine tokenized sequences with numerical data (`Age`, `Gender`, `Health Condition`)
- Encode infection labels and apply class weighting
- Train a deep learning model with the following architecture:
  - Embedding → BiLSTM → BatchNorm → Dropout → BiLSTM → Dense Layers
- Evaluate on test data and visualize results

### 2. **Synthetic Event Sequence Generator for Sepsis Simulation**

- `generate_sepsis_dataset(n=20)` simulates vitals over time (e.g., HRV, BP, Temp, SpO2)
- `detect_events(patient_records)` detects significant clinical changes across time intervals
- Transforms sequential vital changes into a structured `Event Chain`, e.g.:  
  `"High HR Variability (Short) + Low BP (Short) → High Body Temp (Long) + Low SpO2 (Short)"`

This text sequence is compatible with the model’s tokenizer and can be used for inference or further training.

---

## 🚀 How to Run

1. Place your dataset (`infection_dataset_final_balanced.csv`) in the project directory.
2. Run the infection classification model script to train and evaluate:
   ```bash
   python infection_model.py
