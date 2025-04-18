import pandas as pd
import numpy as np
import tensorflow as tf
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# TPU Setup
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("\u2705 TPU detected and initialized")
except:
    print("\u274C TPU not found, using CPU/GPU")
    strategy = tf.distribute.get_strategy()

# Load Dataset
file_path = "infection_dataset_final_balanced.csv"  # Update with correct path

df = pd.read_csv(file_path)

# Preprocessing
def clean_sequence(seq):
    seq = re.sub(r"[^\w\s()]", "", seq)  # Remove special characters
    seq = seq.replace("→", " + ")  # Replace arrows with '+'
    return seq.strip()

df["Sequence"] = df["Sequence"].astype(str).apply(clean_sequence)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
df["Health Condition"] = LabelEncoder().fit_transform(df["Health Condition"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["Sequence"])
sequences = tokenizer.texts_to_sequences(df["Sequence"])
max_seq_length = max(len(seq) for seq in sequences)
X_seq = pad_sequences(sequences, maxlen=max_seq_length, padding="post")

# Prepare numerical features
scaler = StandardScaler()
X_additional = scaler.fit_transform(df[["Age", "Gender", "Health Condition"]])
X = np.hstack((X_seq, X_additional))

# Target variable encoding
y = df["Infection"].astype("category").cat.codes.values

distribution = Counter(y)
max_count = max(distribution.values())
class_weights = {cls: np.sqrt(max_count / count) for cls, count in distribution.items()}
print("Class Weights:", class_weights)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Architecture
with strategy.scope():
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=512, input_length=X.shape[1]),
        Bidirectional(LSTM(256, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(distribution), activation="softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0003, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

# Train Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64, class_weight=class_weights)

# Model Evaluation
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\n📊 *Classification Report:*")
print(classification_report(y_test, predicted_classes))

# Test Model on New Data
new_data = pd.DataFrame({
    "Sequence": [
        "High HR (Long) + Low BP (Short) + High Body Temp (Long) + Low SpO2 (Short)",
        "High HR (Short) + High HR Variability (Long) + Low BP (Long)",
        "High Respiratory Rate (Long) + Low SpO2 (Short) + High Body Temp (Long)",
        "Low BP (Short) + High Body Temp (Short) + High HR Variability (Long)",
        "High HR (Long) + Low BP (Short) + High Body Temp (Short)"
    ],
    "Age": [34, 56, 23, 45, 67],
    "Gender": [1, 0, 1, 1, 0],
    "Health Condition": [2, 3, 1, 0, 2]
})

new_data["Sequence"] = new_data["Sequence"].apply(clean_sequence)
new_data["Gender"] = LabelEncoder().fit_transform(new_data["Gender"])
new_data["Health Condition"] = LabelEncoder().fit_transform(new_data["Health Condition"])
new_sequences = tokenizer.texts_to_sequences(new_data["Sequence"])
new_X_seq = pad_sequences(new_sequences, maxlen=max_seq_length, padding="post")
new_X_additional = scaler.transform(new_data[["Age", "Gender", "Health Condition"]])
new_X = np.hstack((new_X_seq, new_X_additional))

new_predictions = model.predict(new_X)
new_predicted_classes = np.argmax(new_predictions, axis=1)
print("New Data Predictions:", new_predicted_classes)