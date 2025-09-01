# Music Genre Classification with CNNs and RNNs

This repository contains my implementation for **Assessment 2: Music Genre Classification**, where the goal is to classify music genres using log-transformed Mel spectrograms derived from the GTZAN dataset. The project explores multiple deep learning architectures and optimization strategies to achieve high classification accuracy.


## Dataset

The dataset consists of log-transformed Mel spectrograms extracted from 800 songs across 8 genres. Each song is represented by 15 spectrograms, each of shape **(80, 80, 1)**.

- **Training set**: 80% of the data  
- **Validation set**: 20% of the data  

To use the dataset:
1. Download the training and validation folders.
2. Place them in your Google Drive.
3. Mount your Drive in Colab and load the data using:

```python
train_dataset = tf.data.Dataset.load('<path_to_train>')
validation_dataset = tf.data.Dataset.load('<path_to_val>')

## Architectures & Results

This section summarizes the three models implemented for music genre classification using log-transformed Mel spectrograms. Each model was trained for 50 epochs and evaluated on both training and validation sets.


### P1.1 – Shallow Parallel CNN

**Architecture Overview**:
- **Input**: (80, 80, 1)
- **Branch 1**:
  - Conv2D: 3 filters (8×8), LeakyReLU (α=0.3), padding='valid'
  - MaxPooling2D: pool size (4×4)
  - Flatten
- **Branch 2**:
  - Conv2D: 4 filters (4×4), LeakyReLU (α=0.3), padding='valid'
  - MaxPooling2D: pool size (2×2)
  - Flatten
- **Merge**:
  - Concatenate both branches
  - Dense layer with softmax activation (8 classes)

**Training Details**:
- Optimizer: SGD with momentum (0.9)
- Loss: Categorical Crossentropy
- Batch size: 128
- Epochs: 50

**Results**:
- Validation Accuracy: ~70%
- Observation: Overfitting detected — training accuracy high, validation loss increases after a point.


### P1.2 – CNN-RNN Hybrid

**Architecture Overview**:
- Conv2D: 8 filters (4×4) → MaxPooling2D (2×2)
- Conv2D: 6 filters (3×3) → MaxPooling2D (2×2)
- Reshape: (324, 6)
- LSTM: 128 units (return sequences)
- LSTM: 32 units (return last state)
- Dense: 200 units, ReLU
- Dropout: 0.2
- Output: Dense layer with softmax activation (8 classes)

**Training Details**:
- Optimizer: SGD with momentum (0.9)
- Loss: Categorical Crossentropy
- Batch size: 128
- Epochs: 50

**Results**:
- Validation Accuracy: >50%
- Observation: Balanced training and validation performance, minimal overfitting.


### P2 – Enhanced CNN-BiLSTM Model

**Architecture Overview**:
- Conv2D: 8 filters (4×4), padding='same' → BatchNorm → LeakyReLU (α=0.3) → MaxPooling2D (2×2)
- Conv2D: 6 filters (3×3), padding='same' → BatchNorm → LeakyReLU (α=0.3) → MaxPooling2D (2×2)
- Reshape: (100, 24) for richer time-frequency representation
- Bidirectional LSTM: 128 units (return sequences)
- Bidirectional LSTM: 32 units (return last state)
- Dense: 200 units, ReLU
- Dropout: 0.3
- Output: Dense layer with softmax activation (8 classes)

**Training Details**:
- Optimizer: Adam (learning rate = 1e-3)
- Loss: Categorical Crossentropy
- Batch size: 128
- Epochs: 50

**Results**:
- Final Validation Accuracy: **87%**
- Observation: Best-performing model with minimal validation loss and strong generalization.


### Visualizations

Each model includes plots of:
- Training vs. Validation Accuracy
- Training vs. Validation Loss

These plots help diagnose overfitting and track convergence across epochs.

### Notes
-All models were trained for 50 epochs using mini-batch stochastic gradient descent.
-The final model exceeds the required 85% validation accuracy threshold.
-Architectural decisions were guided by spectrogram characteristics and deep learning best practices.
