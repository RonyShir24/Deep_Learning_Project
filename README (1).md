# Deep Learning Projects Portfolio

This repository showcases three deep learning projects focusing on classification, similarity learning, and sequence generation. Each project explores different neural architectures and problem domains, reflecting a breadth of practical experience in applying deep learning techniques.

---

## 📚 Project 1: Fully Connected Neural Network on MNIST

**Objective**: Build and evaluate a deep neural network to classify handwritten digits from the MNIST dataset.

**Key Features**:
- Implemented a 4-layer neural network with ReLU and Softmax activations.
- Conducted experiments using various batch sizes, learning rates, and regularization techniques (Batch Normalization, L2 Regularization).
- Achieved 94.5% test accuracy using L2 regularization.

**Results**:
- **Best accuracy**: 94.5%
- **Regularization** helped reduce overfitting by centering weights around zero.
- **Batch Normalization** improved convergence time but slightly reduced performance due to naïve implementation.

---

## 👥 Project 2: Siamese Network for One-Shot Facial Recognition

**Objective**: Implement a Siamese Neural Network to perform one-shot facial recognition using the LFW-a dataset.

**Key Features**:
- Based on the architecture from *“Siamese Neural Networks for One-shot Image Recognition”*.
- Preprocessing included cropping and resizing images to 105x105, and data augmentation using horizontal flips and rotations.
- Conducted 384 experiments with varying hyperparameters (learning rate, batch size, dropout, etc.).
- Used SGD optimizer with binary cross-entropy loss.

**Results**:
- **Best model** achieved 81.15% test accuracy.
- **Training time** ranged from 230s–340s across top models.
- Misclassification analysis revealed challenges with similar faces and varying expressions.
- Data augmentation and preprocessing significantly improved accuracy.

---

## 🎵 Project 3: Lyric Generation from Melody using LSTM

**Objective**: Generate lyrics from song melodies using an LSTM-based architecture with Word2Vec embeddings and MIDI feature extraction.

**Key Features**:
- Extracted musical features from MIDI files using `pretty_midi`.
- Two feature extraction strategies explored (tempo/chroma vs. instrumentation/duration).
- Lyrics were cleaned and tokenized; Word2Vec embeddings (Google News) used for word representation.
- Combined melody and lyrics in the model input.
- Used metrics like Cosine Similarity, Levenshtein Distance, and Polarity to evaluate output quality.

**Results**:
- **Generated lyrics** align emotionally with seed words ("love", "dark", "light").
- Best models from each feature extraction approach retrained with early stopping.
- Clear trade-offs observed based on preprocessing strategy and hyperparameter choice.
