# Driver-monitoring-system_CV

Repository for monitoring whether a driver is drowsy or non-drowsy.

## Overview

This project trains a deep learning model to classify driver state (drowsy, distracted, etc.) using facial landmarks. It uses a 68-point facial landmark dataset (such as those produced by dlib), formatted as CSVs, and trains a 1D Convolutional Neural Network (CNN) for binary classification.

## Features

- **Facial Landmark Dataset**: The system expects CSV files containing columns: class_label, x1...x68, y1...y68, where each (xi, yi) pair represents a facial landmark.
- **PyTorch Implementation**: The model and data pipeline are implemented in PyTorch, leveraging GPU acceleration if available.
- **CNN Architecture**: A simple Conv1D network processes the 2D landmark coordinates and predicts the driver’s state.
- **Training and Validation**: The dataset is randomly split (90% train, 10% validation). Class balance and performance metrics are tracked.
- **Model Checkpointing**: The trained model is saved for later inference or fine-tuning.

## Data Preparation

- Input: CSV file (`updated_csv.csv` by default) with columns:
  - `class_label`: Label for each sample (e.g., 'drowsy', 'distracted')
  - `x1...x68`: X-coordinates of facial landmarks
  - `y1...y68`: Y-coordinates of facial landmarks

## Model Architecture

- **Input**: Tensor of shape (2, 68) per sample (2 channels: x and y, 68 landmarks each)
- **Convolution Block**: 1D conv (2 → 16 channels) + BatchNorm + ReLU
- **Pooling and Flattening**
- **Fully Connected Head**: Dense layers (68 → 128 → 32 → 1) with dropout and batch normalization
- **Output**: Single logit (interpreted as binary class: drowsy or not)

## Training

- **Optimizer**: Adam
- **Loss Function**: BCEWithLogitsLoss (for binary classification)
- **Metrics**: Tracks epoch-wise accuracy and loss for both train and validation sets
- **Epochs**: Default 15
- **Batch Size**: Default 256
- Plots training and validation loss curves for visual inspection

## Usage

1. Prepare your dataset in CSV format as described above.
2. Adjust parameters (paths, batch size, epochs, etc.) as needed in the notebook.
3. Run the notebook to train the model.
4. The model checkpoint will be saved as `saved_landmark_model.pth` for later inference.

## Example Class Mapping

```
Class → index mapping: {'distracted': 0, 'drowsy': 1}
```

## Results

- The notebook displays accuracy and loss per epoch for both training and validation sets.
- Example output:
  ```
  Epoch 15/15  Train Loss: 0.0858, Train Acc: 0.9706  ||  Val Loss: 0.0618, Val Acc: 0.9780
  ```

## Visualization

- The notebook plots training vs. validation loss to help monitor overfitting and learning progress.

## Model Saving

- After training, the model and the class-to-index mapping are saved using PyTorch’s `torch.save()` to `saved_landmark_model.pth`.

## Requirements

- Python 3.12+
- PyTorch
- pandas, numpy
- tqdm, matplotlib

## Reference

Notebook: [`dlib_68_facial_points_deeplearning_model.ipynb`](https://github.com/saurabh-itg/Driver-monitoring-system_CV/blob/main/dlib_68_facial_points_deeplearning_model.ipynb)

---

If you need a usage example or additional details (such as inference code), let me know!