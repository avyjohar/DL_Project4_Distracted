# README

## Overview

This project is an image classification task aimed at detecting distracted driving behaviors using the State Farm Distracted Driver Detection dataset. The implementation leverages TensorFlow and Keras with a MobileNetV2 architecture for transfer learning.

---

## Installation

Ensure you have Python installed, preferably through Anaconda. Install the required dependencies using pip:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn
```

---

## Dataset

Download the dataset and extract it. Update the `dataset_path` in the code to point to the dataset's location on your system.

---

## Project Structure

- `dataset_path`: Directory containing `train` and `test` folders.
- `train`: Contains subfolders for each class (`c0` to `c9`).
- `test`: Contains images for testing.
- `Result`: Directory where the final submission file will be saved.

---

## Classes

The dataset contains 10 classes:

- `c0`: Safe driving
- `c1`: Texting - right hand
- `c2`: Talking on the phone - right hand
- `c3`: Texting - left hand
- `c4`: Talking on the phone - left hand
- `c5`: Operating the radio
- `c6`: Drinking
- `c7`: Reaching behind
- `c8`: Hair and makeup
- `c9`: Talking to a passenger

---

## Model Architecture

- Base model: MobileNetV2 with pre-trained weights from ImageNet.
- Custom layers: 
  - Global Average Pooling
  - Dropout
  - Dense layers for classification into 10 classes.
- Optimizer: Adam
- Loss function: Categorical Crossentropy

---

## Training

1. Data augmentation is applied to the training set.
2. Train-validation split is 80%-20%.
3. Model checkpoints and early stopping are used for efficient training.

Run the training process:

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)
```

---

## Evaluation

The model's performance is plotted for accuracy and loss over epochs. The best model is evaluated on the test set, and predictions are saved in a submission file.

```python
submission_df.to_csv(submission_path, index=False)
```

---

## Results

- Training accuracy: ~89.1%
- Validation accuracy: ~92.0%
- Final predictions are saved in `Result/submission.csv`.

---

## Visualization

Sample images from each class are displayed for better understanding of the dataset. Model performance is visualized through training and validation accuracy and loss plots.

---

## Directory Setup

Ensure the following directory structure before running the code:

```
state-farm-distracted-driver-detection/
│
├── imgs/
│   ├── train/
│   │   ├── c0/
│   │   ├── c1/
│   │   ├── ...
│   ├── test/
│
├── Result/
```

---

## Submission

The submission file contains:

- `img`: Image filename
- `label`: Predicted class label (`c0` to `c9`)

---

## Notes

- Modify paths as per your system setup.
- Use GPU for faster training.
- For issues, ensure all dependencies are installed correctly and the dataset is structured as expected.



```python

```
