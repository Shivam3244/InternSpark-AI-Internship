# Task 2 — Deep Learning Image Classification

## Objective
Train a small convolutional neural network (CNN) for image classification.

## Dataset
The project uses the CIFAR-10 dataset through TensorFlow/Keras. TensorFlow downloads the dataset automatically on the first run.

## Run
```bash
pip install -r requirements.txt
python train.py
```

## Outputs
- Trained model: `models/cifar10_cnn.keras`
- Training curves: `results/training_accuracy.png` and `results/training_loss.png`
- Test metrics printed in the terminal
