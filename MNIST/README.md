# MNIST Handwritten Digit Classification

This project implements handwritten digit classification using the **MNIST dataset**.  
It includes two versions:
- **From Scratch:** Implemented only with NumPy (no deep learning framework)
- **PyTorch Version:** Implemented using the PyTorch library

## Project Structure
MNIST/
│
├── data/ # MNIST dataset (downloaded manually)
│ ├── train-images-idx3-ubyte(.gz)
│ ├── train-labels-idx1-ubyte(.gz)
│ ├── t10k-images-idx3-ubyte(.gz)
│ └── t10k-labels-idx1-ubyte(.gz)
│
├── results/ # Output images / logs / trained models (if any)
│
├── main_scratch.ipynb # Implementation from scratch (NumPy only)
├── main_pytorch.ipynb # Implementation with PyTorch
└── README.md

## Notifications
- This project does not contain validation part, only training and testing included.