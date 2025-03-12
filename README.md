# Build AutoEncoder From Scratch
This project implements Fully Connected and Convolutional AutoEncoders using PyTorch for MNIST image reconstruction. The Fully Connected AutoEncoder maps images to a 3D latent space, while the ConvAutoEncoder leverages convolutional layers for improved feature extraction. Both models demonstrate strong performance, with clear visual reconstructions and effective training strategies.

## Autoencoder Performance Comparison

**1️⃣ AutoEncoder (Fully Connected - Trained for 20 Epochs)**
<p align="center">
    <img src="https://github.com/user-attachments/assets/3d58bf6b-9779-4b46-818c-4a6c7f4f569b" width="45%">
    <img src="https://github.com/user-attachments/assets/3d5380ef-fced-421a-9326-9937836aeb23" width="45%">
</p>

---

**2️⃣ ConvAutoEncoder (Convolutional - Trained for 10 Epochs)**
<p align="center">
    <img src="https://github.com/user-attachments/assets/ae203527-1675-45bf-9469-e4cee04ae3dd" width="45%">
    <img src="https://github.com/user-attachments/assets/aa7bcf1e-4045-4b2f-9a2f-d884bed627fa" width="45%">
</p>

---

This project implements two types of autoencoders using PyTorch:
1. **Fully Connected AutoEncoder**
2. **Convolutional AutoEncoder (ConvAutoEncoder)**

Both models are trained using the MNIST dataset to reconstruct digit images.

---

## 🚀 Installation
1. Install the dependencies directly within the Jupyter Notebook:
```python
!pip install torch torchvision pandas matplotlib
```

---

## 🧠 Models Overview
### 1️⃣ AutoEncoder (Fully Connected)
- Encoder compresses 28x28 input into a **3-dimensional** latent space.
- Decoder reconstructs the original input from the compressed representation.

**Encoder Architecture:**
- Linear layers with ReLU activation.
- Decreasing layer sizes: `784 → 128 → 64 → 32 → 16 → 12 → 3`

**Decoder Architecture:**
- Linear layers with ReLU activation and a final Tanh layer.
- Increasing layer sizes: `3 → 12 → 16 → 32 → 64 → 128 → 784`

### 2️⃣ ConvAutoEncoder (Convolutional)
- Encoder uses convolution layers with BatchNorm for feature extraction.
- Decoder uses ConvTranspose layers for reconstruction.

**Encoder Architecture:**
- `Conv2d → BatchNorm2d → ReLU` layers with reducing spatial dimensions.

**Decoder Architecture:**
- `ConvTranspose2d → BatchNorm2d → ReLU` layers with increasing spatial dimensions.

---

## 📊 Dataset Preparation
The dataset used is **MNIST**, provided in CSV format.
- **Features:** Pixel values (0-255) scaled to `[0, 1]`.
- **Labels:** Digits (0-9), used for dataset structure but not for training.

**Sample Transformation:**
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
```

---

## 🔥 Training Procedure
### Training Parameters
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Learning Rate:** 1e-3
- **Weight Decay:** 1e-5
- **Batch Size:** 64

### Training Execution
Run each code block in sequence within the Jupyter Notebook to train the models.

---

## 📈 Visualizing Results
Use the provided visualization code to display original and reconstructed images:
```python
visualize_outputs(outputs1, 5)
visualize_outputs(outputs, 5)
```

---

## 📋 Results
- Fully Connected AutoEncoder achieves weak reconstruction on digit images.
- ConvAutoEncoder demonstrates sharper and more accurate reconstructions.

---

## 🧩 Future Enhancements
- Add deeper architectures for improved accuracy.
- Introduce variational autoencoders (VAE) for better latent space learning.
- Implement custom data augmentation for robustness.
- 
---

## 🙌 Acknowledgements
- **PyTorch** — For efficient deep learning model implementation.
- **MNIST Dataset** — For digit image recognition tasks.

