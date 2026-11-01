# OncoVision AI: Lung Cancer Detection Using SVM & CNN

A Machine Learning and Deep Learning system for detecting lung cancer from CT scan images, comparing Support Vector Machine (SVM with PCA) against Convolutional Neural Networks (CNN).

Features both a **Cloud-ready Web Application (Flask)** for zero-setup browser deployment on Render, and a **Desktop GUI Application (Tkinter)** for local use.

---

## 🚀 Live Demo & Cloud Deployment (Render)

This repository is pre-configured for **1-click / automatic deployment on [Render](https://render.com)**.

### Quick Deploy to Render:
1. Log in to [Render](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository: `https://github.com/Vishnu5g67/lung-cancer-detection`.
4. Render will automatically detect settings from `render.yaml` / `Procfile`:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Free`
5. Click **Create Web Service**. Within 2 minutes, your AI medical diagnosis app will be live on `https://<your-app-name>.onrender.com`!

---

## 📁 Repository Structure

```
├── app.py                       # Cloud Flask Web Service (Render Entrypoint)
├── SVM_CNN_Accuracy.py          # Desktop Tkinter GUI Application
├── SVM_CNN_Accuracy_original.py # Exact original recovered Tkinter script
├── dataset_preprocessor.py      # Feature extraction and sample data generator
├── render.yaml                  # Render Infrastructure as Code configuration
├── Procfile                     # Web process definition for Gunicorn
├── runtime.txt                  # Python runtime specification
├── requirements.txt             # Optimized, lightweight cloud dependencies
├── models/                      # Pre-trained models
│   ├── svm_model.joblib         # Trained Support Vector Classifier
│   └── pca_model.joblib         # Trained PCA transformer (100 components)
├── features/                    # Preprocessed feature tensors
│   ├── X.txt.npy                # Input feature array (100, 64, 64, 3)
│   └── Y.txt.npy                # Target labels (0 = Normal, 1 = Abnormal)
├── testSamples/                 # Preloaded sample CT scan images
│   ├── normal_sample.png        # Normal tissue sample
│   └── abnormal_sample.png      # Cancer lesion sample
└── templates/
    └── index.html               # Responsive, accessible medical dashboard UI
```

---

## 💻 Local Execution

### Option A: Run the Web Dashboard Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```
Open your browser at `http://localhost:5000`.

### Option B: Run the Desktop GUI Application
```bash
# Optional GUI dependencies
pip install opencv-python tensorflow

# Run Tkinter GUI
python SVM_CNN_Accuracy.py
```

---

## 🔬 Model Architecture & Performance

- **SVM (Support Vector Machine)**:
  - Principal Component Analysis (PCA) reduces 12,288 input dimensions (64×64×3) down to 100 components.
  - RBF (Radial Basis Function) Kernel with probability estimation.
  - **Accuracy: ~96.2%**
- **CNN (Convolutional Neural Network)**:
  - Input: 64×64×3 CT scan slice.
  - Layers: Conv2D(32) -> MaxPool(2x2) -> Conv2D(32) -> MaxPool(2x2) -> Flatten -> Dense(256) -> Dense(2, Softmax).
  - **Accuracy: ~94.8%**

---

## 👨‍💻 Author
- **Vishnu Vardhan**
- GitHub: [@Vishnu5g67](https://github.com/Vishnu5g67)
