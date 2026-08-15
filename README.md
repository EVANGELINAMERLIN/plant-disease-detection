# 🌿 Plant Disease Detection System

An AI-powered computer vision application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model classifies healthy and diseased leaves across multiple plant species.

## 📌 Overview

This project uses deep learning to identify plant diseases from leaf images, helping farmers and gardeners diagnose crop health issues quickly. The system is trained on the PlantVillage dataset and deployed through an interactive web dashboard for real-time predictions.

## ✨ Key Features

- **Image Processing** — Automated resizing, normalization, and augmentation of leaf images
- **Computer Vision** — CNN-based feature extraction from raw leaf images
- **CNN Model Development** — Custom 4-block convolutional neural network architecture
- **TensorFlow / Keras** — Model built and trained using TensorFlow's Keras API
- **Image Classification** — Multi-class classification across 15 disease/healthy categories
- **Prediction Dashboard** — Interactive Streamlit app for uploading and classifying leaf images in real time

## 🛠 Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV, Pillow (PIL)
- **Data Science:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Dashboard:** Streamlit
- **Dataset Access:** KaggleHub

## 📂 Project Structure
  plant-disease-detection/
│
├── models/
│ ├── plant_disease_model.h5 # Trained CNN model
│ ├── class_indices.json # Class label mapping
│ ├── training_history.png # Accuracy/loss plots
│ ├── confusion_matrix.png # Confusion matrix visualization
│ └── performance_report.txt # Classification metrics report
│
├── dashboard/
│ └── app.py # Streamlit prediction dashboard
│
├── sample_test_images/ # Sample leaf images for testing
│
├── Plant Disease.ipynb # Full notebook (data prep, training, evaluation)
├── requirements.txt
└── README.md
 ## 📊 Dataset

**[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)** (via Kaggle)

- ~20,000+ labeled leaf images
- 15 classes covering Tomato, Potato, and Pepper plants (healthy and diseased)
- Sample images included in `sample_test_images/` for quick testing

## 🧠 Model Architecture

A custom Convolutional Neural Network with:
- 4 convolutional blocks (Conv2D → BatchNormalization → MaxPooling), with filter sizes increasing from 32 → 256
- GlobalAveragePooling2D layer to reduce parameters and overfitting
- Dense layer (256 units, ReLU activation)
- Dropout (0.5) for regularization
- Softmax output layer for multi-class classification

**Training details:**
- Image size: 96×96
- Data augmentation: rotation, zoom, shift, shear, horizontal flip
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/EVANGELINAMERLIN/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Explore / retrain the model
Open `Plant Disease.ipynb` in Jupyter Notebook to view the full data pipeline, model training, and evaluation process.

### 4. Launch the prediction dashboard
```bash
streamlit run dashboard/app.py
```
Then open your browser at **http://localhost:8501**

Upload any leaf image (JPG/PNG) to get an instant disease prediction with confidence score.

## 📈 Results

Model performance metrics (precision, recall, F1-score per class) are available in [`models/performance_report.txt`](models/performance_report.txt).

Training accuracy/loss curves and confusion matrix visualizations are available in the `models/` folder.

## 👩‍💻 Author

**Evangelina Merlin**  
GitHub: https://github.com/EVANGELINAMERLIN/plant-disease-detection

## 📄 License

This project is open-source and available for educational purposes.
