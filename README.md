# 🧬 Melanoma Detection --- End-to-End Deep Learning Project

An end-to-end deep learning solution for detecting melanoma skin cancer
from dermoscopic images using Convolutional Neural Networks (CNNs) and
deploying predictions via a Streamlit web application.

------------------------------------------------------------------------

## 🚀 Project Overview

This project builds an automated melanoma detection system that
classifies skin lesion images into nine diagnostic categories. The
solution leverages deep learning for feature extraction and provides a
user-friendly interface for real-time predictions.

Early melanoma detection is critical because timely diagnosis
significantly improves treatment outcomes and reduces mortality.

------------------------------------------------------------------------

## 🎯 Problem Statement

Manual diagnosis of skin cancer is time-consuming and subjective. This
project aims to:

-   Automate melanoma detection\
-   Enable early diagnosis support\
-   Reduce diagnostic time\
-   Improve accessibility to screening tools\
-   Provide scalable AI-based healthcare assistance

Early detection improves prognosis, reduces mortality, and can lower
treatment costs. fileciteturn2file0

------------------------------------------------------------------------

## 🏗️ Technical Architecture

    User Upload → Streamlit UI → CNN / Transfer Learning Model → Prediction → Display Result

The system processes user-uploaded images through the trained model and
returns the predicted class via the web interface.

------------------------------------------------------------------------

## 📂 Dataset

-   **Source:** Kaggle --- Skin Cancer ISIC Dataset\
-   **Total Images:** \~2239 (original)\
-   **Balanced Dataset:** \~5850 (after augmentation)\
-   **Classes:** 9 skin lesion categories\
-   **Format:** JPG images

🔗 Dataset:\
https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python\
-   TensorFlow / Keras\
-   CNN & Transfer Learning (VGG16)\
-   Augmentor\
-   Streamlit\
-   NumPy, Pandas, Matplotlib\
-   PIL / OpenCV

------------------------------------------------------------------------

# ⚙️ Implementation Guide

------------------------------------------------------------------------

## 1️⃣ Environment Setup

### Prerequisites

-   Python 3.8+\
-   VS Code\
-   Kaggle Notebook (GPU recommended)\
-   Streamlit

### Install Dependencies

``` bash
pip install tensorflow streamlit augmentor numpy pandas matplotlib pillow
```

------------------------------------------------------------------------

## 2️⃣ Project Structure

    Melanoma_Detection/
    │
    ├── Dataset/
    ├── Model/
    ├── Training/
    ├── web.py
    └── README.md

The project uses a Streamlit script (`web.py`) for the web interface,
while the Training folder contains model-building code.
fileciteturn2file0

------------------------------------------------------------------------

## 3️⃣ Data Collection & Preparation

### Steps

1.  Download dataset from Kaggle\
2.  Import data into Kaggle notebook\
3.  Configure GPU acceleration\
4.  Read images using `image_dataset_from_directory()`\
5.  Extract class names\
6.  Visualize sample images

The dataset contains labeled dermoscopic images used for supervised
learning. fileciteturn2file0

------------------------------------------------------------------------

## 4️⃣ Exploratory Data Analysis (EDA)

### Performed Analysis

-   Dataset inspection\
-   Class distribution check\
-   Image visualization\
-   Batch inspection

EDA helps understand dataset imbalance and image characteristics.

------------------------------------------------------------------------

## 5️⃣ Data Splitting

Dataset split ratio:

    Train : Test : Validation = 8 : 1 : 1

This ensures proper generalization and evaluation.

------------------------------------------------------------------------

## 6️⃣ Data Optimization & Augmentation

Techniques applied:

-   Prefetching & caching\
-   Image resizing & rescaling\
-   Data augmentation\
-   Dataset balancing using Augmentor

After augmentation, the dataset becomes balanced (\~5850 images).
fileciteturn2file0

------------------------------------------------------------------------

## 7️⃣ Model Building

### Model Variants

**Model 1** - Sequential CNN\
- Baseline performance

**Model 2** - Enhanced augmentation\
- Regularization improvements

**Model 3 (Best)** - Transfer learning with VGG16\
- Balanced dataset\
- Learning rate scheduling

Transfer learning significantly improves performance.

------------------------------------------------------------------------

## 8️⃣ Multi-GPU Training

Training uses TensorFlow mirrored strategy for parallel GPU execution,
improving training speed and scalability. fileciteturn2file0

------------------------------------------------------------------------

## 9️⃣ Model Evaluation

Performance is evaluated using:

-   Training vs validation accuracy\
-   Training vs validation loss\
-   Visual performance plots

Expected plots are shown in the documentation.

------------------------------------------------------------------------

## 🔟 Model Saving

Best model is saved as:

``` python
model.save("Melanoma_Detection.h5")
```

This enables reuse during deployment. fileciteturn2file0

------------------------------------------------------------------------

# 🌐 Web Application (Streamlit)

The trained model is deployed via a Streamlit interface for real-time
predictions.

------------------------------------------------------------------------

## 1️⃣ Install Streamlit

``` bash
pip install streamlit
```

------------------------------------------------------------------------

## 2️⃣ web.py Features

-   Image upload via `st.file_uploader()`\
-   Image preprocessing\
-   Model inference\
-   Predicted class display\
-   Confidence score output

The UI prompts the user until an image is uploaded, then shows the
prediction. fileciteturn2file0

------------------------------------------------------------------------

## 3️⃣ Run the Application

``` bash
streamlit run web.py
```

After successful launch, the app runs on:

    http://localhost:8501

------------------------------------------------------------------------

## 🖥️ Application Flow

1.  User uploads skin lesion image\
2.  Image is resized and normalized\
3.  Model predicts lesion class\
4.  Result + confidence displayed\
5.  UI updates in real time

------------------------------------------------------------------------

## 📊 Model Classes

The model predicts one of **9 skin lesion categories** (e.g., melanoma,
nevus, etc.).

------------------------------------------------------------------------

## 🧪 How to Reproduce

``` bash
# 1. Download dataset
# 2. Train model (Training/)
# 3. Save best model to Model/
# 4. Install dependencies
# 5. Run Streamlit app
streamlit run web.py
```

⭐ If you found this project useful, consider giving it a star!
