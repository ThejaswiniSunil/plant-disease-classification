# 🥔 Potato Disease Classification — End-to-End AI & Edge Deployment

> **An end-to-end computer vision system for potato leaf disease classification, extended from model training and cloud deployment to ONNX conversion and hardware-accelerated Edge AI inference on Qualcomm Snapdragon X Elite.**

This project started as a deep learning application for identifying potato plant diseases from leaf images and evolved into a broader **ML deployment and Edge AI engineering project**.

The system covers the complete lifecycle of an AI model:

**Dataset → Model Training → Model Validation → API → Web/Mobile Applications → Cloud Deployment → TensorFlow Lite → ONNX → Snapdragon NPU Acceleration → Performance Benchmarking**

---

## 🚀 Project Overview

The system uses a deep learning image-classification model to identify the health condition of potato leaves.

Given an image of a potato leaf, the trained model predicts the corresponding disease category and returns the prediction through an API that can be consumed by web and mobile applications.

The project was subsequently extended to investigate **on-device AI inference and hardware acceleration**, converting the trained TensorFlow/Keras model to ONNX and compiling it for **Qualcomm Snapdragon X Elite** using Qualcomm AI Hub.

This allowed the same trained model to be evaluated across:

* Traditional server-side inference
* TensorFlow Serving
* TensorFlow Lite
* ONNX inference
* Qualcomm Snapdragon CPU
* Qualcomm Snapdragon NPU

---

# 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │   Potato Leaf Image │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ TensorFlow / Keras  │
                    │ Disease Classifier  │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
          FastAPI API    TensorFlow Lite    ONNX
                │                             │
                ▼                             ▼
        Web / Mobile App             Qualcomm AI Hub
                                              │
                                              ▼
                                    Snapdragon X Elite
                                              │
                                     ┌────────┴────────┐
                                     ▼                 ▼
                                    CPU               NPU
```

---

# 🎯 Objectives

The project was developed with several goals:

* Build an end-to-end computer vision classification system
* Train a deep learning model for plant disease recognition
* Expose model inference through a REST API
* Build web and mobile interfaces for predictions
* Explore cloud-based model serving
* Convert the model for lightweight deployment
* Investigate cross-platform model portability using ONNX
* Deploy and profile the model on dedicated Edge AI hardware
* Compare CPU and NPU inference performance

---

# 📊 Dataset

The model was trained using the **PlantVillage dataset**.

The dataset contains labeled plant leaf images covering multiple crop and disease categories.

For this project, only the potato-related classes were retained.

### Potato Classes

The classification task covers:

* 🥔 Potato — Early Blight
* 🥔 Potato — Late Blight
* 🌿 Potato — Healthy

The dataset was prepared and processed before being used for model training.

---

# 🤖 Deep Learning Model

The original classifier was developed using:

* **Python**
* **TensorFlow**
* **Keras**
* Image preprocessing and normalization
* CNN-based image classification

The trained model was initially saved as a TensorFlow/Keras `.h5` model.

```text
Input Image
     ↓
Image Preprocessing
     ↓
CNN Feature Extraction
     ↓
Classification Layer
     ↓
Disease Probabilities
     ↓
Predicted Disease
```

The model can therefore be used independently of the application layer and deployed through multiple inference backends.

---

# 🔌 Backend API

A **FastAPI** backend provides an interface between the trained model and the client applications.

### Request Flow

```text
Client
  │
  │ Image
  ▼
FastAPI
  │
  ▼
Model Inference
  │
  ▼
Prediction
  │
  └──► Disease + Confidence
```

The API can be run locally using:

```bash
cd api

uvicorn main:app --reload --host 0.0.0.0
```

The API is then available on:

```text
http://localhost:8000
```

---

# 🌐 Web Application

A React-based frontend provides a user interface for interacting with the disease classification API.

### Technologies

* ReactJS
* JavaScript
* REST API
* NPM

Install dependencies:

```bash
cd frontend
npm install
```

Configure the API endpoint through:

```text
.env
```

Then start the application:

```bash
npm run start
```

---

# 📱 Mobile Application

The project also includes a React Native application for mobile-based disease classification.

```bash
cd mobile-app
yarn install
```

Configure the API endpoint in:

```text
.env
```

Run on Android:

```bash
npm run android
```

or iOS:

```bash
npm run ios
```

This creates a mobile interface capable of sending plant images to the backend for inference.

---

# ☁️ Cloud Deployment

The project was also designed to support cloud-based model serving.

### TensorFlow Serving

TensorFlow Serving can be used as the model-serving layer while FastAPI acts as the application/API layer.

```text
Client
   ↓
FastAPI
   ↓
TensorFlow Serving
   ↓
TensorFlow Model
   ↓
Prediction
```

TensorFlow Serving can be started using Docker with the appropriate model configuration.

---

# 📱 TensorFlow Lite

To investigate lightweight inference, the TensorFlow model can also be converted into a **TensorFlow Lite model**.

The conversion workflow is provided in:

```text
training/tf-lite-converter.ipynb
```

The resulting model can be used for more resource-constrained deployment scenarios, including mobile and edge environments.

---

# ⚡ Edge AI Extension

## From Cloud Inference to Hardware-Accelerated AI

The project was extended beyond conventional server-side inference to investigate **Edge AI deployment and hardware acceleration**.

The trained TensorFlow/Keras model was converted into **ONNX**, allowing it to be used with a broader range of deployment and hardware acceleration frameworks.

### Edge AI Pipeline

```text
TensorFlow / Keras
        │
        ▼
   Trained .h5 Model
        │
        ▼
   ONNX Conversion
        │
        ▼
 Qualcomm AI Hub
        │
        ▼
 Snapdragon X Elite
        │
   ┌────┴────┐
   ▼         ▼
 CPU        NPU
```

---

# 🔄 ONNX Model Conversion

The original Keras model was converted to ONNX to improve deployment portability.

During conversion, several real-world compatibility issues were encountered, including:

* Keras/TensorFlow version compatibility
* Model compilation configuration
* Tensor naming compatibility
* SavedModel interoperability

The final conversion pipeline used TensorFlow's SavedModel representation as an intermediate format before producing the ONNX model.

The resulting ONNX model was then independently tested to verify that inference remained valid after conversion.

### Conversion Validation

The converted model was tested using sample inputs and produced valid class probability distributions, confirming that the conversion did not silently break model inference.

---

# 🧩 Qualcomm AI Hub Deployment

The ONNX model was compiled for **Qualcomm Snapdragon X Elite** using **Qualcomm AI Hub Workbench**.

This enabled profiling on Snapdragon hardware without requiring a dedicated physical development device.

The deployment process included:

1. Preparing the ONNX model
2. Defining the model input specification
3. Compiling the model for Snapdragon X Elite
4. Profiling the compiled model
5. Evaluating processor utilization
6. Comparing NPU and CPU execution

---

# ⚙️ NPU vs CPU Benchmark

One of the key outcomes of the Edge AI extension was a direct comparison between CPU inference and NPU-accelerated inference.

### Measured Benchmark

| Metric                   | Snapdragon NPU |     CPU |
| ------------------------ | -------------: | ------: |
| Median inference latency |        ~0.2 ms | ~2.3 ms |
| Reported memory usage    |          ~2 MB |  ~35 MB |
| Model execution          |       100% NPU |     CPU |

### Observed Difference

In this specific benchmark:

* NPU inference showed approximately **5–7× lower latency**
* Reported memory usage was approximately **17× lower**
* The profiling run executed **100% of the model on the NPU**

> **Important:** These numbers represent the measured benchmark configuration and should not be interpreted as universal performance characteristics for all models or Snapdragon devices.

---

# 🔬 Why the Edge AI Extension Matters

The Edge AI portion demonstrates more than simply converting a model between formats.

It covers several practical ML engineering concepts:

### Model Portability

Moving a trained model from TensorFlow/Keras into the ONNX ecosystem.

### Hardware-Aware Deployment

Compiling and preparing the model for a specific AI accelerator.

### NPU Acceleration

Using dedicated neural processing hardware instead of relying exclusively on the CPU.

### Performance Profiling

Measuring real inference latency and memory consumption rather than relying only on theoretical specifications.

### Deployment Optimization

Understanding the differences between:

```text
Cloud / Server Inference
        ↓
Mobile / Lightweight Inference
        ↓
Edge CPU Inference
        ↓
Dedicated NPU Inference
```

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* TensorFlow
* Keras
* Convolutional Neural Networks
* Image Classification

### Model Deployment

* TensorFlow Serving
* TensorFlow Lite
* ONNX
* Qualcomm AI Hub Workbench

### Backend

* FastAPI
* Uvicorn
* REST APIs

### Frontend

* ReactJS
* JavaScript
* NPM

### Mobile

* React Native
* Android
* iOS

### Cloud

* Google Cloud Platform
* Google Cloud Storage
* Google Cloud Functions
* Docker

### Edge AI

* Qualcomm Snapdragon X Elite
* NPU acceleration
* CPU inference
* Hardware profiling
* Latency benchmarking

---

# 📁 Project Structure

```text
plant-disease-classification/
│
├── api/
│   ├── main.py
│   ├── main-tf-serving.py
│   ├── requirements.txt
│   └── models.config.example
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── mobile-app/
│   ├── android/
│   ├── ios/
│   └── package.json
│
├── training/
│   ├── potato-disease-training.ipynb
│   ├── tf-lite-converter.ipynb
│   └── requirements.txt
│
├── models/
│   └── ...
│
├── tf-lite-models/
│   └── ...
│
├── gcp/
│   └── ...
│
└── README.md
```

---

# 🧪 Training the Model

Download the PlantVillage dataset:

```text
https://www.kaggle.com/arjuntejaswi/plant-village
```

Keep only the potato-related classes.

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
training/potato-disease-training.ipynb
```

Update the dataset path and execute the notebook.

The resulting model can then be stored under:

```text
models/
```

with an appropriate version number.

---

# 🚀 Installation

## Python

Install the required dependencies:

```bash
pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt
```

## React

```bash
cd frontend
npm install
```

Configure:

```text
.env
```

with the appropriate API URL.

## React Native

```bash
cd mobile-app
yarn install
```

For iOS:

```bash
cd ios
pod install
cd ../
```

---

# 🔮 Future Improvements

Potential future extensions include:

* Model quantization
* INT8 inference
* Further ONNX Runtime optimization
* Snapdragon GPU vs NPU benchmarking
* Additional mobile-device benchmarking
* Batch vs single-image latency analysis
* Power/energy profiling
* Model compression
* Larger and more diverse real-world datasets
* Additional crop and disease categories
* Continuous model monitoring
* Containerized production deployment
* CI/CD for model deployment

---

# 📈 Project Evolution

This project evolved through multiple stages:

```text
                    PHASE 1
               Deep Learning
                     │
                     ▼
             Potato Disease
              Classification
                     │
                     ▼
                    PHASE 2
               Application Layer
                     │
             ┌───────┴───────┐
             ▼               ▼
          FastAPI          React
             │
             ▼
       React Native
                     │
                     ▼
                    PHASE 3
              Cloud Deployment
                     │
             TensorFlow Serving
                     │
                     ▼
               TensorFlow Lite
                     │
                     ▼
                    PHASE 4
                 Edge AI
                     │
                  ONNX
                     │
                     ▼
             Qualcomm AI Hub
                     │
                     ▼
            Snapdragon X Elite
                     │
              ┌──────┴──────┐
              ▼             ▼
             CPU           NPU
              │             │
              └──────┬──────┘
                     ▼
              Benchmarking
```

---

# 💡 Key Engineering Takeaways

This project provided hands-on experience across the complete AI deployment lifecycle:

**1. Model Development**

Training and validating a computer vision model using TensorFlow/Keras.

**2. Model Serving**

Exposing ML inference through a FastAPI backend and TensorFlow Serving.

**3. Application Integration**

Connecting machine learning inference to web and mobile applications.

**4. Cloud Deployment**

Exploring deployment of ML inference components through Google Cloud.

**5. Model Portability**

Converting a TensorFlow/Keras model into ONNX for cross-platform deployment.

**6. Edge AI**

Compiling and profiling the model on Qualcomm Snapdragon X Elite hardware.

**7. Hardware Acceleration**

Comparing CPU execution with dedicated NPU inference.

**8. Performance Engineering**

Using measured latency and memory results to evaluate deployment efficiency.

---

# 👩‍💻 Author

**Thejaswini Sunil**

Computer Science | AI / Machine Learning | Computer Vision | Edge AI

[GitHub](https://github.com/ThejaswiniSunil)

---

## ⭐ Project Summary

> **A full-stack computer vision project that evolved from a plant disease classifier into an end-to-end AI deployment experiment spanning deep learning, API development, web/mobile applications, cloud serving, model conversion, and hardware-accelerated Edge AI inference.**

**TensorFlow → FastAPI → React → React Native → GCP → TensorFlow Lite → ONNX → Qualcomm AI Hub → Snapdragon NPU**

