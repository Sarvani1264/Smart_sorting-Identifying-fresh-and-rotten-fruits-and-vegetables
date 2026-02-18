# Smart_sotring-Identifying-fresh-and-rotten-fruits-and-vegetables

# Project Overview

Smart Sorting is a deep learning–based web application that identifies whether a fruit or vegetable is fresh or rotten from an uploaded image.

The model was trained using Convolutional Neural Networks (CNN) and deployed through a Flask web application for real-time predictions.

This project demonstrates the integration of machine learning with web development for practical quality inspection.

# Model Training Environment (Google Colab)

The model was trained using Google Colab.

Libraries used during training:

TensorFlow / Keras – Model building and training

NumPy – Numerical operations

Pandas – Dataset handling

Matplotlib – Accuracy and loss visualization

Scikit-learn – Data splitting and evaluation metrics

# Application Development Environment (VS Code)

The trained model (.h5 file) was integrated into a Flask web application and developed locally using VS Code.

Libraries required to run the web application:

Flask – Backend framework

TensorFlow – Load and run the trained model

NumPy – Image array processing

Pillow – Image handling

OpenCV – Image preprocessing

Python-dotenv – Environment variable management

All required libraries are listed in the requirements.txt file.

# Dataset Information

The dataset used for training is large in size and cannot be uploaded to this repository due to GitHub file size limitations.

You can download the dataset from Kaggle using the link below:

🔗 Dataset Link:
https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

After downloading:

Extract the dataset.

Place it inside the project directory.

Run the training notebook if needed.

# Features

✔ Upload image through web interface
✔ Automatic image preprocessing
✔ Real-time freshness prediction
✔ Clean and simple user interface
