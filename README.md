# Chronic-Kidney-Disease-Diagnosis-Through-DenseNet-121Approach

This repository contains a deep learning framework designed for the automated classification of kidney stones, cysts, normal tissue, and tumors in medical imaging data. The system leverages the power of transfer learning, utilizing a pre-trained DenseNet121 model renowned for its effectiveness in image classification tasks.

## Dataset

The dataset used for training and evaluation can be found on Kaggle: [CT Kidney Dataset - Normal, Cyst, Tumor, and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

This dataset consists of medical images representing different classes of kidney conditions including normal tissue, cysts, tumors, and stones. It serves as the basis for training and testing the automated kidney stone classification model.

## Features

- Utilizes transfer learning with DenseNet121 for kidney stone classification.
- Custom layers for feature extraction tailored to medical imaging data.
- Implementation of advanced regularization techniques such as dropout and batch normalization for model robustness.
- High accuracy demonstrated through extensive experimentation and evaluation on diverse datasets.
- Valuable resource for healthcare professionals in nephrology and urology, aiding in accurate diagnosis and patient care improvement.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

## Usage

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) and extract it into the appropriate directory.
3. Train the model using `python train.py`.
4. Evaluate the model performance with `python evaluate.py`.
5. Integrate the trained model into your workflow for automated kidney stone classification.
