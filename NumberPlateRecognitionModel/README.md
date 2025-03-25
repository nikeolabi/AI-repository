# Automatic Number Plate Detection and Recognition

## Description
This project consists of two main components: number plate **detection** and **character/number recognition** using different AI models. It aims to localize license plates in vehicle images and extract the textual content from them.

## Components
- `NumberPlateRecognitionModel.ipynb`: Uses TensorFlow Object Detection API with the pretrained SSD MobileNet v2 FPNLite 320x320 (version 17). In this part the model is trained to recognize number plates on a car image from Kaggle dataset.
- `PlateDetection.py`: Responsible for detecting and cropping number plates from images and recognizing the characters and digits using EasyOCR.


## Technologies Used
- Python
- OpenCV
- Tesseract OCR / Keras
- NumPy, Matplotlib, etc.

## Usage
Run the notebooks in the following order:

1. `NumberPlateRecognitionModel.ipynb` – detects license plates from vehicle images.
2.  `PlateDetection.py– recognizes characters from the detected plates.

## SSD Model limitations:
It was trained on a small dataset (692 car images).
The dataset primarily represents license plates from the USA and India (sourced from Kaggle).
Thus, number plates from other counties are not easily detected by te SSD model.