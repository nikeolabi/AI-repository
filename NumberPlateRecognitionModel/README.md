# Automatic Number Plate Detection and Recognition

## Description
This project consists of two main components: number plate **detection** and **character/number recognition** using different AI models. It aims to localize license plates in vehicle images and extract the textual content from them.

## Components
- `NumberPlateRecognitionModel.ipynb`: Uses TensorFlow Object Detection API with the pretrained SSD MobileNet v2 FPNLite 320x320 (version 17). In this part the model is trained to recognize number plates on car images from Kaggle dataset. This part is not mandatory to run since the model is already trained and saved. The trained model is saved in Model folder. It is then used in `PlateDetection.py`.
- `PlateDetection.py`: Responsible for detecting and cropping number plates from images and recognizing the characters and digits using EasyOCR.
- The object_detection package is part of the TensorFlow Object Detection API and must be available for the script to function properly. To avoid cloning the entire API repository, the necessary parts of the package were copied directly into the project root for convenience.

## Technologies Used
- Python
- OpenCV
- Tesseract OCR / Keras
- NumPy, Matplotlib, etc.

## Usage

### Prerequisits: Python 3.8.0

Run the notebook and `PlateDetection.py` in the following order:

1. `NumberPlateRecognitionModel.ipynb` – training of the model Step 0 to 9. You need to open the file in Jupyter Notebook   
    to see the Steps. All paths must be changes to the actual ones. This part is OPTIONAL.
2.  `PlateDetection.py` – recognizes characters from the detected plates. Can be opened in VSCode and run there. The demo 
    images to recognize are available in folder `DemoPictures`.
    
### Known issues: 
When making IMPORTS in `PlateDetection.py` an error will appear:
- ImportError: cannot import name 'builder' from 'google.protobuf.internal' (C:\Users\path\RadarsInMovement\RadarsInMovEnv\lib\site-packages\google\protobuf\internal__init__.py)
- The builder is missing and must by copied into your protobuf\internal
- The missing file can be found in this Git repository in ROOT folder:
- Solution:
   - copy it from the root folder into your protobuf\internal
   - OR install later version and copy builder file into Internal
      - pip install protobuf==4.21.1 and cp .\builder.py into your protobuf\internal
- https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal


## SSD Model limitations:
- It was trained on a small dataset (692 car images).
- The dataset primarily represents license plates from the USA and India (sourced from Kaggle).
- Thus, number plates from other counties are not easily detected by the trained SSD model.
