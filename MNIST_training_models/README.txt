# MNIST Handwritten Digit Recognition

This project provides an interactive Streamlit web application that allows users to draw a digit and get a prediction using trained machine learning models. It also includes scripts for training models on the MNIST dataset.

## Project Structure

- `ModelTraining.py`: Trains multiple machine learning models on the MNIST dataset and saves the best-performing models.
- `MNIST model and streamlit.py`: Implements a Streamlit application to allow users to draw digits and get predictions from trained models.

## Installation

### Prerequisites
- Python 3.12
- Virtual environment (optional but recommended)
- Required Python packages

### Setup
1. Clone this repository:
   ```sh
   git clone <repo_url>
   cd <repo_folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```sh
   streamlit run "MNIST model and streamlit.py"
   ```

## Model Training

The `ModelTraining.py` script trains three different models on the MNIST dataset:
1. **Random Forest Classifier**
2. **Extra Trees Classifier**
3. **Support Vector Machine (SVM) Classifier**

### Training Process
- The MNIST dataset is fetched and split into training, validation, and test sets.
- Data is scaled using `StandardScaler` for SVM.
- Hyperparameter tuning is performed using `GridSearchCV`.
- Trained models are saved as compressed `.zip` files in the `SavedModels` directory.

To train models manually, run:
```sh
python ModelTraining.py
```

## Streamlit Application

The Streamlit app allows users to draw a digit on a canvas and predict the number using one of the trained models.

### Features
- Draw a digit using the Streamlit canvas.
- Select from trained models.
- View model prediction with probability distribution.

### Running the App
```sh
streamlit run "MNIST model and streamlit.py"
```

## Model Usage
The trained models are stored in `.zip` format. When selected in the app, they are extracted and loaded for inference.

### Supported Models
- **Random Forest** (`best_rf_model_non_scaled.pkl`)
- **Extra Trees** (`extra_trees_clf.pkl`)
- **SVM** (`svm_classifier.pkl`)
- **SVM** (`svm_classifier_pca.pkl`) => number of features reduced under training of this model

## Notes
- The SVM model was trained on only 10,000 samples for performance reasons.
- Digits 6 and 9 are sometimes misclassified.

## Author
Created by **Nike Olabiyi** (Version 1.0, February 2025).