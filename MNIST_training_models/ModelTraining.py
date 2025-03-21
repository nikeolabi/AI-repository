# imports the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

import joblib

#######################################################
# Function to save the model and move it to SavedModels
#######################################################

def move_to_saved_folder_zip(model, filename):
    """
    Saves the model, compresses it into a ZIP file, and moves it to the 'SavedModels' folder.
    
    Args:
        model: The trained model to be saved.
        filename (str): The name of the file to save the model as (should end in .pkl).
    """
    # Ensure the SavedModels directory exists
    save_dir = "SavedModels"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model
    joblib.dump(model, filename)
    
    # Create ZIP file
    zip_filename = os.path.join(save_dir, filename.replace(".pkl", ".zip"))
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(filename)
    
    # Remove the original .pkl file after zipping
    os.remove(filename)
    
    print(f"Model saved and zipped to: {zip_filename}")

# fetches the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

# Data split
# First, split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then, split the 80% training data into 80% training and 20% validation (which is 16% of full data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(len(X_train))
print(len(X_val))
len(X_test)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

######################################################################
# Try non-linear SVM
######################################################################
# Define the parameter grid for tuning
svm_param_grid = {
    'C': [0.5, 1, 10],  # Regularization parameter
    'kernel': ['poly'],  # Types of kernels to try
    'gamma': [0, 1, 2]
}

# Initialize the SVM classifier
svm_model = SVC(C = 0.5, gamma = 1, kernel = 'poly', probability=True, random_state=42)

# Initialize GridSearchCV to search over the hyperparameter grid
grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
print("Best SVM Hyperparameters found by Grid Search:", grid_search.best_params_)

# Evaluate the best model on the test data
best_svm_model = grid_search.best_estimator_
test_accuracy = best_svm_model.score(X_val_scaled, y_val)
print(f"Test Accuracy of Best SVM classifier Model: {test_accuracy * 100:.4f}%")

# Save the model
move_to_saved_folder_zip(best_svm_model, 'svm_classifier.pkl')
# Save the scaler
joblib.dump(scaler, 'SavedModels/scaler.pkl')

######################################################################
# Try Random Forrest
######################################################################

# Define the parameter grid for tuning
rf_param_grid = {  
    'max_depth': [10, 20, None],      # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Min samples to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Min samples required in a leaf node
}

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV to search over the hyperparameter grid
grid_search_rf = GridSearchCV(rf_model, rf_param_grid, cv=5, verbose=3, n_jobs=-1)

# Fit the model on scaled data
grid_search_rf.fit(X_train_scaled, y_train)

# Get the best hyperparameters
print("Best Random Forest Hyperparameters:", grid_search_rf.best_params_)

# Evaluate the best model on the test data
best_rf_scaled_model = grid_search_rf.best_estimator_
val_accuracy = best_rf_scaled_model.score(X_val_scaled, y_val)
print(f"Test Accuracy of Best Random Forest Scaled Model: {val_accuracy * 100:.2f}%")


# Best Random Forest Hyperparameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2}
# Test Accuracy of Best Random Forest Model: 96.41%
# Save the model
move_to_saved_folder_zip(best_rf_scaled_model, 'best_rf_model_scaled.pkl')

# Fit the model on non-scaled data
grid_search_rf.fit(X_train, y_train)

# Get the best hyperparameters
print("Best Random Forest Hyperparameters:", grid_search_rf.best_params_)

# Evaluate the best model on the test data
best_rf_model = grid_search_rf.best_estimator_
val_accuracy = best_rf_model.score(X_val, y_val)
print(f"Test Accuracy of Best Random Forest Model: {val_accuracy * 100:.2f}%")

#Fitting 5 folds for each of 27 candidates, totalling 135 fits
#Best Random Forest Hyperparameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2}
#Test Accuracy of Best Random Forest Model: 96.42%

# Save the model
move_to_saved_folder_zip(best_rf_model, 'best_rf_model_non_scaled.pkl')

######################################################################
# End Random Forrest
######################################################################

######################################################################
# Try Extra Trees
######################################################################

extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
extra_trees_clf.fit(X_train, y_train)
val_accuracy = extra_trees_clf.score(X_val, y_val)
print(f"Test Accuracy of ExtraTreesClassifier Model: {val_accuracy * 100:.2f}%")
# Test Accuracy of ExtraTreesClassifier Model: 97.16%

# Test model on test data
extra_trees_clf_score = extra_trees_clf.score(X_test, y_test)
print("extra_trees_clf:", extra_trees_clf_score)

# same as above:
y_pred_test = extra_trees_clf.predict(X_test)
print(extra_trees_clf.__class__.__name__, accuracy_score(y_test, y_pred_test))

# extra_trees_clf: 0.9678571428571429
# ExtraTreesClassifier 0.9678571428571429

# Save the model
move_to_saved_folder_zip(extra_trees_clf, 'extra_trees_clf.pkl') 
######################################################################
# End Extra Trees
######################################################################

##################################################
# non-linear SVM with less features and samples
##################################################  

# Runs very slowly => need to reduce the number of features and samples
X = mnist.data[:10000]
y = mnist.target[:10000].astype(np.uint8)

# First, split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then, split the 80% training data into 80% training and 20% validation (which is 16% of full data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(len(X_train))
print(len(X_val))
len(X_test)

# Try PCA
# Scale only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test sets using the same scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply PCA (Fit on train, transform others)
pca = PCA(n_components=0.95)  # Preserve 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Define the parameter grid for tuning
svm_param_grid = {
    'C': [0.5, 1, 10],  # Regularization parameter
    'kernel': ['poly'],  # Types of kernels to try
    'gamma': [0, 1, 2]
}

# Initialize the SVM classifier
svm_model = SVC(probability=True, random_state=42)

# Initialize GridSearchCV to search over the hyperparameter grid
grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5)

# Fit the model
grid_search.fit(X_train_pca, y_train)

# Get the best hyperparameters
print("Best SVM Hyperparameters found by Grid Search:", grid_search.best_params_)

# Evaluate the best model on the test data
best_svm_model = grid_search.best_estimator_
test_accuracy = best_svm_model.score(X_val_pca, y_val)
print(f"Test Accuracy of Best SVM classifier Model: {test_accuracy * 100:.4f}%")
# Test Accuracy of Best SVM classifier Model: 95.0625%

move_to_saved_folder_zip(best_svm_model, 'svm_classifier_pca.pkl')
# Save the scaler
joblib.dump(pca, 'SavedModels/pca_scaler.pkl')