import joblib
from argparse import ArgumentParser
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import image
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tempfile import mkdtemp
from joblib import Memory
from sklearn.pipeline import Pipeline

#Function to use Patch Extractor on input images
def patch(n_pixels):
    
    patch_extractor = image.PatchExtractor(patch_size=(n_pixels, n_pixels), max_patches=1, random_state=0)
    sub_images = patch_extractor.transform(images)

    return sub_images

# Function to rotate images randomly
def rotate_images(image1):

    rotated_images = []
    rotated_labels = []

    for i in range(len(image1)):
        #Random rotation
        num_rotations = np.random.choice([0, 1, 2, 3])

        rotated_image = np.rot90(image1[i], k=num_rotations)
        rotated_images.append(rotated_image.flatten())
        rotated_labels.append(num_rotations)

    #Return numpy arrays
    return np.array(rotated_images).reshape(len(images),-1), np.array(rotated_labels).ravel()

#SVM Function
def svm_classifier(X,y,pixels):

    #enhancing the efficiency of memory-intensive tasks
    cachedir = mkdtemp()
    memory = Memory(cachedir, verbose=2)
    X = X.reshape((len(X), -1))

    #Input data split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #Pipeline to include Scaler, PCA and Hyper-Parameters
    if pixels == 90:
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=50)),
                ('svc', SVC(kernel='rbf', C=2, gamma='scale')),
            ],memory=memory)
        
        pipeline.fit(X_train, y_train)

        return pipeline
    
    elif pixels == 50:
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=40)),
                ('svc', SVC(kernel='rbf', C=2, gamma='scale')),
            ],memory=memory)
        
        pipeline.fit(X_train, y_train)

        return pipeline
    
    elif pixels == 30:
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=40)),
                ('svc', SVC(kernel='rbf', C=2, gamma='scale')),
            ],memory=memory)
        
        pipeline.fit(X_train, y_train)

        return pipeline

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("train_data_file_name", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument("n_pixels", type=int)
    args = parser.parse_args()

    #Passing command line arguments into variables
    train_file_name = args.train_data_file_name
    model_file = args.model_file
    pixels = args.n_pixels

    #Loading the input file
    images, person_id = joblib.load(train_file_name)

    #Call Patch Extractor
    sub_images = patch(pixels)

    #Call Rotate Image function 
    rotated_images, rotated_labels = rotate_images(sub_images)
    #Call SVM Function
    model = svm_classifier(rotated_images, rotated_labels,pixels)

    #Dump the classified SVM model into model.joblib 
    joblib.dump(model,model_file,compress=6)     
    print("Dump Successful")

#C:\Sheffield\DSP\Assignment 2\train.py