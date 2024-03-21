#Run this file first but make sure the cluster folder is present at the same directory as this file.
#This program is used to create a machine learning model which can be used o classify the image into a existing cluster.

#Open command prompt and install the libraries suing bellow commands

#pip install opencv-python
#pip install numpy
#pip install scikit-learn
#pip install joblib

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load

# Path to the folder containing unlabeled images
path = 'cluster'

# Number of clusters
c = 3

# Common image shape
img_shape = (100, 100)

# Read in images from folder and convert to numpy array
images = []
for file_name in os.listdir(path):
    img = cv2.imread(os.path.join(path, file_name))
    if img is not None:
        img_resized = cv2.resize(img, img_shape)
        images.append(img_resized)
images = np.array(images)

# Reshape images into one-dimensional vectors
images = images.reshape(images.shape[0], -1)

# Fit k-means clustering model
kmeans = KMeans(n_clusters=c, random_state=0).fit(images)

# Save the model
model_path = 'model.joblib'
dump(kmeans, model_path)
# Assign cluster labels to each image
labels = kmeans.predict(images)

# Print the number of images assigned to each cluster
for i in range(c):
    print(f"Number of images in cluster {i}: {sum(labels == i)}")

#0 - Graphics image without text.
#1 - Image which contains some text.
#2 - Traffic images.
