# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:48:19 2023

@author: CMP
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the color CT scan image
image = cv2.imread('color_ct_scan_image.png')

# Convert the image from BGR to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Define the number of clusters (i.e., segments)
n_clusters = 3

# Fit the k-means model to the pixel data
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(pixels)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Reshape the labels array to the original image shape
segmented_image = labels.reshape(image.shape[:-1])

# Display the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(segmented_image, cmap='jet')
plt.title('Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()
