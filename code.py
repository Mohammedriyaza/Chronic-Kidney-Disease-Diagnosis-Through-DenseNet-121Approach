
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from tkinter import filedialog
import random

# Function to visualize feature maps
# Function to visualize feature maps and filters
def visualize_feature_maps(model, img_path, layer_names):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (65, 65))  # Resize image to match model input shape

    # Preprocess the image
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Get the convolutional layers
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]

    # Plot feature maps and filters
    for layer_name in layer_names:
        for layer in conv_layers:
            if layer.name == layer_name:
                # Display feature maps
                feature_map_model = Model(inputs=model.input, outputs=layer.output)
                feature_maps = feature_map_model.predict(img_resized)
                plt.figure(figsize=(8, 8))
                plt.title(layer_name + ' - Feature Maps')
                for i in range(feature_maps.shape[-1]):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
                    plt.axis('off')
                plt.show()

                # Display filters
                filters = layer.get_weights()[0]
                plt.figure(figsize=(8, 8))
                plt.title(layer_name + ' - Filters')
                for i in range(filters.shape[-1]):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(filters[:, :, 0, i], cmap='gray')
                    plt.axis('off')
                plt.show()


# Input Data
path = 'data/'
categories = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Display sample images
for category in categories:
    fig, _ = plt.subplots(3, 4)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path + category)[:12]):
        img = plt.imread(path + category + '/' + v)
        plt.subplot(3, 4, k + 1)
        plt.axis('off')
        plt.imshow(img)
        cv2.imshow(' Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    plt.show()

# Load and segment tumor image
image = cv2.imread('data/Tumor/Tumor- (1).jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape(-1, 3)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(pixels)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
segmented_image = labels.reshape(image.shape[:-1])

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

# Load data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 65
WIDTH = 65
N_CHANNELS = 3

for category in categories:
    for f in os.listdir(path + category):
        imagePaths.append([path + category + '/' + f, categories.index(category)])

random.shuffle(imagePaths)
print(imagePaths[:10])

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))
    data.append(image)
    labels.append(imagePath[1])

# Convert lists to numpy arrays
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Display sample input images
fig, _ = plt.subplots(4, 5)
fig.suptitle("Sample Input")
fig.patch.set_facecolor('xkcd:white')
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(data[i])
    plt.axis('off')
    cv2.imshow(' Image', data[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
plt.show()

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = to_categorical(trainY, 4)

# Build Densenet121 model
def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)

    input = Input(shape=(HEIGHT, WIDTH, N_CHANNELS))
    x = Conv2D(3, (3, 3), padding='same')(input)
    x = densenet(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='softmax', name='root')(x)
 
    model = Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model

# Train the model
model = build_densenet()
model.fit(trainX, trainY, batch_size=32, epochs=1, verbose=1)

# Plot training history
history = model.history.history
train_loss = history['loss']
train_acc = history['accuracy']

plt.figure()
plt.plot(train_loss, label='Loss')
plt.plot(train_acc, label='Accuracy')
plt.title('Performance Plot')
plt.legend()
plt.show()

# Evaluate the model
print("Accuracy of the DBN is:", model.evaluate(trainX, trainY)[1] * 100, "%")
pred = model.predict(testX)
predictions = argmax(pred, axis=1) 
print('Classification Report')
cr = classification_report(testY, predictions, target_names=categories)
print(cr)
print('Confusion Matrix')
cm = confusion_matrix(testY, predictions)

# Plot Confusion Matrix
plt.figure()
plot_confusion_matrix(cm, figsize=(15, 15), class_names=categories, show_normed=True)
plt.title("Model confusion matrix")
plt.style.use("ggplot")
plt.show()

# Prediction
test_data = []
Image = filedialog.askopenfilename()
head_tail = os.path.split(Image)
fileNo = head_tail[1].split('.')
test_image_o = cv2.imread(Image)
test_image = cv2.resize(test_image_o, (WIDTH, HEIGHT))

test_data = np.array(test_image, dtype="float") / 255.0
test_data = test_data.reshape([-1, 65, 65, 3])
pred = model.predict(test_data)
predictions = argmax(pred, axis=1) # return to label
print ('Prediction : ' + categories[predictions[0]])

# Imersing into the plot
fig = plt.figure()
fig.patch.set_facecolor('xkcd:white')
plt.title(categories[predictions[0]])
plt.imshow(test_image_o)
plt.show()

# Visualize feature maps
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]  # Get all convolutional layers
img_path = Image  # Path to the image
visualize_feature_maps(model, img_path, layer_names)
