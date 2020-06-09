# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#from keras.models import load_model
from keras.models import model_from_json

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

print("Tensorflow version used: ", tf.__version__)
#os.system("pause")

# Download and load the Fashion MINST Dataset
print("Downloading dataset...")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#os.system("pause")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print('Classes: ', class_names)

# Explore the data
print("Shape of Training dataset: ", train_images.shape)
print("Length of Training dataset: ", len(train_labels))
print("Training labels:", train_labels)
print("Shape of Test dataset: ", test_images.shape)
print("Length of Test dataset: ", len(test_labels))
os.system("pause")

# Processing the data
print("First image in the training dataset:")
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()
train_images = train_images / 255.0
test_images = test_images / 255.0
os.system("pause")
print("Displaying first 25 images in the training dataset: ")
# %matplotlib inline
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Building the model
print("Setting up the layers: ")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## Model reconstruction from JSON file
#with open('trained models/model_architecture.json', 'r') as f:
#    model = model_from_json(f.read())

## Load weights into the new model
#model.load_weights('trained models/model_weights.h5')

print("Compiling the model: ")
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
os.system("pause")

print("Training the model: ")
model.fit(train_images, train_labels, epochs=10)
os.system("pause")

# Save the weights
print("Saving.........")
model.save_weights('trained models/model_weights.h5')

# Save the model architecture
with open('trained models/model_architecture.json', 'w') as f:
    f.write(model.to_json())

# Evaluating the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Make Predictions
predictions = model.predict(test_images)
for i in range(5):
    print('Making predictions', predictions[i])
    print('Predicted value: ', class_names[np.argmax(predictions[i])])
    print('True Value: ', class_names[test_labels[i]])
    print('\n');
os.system("pause")

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
print("Plotting predictions: ")
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[true_label]), color=color)
plt.show()
os.system("pause")
print("Grabbing test image: ")
# Grab an image from the test dataset
img = test_images[0]
print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)
predictions = model.predict(img)
print("Printing predictions: ", predictions)
prediction = predictions[0]
print(np.argmax(prediction))