
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import utils

print("Tensorflow version used: ", tf.__version__)
os.system("pause")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Download and load the Fashion MINST Dataset
print("Downloading dataset...")
fashion_mnist = keras.datasets.fashion_mnist
(x_images, x_labels), (y_images, y_labels) = fashion_mnist.load_data()
os.system("pause")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print('Classes: ', class_names)

train_images = x_images.reshape(x_images.shape[0], 1, 28, 28)
test_images = y_images.reshape(y_images.shape[0], 1, 28, 28)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255
train_labels = utils.to_categorical(x_labels, 10)
test_labels = utils.to_categorical(y_labels, 10)

# Explore the data
print("Shape of Training dataset: ", train_images.shape)
print("Length of Training dataset: ", len(train_labels))
print("Training labels:", train_labels)
print("Shape of Test dataset: ", test_images.shape)
print("Length of Test dataset: ", len(test_labels))
os.system("pause")


# Building the model
print("Setting up the layers: ")
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Model reconstruction from JSON file
with open('trained models/model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('trained models/model_weights.h5')

print("Compiling the model: ")
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
os.system("pause")

##Training the model
#print("Training the model: ")
#model.fit(train_images, train_labels, epochs=10)
#os.system("pause")

## Save the weights
#print("Saving.........")
#model.save_weights('trained models/model_weights.h5')

## Save the model architecture
#with open('trained models/model_architecture.json', 'w') as f:
#    f.write(model.to_json())

# Evaluating the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Make Predictions
predictions = model.predict(test_images)
for i in range(5):
    print('Making predictions', predictions[i])
    print('Predicted value: ', class_names[np.argmax(predictions[i])])
    print('True Value: ', class_names[np.argmax(test_labels[i])])
    print('\n')
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
    plt.imshow(y_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[true_label]), color=color)
plt.show()
