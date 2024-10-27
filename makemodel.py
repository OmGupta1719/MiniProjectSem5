import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers, models
import matplotlib.pyplot as plt

# Parameters
img_size = 224  # Image size set to 224x224
dataset_path = r'C:\Users\gupta\Downloads\fuckit\nothing'  # Change to your dataset path

# Load and preprocess data
def load_data():
    images = []
    labels = []
    class_names = os.listdir(dataset_path)  # Get all class names (A-Z)

    # Sort class names to ensure consistent labeling
    class_names.sort()

    for label in class_names:
        label_path = os.path.join(dataset_path, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(class_names.index(label))  # Use index as label

    return np.array(images), np.array(labels), class_names

# Load data
images, targets, class_names = load_data()
images = images.astype('float32') / 255.0  # Normalize images

# Split the data
X_train, X_val, y_train, y_val = train_test_split(images, targets, test_size=0.2, random_state=42)

# Create model
def create_model(num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer
    return model

# Define and compile the model
model = create_model(num_classes=len(np.unique(targets)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 3
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

# Save the model
model.save("model.h5")

# Create and save labels.txt
with open("Model/labels.txt", "w") as f:
    for label in class_names:
        f.write(f"{label}\n")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
