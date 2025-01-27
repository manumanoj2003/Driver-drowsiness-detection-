import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

# Load dataset from the 'dataset' folder
def load_dataset(dataset_path):
    data = []
    labels = []

    for label, folder in enumerate(["open", "closed"]):
        folder_path = os.path.join(dataset_path, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Convert image to grayscale and resize to (64, 64)
                img = Image.open(img_path).convert("L")
                img = img.resize((64, 64))
                data.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels)

# Prepare the dataset
print("Loading dataset...")
data, labels = load_dataset("dataset")
data = data / 255.0  # Normalize pixel values to the range [0, 1]
data = np.expand_dims(data, axis=-1)  # Add channel dimension (for grayscale)
labels = to_categorical(labels, num_classes=2)  # Convert labels to one-hot encoding

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Create the model
print("Building the model...")
model = create_model()

# Data Augmentation
print("Applying data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(X_train)

# Train the model
print("Training the model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=15,
    verbose=1
)

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/drowsiness_model.h5")
print("Model saved to models/drowsiness_model.h5")

# Evaluate the model
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
