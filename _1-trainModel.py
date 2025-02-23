import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from huggingface_hub import HfApi, create_repo, upload_file, login
from datasets import load_dataset
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Load dataset from hugging face
login(token="YOUR LOGIN TOKEN")
dataset = load_dataset("Seb0099/autopilot-dataset-pictures", split="train") 

# Define the local directory for storing images
train_data_dir = "downloaded_images"

# Ensure directory structure exists
os.makedirs(os.path.join(train_data_dir, "left"), exist_ok=True)
os.makedirs(os.path.join(train_data_dir, "center"), exist_ok=True)
os.makedirs(os.path.join(train_data_dir, "right"), exist_ok=True)

# Function to save images to the correct folder
def save_image(example, index):
    image = example["image"]
    label = example["folder"]
    
    if isinstance(label, str):
        label_name = label.lower() 
    
    if label_name in ["center", "left", "right"]:
        image_path = os.path.join(train_data_dir, label_name, f"{index}.jpg")
        image.save(image_path)

# Download and save images
for i, imgItem in enumerate(dataset):
    save_image(imgItem, i)

print("Images successfully downloaded and organized.")

# Set hyperparameters
batch_size = 32
img_height = 180
img_width = 320
epochs = 30  # Adjust as needed

early_stopping = EarlyStopping(
    monitor='val_loss',  # Watch validation loss
    patience=5,  # Stop if no improvement after 5 epochs
    restore_best_weights=True
)# Create data generators with augmentation

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.3
)

# Training Data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,  # Updated to the downloaded folder
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'binary' for two classes
    shuffle=True,
    classes=['center', 'left', 'right'],
    subset="training"
)

# Validation Data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=True,
    classes=['center', 'left', 'right'],
    subset="validation"
)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # 3 output neurons for left, center, right
])

print(train_generator.class_indices)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Matches categorical labels
              metrics=['accuracy'])

# Train the model with validation
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    epochs=epochs,
    callbacks=[early_stopping]  # Apply early stopping
)

# Save the trained model
model.save('image_classifier.keras')

# #convert model to tensorflow light
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# # Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("Model training and conversion to TFLite complete!")