import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Define the paths to your dataset folders
train_data_dir = 'Images/'  # e.g., 'train/'

# Set hyperparameters
batch_size = 32
img_height = 180
img_width = 320
epochs = 10  # You can adjust this number as needed

# Create data generators with data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Set to 'categorical' for multiple classes
    shuffle=True,  # Shuffle the data for better training
    classes=['left', 'right']
    # classes=['left', 'center', 'right']  # Specify the class names
)

# Get a batch of images and labels
images, labels = next(train_generator)

# Display one image from the batch
plt.figure(figsize=(6, 6))
plt.imshow(images[0])
plt.title('Augmented Image')
plt.axis('off')
plt.show()

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
    Dense(2, activation='softmax')  # 3 output neurons for three categories (left, center, right)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical crossentropy for multiple classes
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs)

# Save the trained model
model.save('image_classifier.keras')

# #convert model to tensorflow light
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# # Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

#Test
# Load the image you want to classify
# image_path = 'toto.jpg'  # Replace with the actual path to your image
# model = tf.keras.models.load_model('image_classifier.h5')
# img = image.load_img(image_path, target_size=(img_height, img_width))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# # Preprocess the image (same preprocessing as during training)
# img_array /= 255.0  # Normalize pixel values (assuming your training data was normalized)

# # Make predictions
# predictions = model.predict(img_array)

# Convert the class probabilities to class labels
# class_labels = ['left', 'center', 'right']  # Define your class labels
# predicted_class_index = np.argmax(predictions)
# predicted_class_label = class_labels[predicted_class_index]

# Display the prediction result
# print(f"Predicted class: {predicted_class_label}")
# print(f"Class probabilities: {predictions[0]}")
# print(predictions[0][0])

# # Decode the predictions (if you used one-hot encoding)
# if predictions[0][0] > 0.5:
#     result = "Dog"
# else:
#     result = "Cat"

# # Print the predicted class and confidence score
# print(predictions[0][0]);
# print(predictions[0][0]);
# print(f'Predicted class: {result}')
# print(f'Confidence score: {predictions[0][0] * 100:.2f}%')



