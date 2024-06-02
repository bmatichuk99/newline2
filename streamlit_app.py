import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

# Function to create and compile the model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the MNIST data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels

# Function for data augmentation
def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(train_images)
    return datagen

# Train the model with data augmentation
def train_model(model, train_images, train_labels, epochs=5, batch_size=64):
    datagen = augment_data(train_images, train_labels)
    model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size), epochs=epochs)
    model.save('mnist_model.keras')
    return model

# Predict digit from drawn image
def predict_digit(model, image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = 255 - image
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    return np.argmax(prediction), image

st.title('MNIST Digit Recognizer')

# Load or create the model
if 'model' not in st.session_state:
    model = create_model()
    st.session_state['model'] = model
else:
    model = st.session_state['model']

# User interface for training the model
if st.button('Train Model'):
    train_images, train_labels, _, _ = load_data()
    with st.spinner('Training...'):
        model = train_model(model, train_images, train_labels)
    st.success('Model trained and saved!')

# Drawing canvas for digit input
canvas_result = st_canvas(
    stroke_width=18,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=200,
    height=200,
    drawing_mode='freedraw',
    key='canvas'
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype('uint8'))
    if st.button('Predict Digit'):
        digit, processed_image = predict_digit(model, image)
        st.write(f'Predicted Digit: {digit}')
        st.image(processed_image.reshape(28, 28), caption='Processed Image', width=100)

