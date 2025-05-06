import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image

# Load the model once when the application starts
model = tf.keras.models.load_model('trained_model.keras')

# Class names for disease predictions
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Initialize FastAPI app
app = FastAPI()

# Function to predict the disease from the uploaded image
def predict_disease(image: Image.Image):
    # Resize and prepare the image for prediction
    image = image.resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Create a batch of size 1
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)  # Get the index of the highest probability
    model_prediction = class_name[result_index]  # Get the corresponding disease name
    return model_prediction

# FastAPI route to receive the image and predict the disease
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Convert the uploaded image to a format suitable for prediction
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    
    # Call the predict function
    prediction = predict_disease(image)
    
    return {"prediction": prediction}
