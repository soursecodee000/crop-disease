from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('trained_model.keras')

# Create FastAPI app
app = FastAPI()

class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
              'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
              'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
              'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image from the uploaded file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((128, 128))
    
    # Convert the image to a numpy array
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Make it a batch of one image

    # Model prediction
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    model_prediction = class_name[result_index]
    
    return {"prediction": model_prediction}
