# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from test import predict_image

app = FastAPI()

# Enable CORS for Flutter (optional but useful)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your Flutter app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return {"predicted_class": result}
