from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# from architecture import build_model
from src.model.architecture import build_model

# Initialize the FastAPI application
app = FastAPI(
    title="Pneumonia Detection API",
    description="A medical imaging API that classifies Chest X-rays as Normal or Pneumonia."
)

# Global variable to hold our model
model = None

# This runs once when the server starts up
@app.on_event("startup")
async def load_model_weights():
    global model
    # 1. Build the model structure
    model = build_model()
    
    # 2. Define the path to the weights file
    weights_path = os.path.join("saved_models", "best_pneumonia_model.weights.h5")
    print(f"Loading weights from {weights_path}...")
    
    # 3. Load the weights into the model structure
    model.load_weights(weights_path)
    print("Model weights loaded successfully!")

def prepare_image(image_bytes: bytes):
    """
    Transforms the raw uploaded image to match the exact format 
    the model expects (224x224, normalized).
    """
    try:
        # Load the image and convert to RGB (in case someone uploads a grayscale or RGBA image)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to match config.IMG_SIZE
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # INFERENCE PARITY: Apply the exact same normalization used in data_loader.py
        img_array = img_array / 255.0
        
        # The model expects a "batch" of images, even if we only send one.
        # We expand the dimensions from (224, 224, 3) to (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image file and returns the model's prediction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the file bytes from the incoming request
        image_bytes = await file.read()
        
        # Process the image
        processed_image = prepare_image(image_bytes)
        
        # Make the prediction
        # The model returns a probability between 0.0 (Normal) and 1.0 (Pneumonia)
        prediction_prob = model.predict(processed_image)[0][0]
        
        # Determine the label based on a 0.5 threshold
        is_pneumonia = bool(prediction_prob > 0.5)
        label = "Pneumonia" if is_pneumonia else "Normal"
        
        # Return a structured JSON response
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": label,
            "confidence_score": float(prediction_prob) if is_pneumonia else float(1.0 - prediction_prob),
            "raw_probability": float(prediction_prob)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))