from fastapi import FastAPI, UploadFile, File, Response
from PIL import Image
import io
import numpy as np
import cv2

from inference import load_model, predict

app = FastAPI()

model = load_model("mobilenet") 

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    mask = predict(image, model)

    mask_png = (mask * 20).astype("uint8")
    _, encoded = cv2.imencode(".png", mask_png)

    return Response(
        content=encoded.tobytes(),
        media_type="image/png"
    )