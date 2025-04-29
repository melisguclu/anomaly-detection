from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
from inference import run_inference
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((256, 256))  # Aynı eğitimdeki gibi
    image = center_crop(image, 224)   # Eğitimdeki center crop ile aynı
    image_array = np.array(image).astype(np.float32) / 255.0

    score, mask_filename = run_inference(image_array)

    return JSONResponse({
        "anomaly_score": score,
        "mask_image_url": f"/static/{mask_filename}"
    })


# statik görsel servisi
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="../static"), name="static")
