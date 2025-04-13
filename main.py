from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForImageClassification, ViTImageProcessor, ViTConfig
import torch
from PIL import Image
import os
import io
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_name = "iDharshan/vit-base-patch16-224-SIViT"
processor = ViTImageProcessor.from_pretrained(model_name, use_safetensors=True)
config = ViTConfig.from_pretrained(model_name, num_labels=6, use_safetensors=True)
model = AutoModelForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True, use_safetensors=True)

class_names = ["E8", "E3", "E1", "E2", "E6", "EHRB"]
class_meanings = {
    "E1": "Old thin steel scrap (≤1.5x0.5x0.5 m, thickness <6 mm)",
    "E2": "Thick new production steel scrap (≤1.5x0.5x0.5 m, thickness ≥3 mm)",
    "E3": "Old thick steel scrap (≤1.5x0.5x0.5 m, thickness ≥6 mm)",
    "E6": "Thin new production steel scrap, compressed or baled (thickness <3 mm)",
    "E8": "Thin new production steel scrap (≤1.5x0.5x0.5 m, thickness <3 mm)",
    "EHRB": "Old and new steel scrap, mainly rebars and merchant bars (max 1.5x0.5x0.5 m)"
}

def predict_image(image):
    image = Image.open(io.BytesIO(image)).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = class_names[predicted_class_idx]
    predicted_meaning = class_meanings[predicted_class]

    return f"Scrap Category: {predicted_class}<br>Description: {predicted_meaning}"

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, port=5050)