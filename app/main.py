from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from app.torch_utils import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5  

app = FastAPI(title="ResNet50 Image Classifier")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model.load_state_dict(torch.load("app/dinosaur.pth"))
model.to(device)
model.eval()

print("Model loaded")

class_names = [
    "Theropod",
    "Sauropod",
    "Ornithischia",
    "Marine",
    "Unknown"
]

def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return preds.item()



@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    pred_idx = predict_image(image)
    pred_label = class_names[pred_idx]

    return {
        "predicted_class": pred_label,
        "class_index": pred_idx
    }
