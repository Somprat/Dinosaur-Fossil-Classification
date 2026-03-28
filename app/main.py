from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import sys
import random
import json

from app.torch_utils_scratch import model_scratch
from app.torch_utils_finetuned import model_finetuned


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5  

app = FastAPI(title="Dinosaur Fossil Classifier")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model_scratch.load_state_dict(torch.load("app/resnet_scratch.pth"))
model_scratch.to(device)
model_scratch.eval()

model_finetuned.load_state_dict(torch.load("app/resnet_finetuned.pth"))
model_finetuned.to(device)
model_finetuned.eval()

print("Model loaded")

class_names = [
    "Theropod",
    "Sauropod",
    "Ornithischia",
    "Marine",
    "Unknown"
]

def predict_image_scratch(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_scratch(image)
        _, preds = torch.max(outputs, 1)

    return preds.item()

def predict_image_finetuned(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_finetuned(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()

@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    pred_idx_scratch = predict_image_scratch(image)
    pred_label_scratch = class_names[pred_idx_scratch]

    pred_idx_finetuned = predict_image_finetuned(image)
    pred_label_finetuned = class_names[pred_idx_finetuned]
# remove the "app" in front of app/example.json and app/fun_facts.json 

    with open("app/example.json") as f1:
        examples = json.load(f1)
    category_example_scratch = pred_label_scratch.lower() + "_example"
    example_scratch = examples[category_example_scratch]

    category_example_finetuned = pred_label_finetuned.lower() + "_example"
    example_finetuned = examples[category_example_finetuned]


    with open("app/fun_facts.json", 'r') as f2:
        fun_facts = json.load(f2)
    category_fact_scratch = pred_label_scratch.lower() + "_fun_facts"
    fun_fact_scratch = random.choice(fun_facts[category_fact_scratch])

    category_fact_finetuned = pred_label_finetuned.lower() + "_fun_facts"
    fun_fact_finetuned = random.choice(fun_facts[category_fact_finetuned])


    return {
        "predicted_class_scratch": pred_label_scratch,
        "Example Dinosaurs_scratch": example_scratch,
        "Fun Fact!_scratch": fun_fact_scratch,

        "predicted_class_finetuned": pred_label_finetuned,
        "Example Dinosaurs_finetuned": example_finetuned,
        "Fun Fact!_finetuned": fun_fact_finetuned,
    }


