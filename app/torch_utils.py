import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 5)

PATH = 'app/dinosaur.pth'
model.load_state_dict(torch.load(PATH))

def transform_image(image_bytes):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):

    outputs = model(image_tensor)
    _, preds =torch.max(outputs,1)
    return preds

