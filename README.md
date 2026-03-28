# 🦴 Dinosaur Fossil Classification with ResNet50

## Overview

This project explores image classification using ResNet50 through two different approaches:

1. **Training from scratch** (custom implementation in PyTorch)
2. **Transfer learning** (pretrained ResNet50 fine-tuned on custom data)

The goal is to understand both the theoretical foundations of deep convolutional networks and the practical advantages of pretrained models.

---

## Dataset

* ~1400 images of dinosaur fossils
* Manually collected and organized into class folders
* Categories include examples such as:

  * Sauropod
  * Theropod
  * Ammonite
  * (add others here)

> Note: The dataset is relatively small, which significantly impacts model performance when training from scratch.

---

## Models

### 1. ResNet50 (From Scratch)

* Implemented manually in PyTorch
* Includes residual blocks and skip connections
* Trained entirely on the custom dataset

**Result:**

* Accuracy: 61%

**Challenges:**

* Limited dataset size
* Slower convergence
* Overfitting and generalization issues

---

### 2. ResNet50 (Transfer Learning)

* Pretrained on ImageNet
* Fine-tuned on fossil dataset
* Leveraged learned low-level visual features

**Result:**

* Accuracy: 85%

**Advantages:**

* Faster convergence
* Better generalization
* More robust feature extraction

---

## Model Comparison

| Model             | Accuracy |
| ----------------- | -------- |
| From Scratch      | 61%      |
| Transfer Learning | 85%      |

### Key Insight

Transfer learning significantly improves performance when working with limited data by reusing features learned from large-scale datasets.

---

## Deployment

The project is deployed as an interactive web application:

* **Frontend:** Gradio
* **Backend:** FastAPI
* **Infrastructure:** AWS ECS Fargate

### Features

* Upload an image of a fossil
* Run inference on both models simultaneously
* Compare predictions side-by-side

---

## Project Structure

```
.
├── app/
│   ├── main.py
│   └── example.json
│   └── fun_facts.json
│   └── resnet_finetuned.pth
│   └── resnet_scratch.pth
│   └── torch_utils_finetuned.py
│   └── torch_utils_scratch.py
├── front-end/
│   ├── Dockerfile
│   ├── requirement.txt
│   └── T-Rex-background.jpg
│
├── scratch.ipynb
├── finetuned.ipynb
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run backend

```bash
uvicorn app.main:app --reload
```

### 3. Launch Gradio UI

```bash
python app/gradio_ui.py
```

---

## Future Improvements

* Increase dataset size
* Apply data augmentation techniques
* Experiment with other architectures (EfficientNet, ViT)

---

## Acknowledgments

* PyTorch
* ImageNet pretrained models
* Open-source fossil image sources

---

## Author

Somprat Suratannon
---

## License

MIT License
