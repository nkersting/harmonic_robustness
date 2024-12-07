from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image, ImageOps
import numpy as np


#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]


image = Image.open('/Users/lordkersting/neuro/Downloads/animals_small/raw-img/elefante/e032b10b2df21c22d2524518b7444f92e37fe5d404b0144390f8c07aa5ecb6_640.jpg')

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(logits.argmax(-1), model.config.id2label[predicted_label])
print(logits[0][logits.argmax(-1)], torch.mean(logits[0]))
