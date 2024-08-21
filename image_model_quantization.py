
"3.2 - facebook/detr-resnet-50"
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
from helper import plot_results
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm")


previous_memory_footprint = model.get_memory_footprint()
print("Footprint of the model in MBs: ",
      previous_memory_footprint/1e+6)
# img_path = "dinner_with_friends.png"
img_path = "Times-Square-04378-1536x1025.jpg"
image = Image.open(img_path).convert("RGB")
image

# Model before Quantization
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9)[0]

plot_results(model, image, results)

model

replace_linear_with_target_and_quantize(model,
                                        W8A16LinearLayer,
                                        ["0", "1", "2", "class_labels_classifier"])


# Model after quantization
model

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9)[0]

plot_results(model, image, results)
new_footprint = model.get_memory_footprint()


print("Footprint of the model in MBs: ",
      new_footprint/1e+6)


# Memory saved
print("Memory saved in MBs: ",
      (previous_memory_footprint - new_footprint)/1e+6)
