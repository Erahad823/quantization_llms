# """
# Quantize any Model(Audio,Vido,Text) into W8A16LinearLayer weights 8bit Activation 16bit Using per channel Linear quantization scheme

# 1. Creating a W8A16LinearLayer class to store 8-bit weights and scales

# 2. Replacing all `torch.nn.Linear` Layers with `W8A16LinearLayer`

# 3. Building a quantizer and quantize a model end-to-end.
 
# 4. Testing the naive absmax quantization on many scenario and study its impact.

# """



# """
# ###Building W8A16LinearLayer class
# Build the target class, W8A16LinearLayer(), that will be responsible for quantizating your model


# w8_a16_forward Function
# """


# # W8A16LinearLayer #8-bit #16-bit #optional
# """w8_a16_forward --> weights 8 bit , input 16 bit, scales, bias = None optional

# `==>> Cast the 8 bit weights to the same data type as the input, "casted weights", 
# `
# ==>>Keeping the casted weights in the same range as before [-128, 127]

# ==>> Next, $$(({inputs} \cdot \ text{``casted weights``}) * {scale}) + {bias}$$

# """

# import torch.nn.functional as F
# import torch 
# import torch.nn as nn


# random_int8 = torch.randint(-128, 127, (32, 16)).to(torch.int8)
# random_hs = torch.randn((1,16), dtype = torch.bfloat16)
# scales = torch.randn((1,32), dtype = torch.bfloat16)
# bias = torch.randn((1,32), dtype = torch.bfloat16)


# # random_int8 = torch.randint(-128, 127, (32, 16)).to(torch.int8)
# # random_hs = torch.randn((1, 16), dtype=torch.bfloat16)
# # scales = torch.randn((1, 32), dtype=torch.bfloat16)
# # bias = torch.randn((1, 32), dtype=torch.bfloat16)


# # print(F.linear(random_hs, random_int8.to(random_hs.dtype)))
# # print(F.linear(random_hs, random_int8.to(random_hs.dtype)) * scales)
# # print((F.linear(random_hs, random_int8.to(random_hs.dtype)) * scales) + bias)


# def w8_a16_forward(weight , input, scales, bias = None):
#     casted_weights = weight.to(input.dtype)
#     output = F.linear(input,casted_weights) * scales
#     if bias is not None:
#         output = output + bias
#     return output


# # print("With bias:\n\n", w8_a16_forward(random_int8, random_hs, scales, bias))

# # print("Without bias:\n\n", w8_a16_forward(random_int8, random_hs, scales, bias))



# """
# Init function of class W8A16LinearLayer

# """
# # This is how the init is of PyTorch Linear layer "https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear":


# # def __init__(self, in_features, out_features, bias=True,
# #              device=None, dtype=None)


# # running this will result in an error beacuse we cannot directly convert float to int 

# class W8A16LinerLayer(nn.Module):
#     def __init__(self,in_features,out_features,bias=True,dtype=torch.float32):
#         super().__init__()
#         self.int8_weights = nn.Parameter(torch.Tensor([0,1]).to(dtype=torch.int8))

# try:
#     W8A16LinerLayer(1,1)
# except Exception as error:
#     print(error)
#     print("\033[91m", type(error).__name__, ": ", error, "\033[0m")


# "THIS WILL GIVE ERROR RuntimeError:  Only Tensors of floating point and complex dtype can require gradients "

# "We have to use below functions"
# class W8A16LinearLayer(nn.Module):
#     def __init__(self, in_features, out_features,
#                  bias=True, dtype=torch.float32):
#         super().__init__()

#         self.register_buffer(
#             "int8_weights",
#             torch.randint(
#                 -128, 127, (out_features, in_features), dtype=torch.int8
#             )
#         )

#         self.register_buffer("scales",
#                              torch.randn((out_features), dtype=dtype))

#         if bias:
#             self.register_buffer("bias",
#                                  torch.randn((1, out_features),
#                                              dtype=dtype))

#         else:
#             self.bias = None


# # dummy_instance = W8A16LinearLayer(16, 32)
# # print(dummy_instance.int8_weights.shape)
# # print(dummy_instance.scales.shape)
# # print(dummy_instance.bias.shape)


# """1.3 - forward Function of class W8A16LinearLayer
# Use the w8_a16_forward defined earlier (Step 1.1) to define the forward function."""


# class W8A16LinearLayer(nn.Module):
#     def __init__(self, in_features, out_features,
#                  bias=True, dtype=torch.float32):
#         super().__init__()

#         self.register_buffer(
#             "int8_weights",
#             torch.randint(
#                 -128, 127, (out_features, in_features), dtype=torch.int8
#             )
#         )

#         self.register_buffer("scales",
#                              torch.randn((out_features), dtype=dtype))

#         if bias:
#             self.register_buffer("bias",
#                                  torch.randn((1, out_features),
#                                              dtype=dtype))

#         else:
#             self.bias = None

#     def forward(self, input):
#         return w8_a16_forward(self.int8_weights,
#                               input, self.scales, self.bias)


# module = W8A16LinearLayer(16, 32)
# dummy_hidden_states = torch.randn(1, 6, 16)

# print(dummy_hidden_states)
# print(module(dummy_hidden_states).shape)
# print(module(dummy_hidden_states).dtype)


# """quantize Function of class W8A16LinearLayer
# quantize function will dynamically quantize half-precision weights into torch.int8"""


# class W8A16LinearLayer(nn.Module):
#     def __init__(self, in_features, out_features,
#                  bias=True, dtype=torch.float32):
#         super().__init__()
#         print("0")

#         self.register_buffer(
#             "int8_weights",
#             torch.randint(
#                 -128, 127, (out_features, in_features), dtype=torch.int8
#             )
#         )

#         self.register_buffer("scales",
#                              torch.randn((out_features), dtype=dtype))

#         if bias:
#             self.register_buffer("bias",
#                                  torch.randn((1, out_features),
#                                              dtype=dtype))

#         else:
#             self.bias = None

#     def quantize(self, weights):
#         w_fp32 = weights.clone().to(torch.float32)

#         scales = w_fp32.abs().max(dim=-1).values / 127
#         print("scales: ", scales.dtype)
#         scales = scales.to(weights.dtype)
#         print("1")
#         print("scales: ", scales)

#         int8_weights = torch.round(weights
#                                    / scales.unsqueeze(1)).to(torch.int8)
#         print("2")
#         print("int8_weightsint8_weights: ", int8_weights)


#         self.int8_weights = int8_weights
#         print("3")
#         self.scales = scales

#     def forward(self, input):
#         print("4")
#         return w8_a16_forward(self.int8_weights,
#                               input, self.scales, self.bias)


# module = W8A16LinearLayer(4, 8)
# print("Weights before:\n", module.int8_weights)
# print(module.scales)
# print(module.scales.shape)
# print(module.int8_weights.shape)

# random_matrix = torch.randn((4, 8), dtype=torch.bfloat16)
# module.quantize(random_matrix)
# print("Quantize====>>>", module.quantize(random_matrix))

# print("Weights After:\n", module.int8_weights)
# print(module.scales)
# print(module.scales.shape)
# print(module.int8_weights.shape)


# # dequantized weights
# dequantized_weights  = module.int8_weights * module.scales.unsqueeze(1)
# print(dequantized_weights)
# print("original weights",random_matrix)
# print((random_matrix - module.int8_weights
#        * module.scales.unsqueeze(1)).abs().mean())


"""L4-B - Building your own Quantizer: Replace PyTorch layers with Quantized Layers¶
In this lesson, you will learn about the quantization pipline using your own 8-bit quantizer."""


# import torch
# import torch.nn as nn

# from helper import W8A16LinearLayer


"""
Step 2: Quantization Pipeline
Replace all of the torch.nn.Linear layers with the W8A16LinearLayer layer.
Call quantize on the linear layers using the original weights.

2.1 - Model In-place Linear Layer Replacement
Implement replace_linear_with_target
"""


# def replace_linear_with_target(module,
#                                target_class, module_name_to_exclude):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Linear) and not \
#                 any([x == name for x in module_name_to_exclude]):
#             old_bias = child.bias

#             new_module = target_class(child.in_features,
#                                       child.out_features,
#                                       old_bias is not None,
#                                       child.weight.dtype)
#             setattr(module, name, new_module)
#             if old_bias is not None:
#               getattr(module, name).bias = old_bias
#         else:
#             # Recursively call the function for nested modules
#             replace_linear_with_target(
#                 child, target_class, module_name_to_exclude)
            

# class DummyModel(torch.nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.emb = torch.nn.Embedding(1, 1)
#     # Try with bias
#     self.linear_1 = nn.Linear(1, 1)
#     # Try without bias
#     self.linear_2 = nn.Linear(1, 1, bias=False)
#     # Lm prediction head
#     self.lm_head = nn.Linear(1, 1, bias=False)


# model_1 = DummyModel()
# model_2 = DummyModel()

# replace_linear_with_target(model_1, W8A16LinearLayer, ["lm_head"])
# print(model_1)


# replace_linear_with_target(model_2, W8A16LinearLayer, [])
# print(model_2)


# """2.2 - Linear Layer Replacement + Quantization
# Modify the replace_linear_with_target function to also perform quantization.
# Implement replace_linear_with_target_and_quantize."""


# def replace_linear_with_target_and_quantize(module,
#                                             target_class, module_name_to_exclude):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Linear) and not \
#                 any([x == name for x in module_name_to_exclude]):
#             old_bias = child.bias
#             old_weight = child.weight

#             new_module = target_class(child.in_features,
#                                       child.out_features,
#                                       old_bias is not None,
#                                       child.weight.dtype)
#             setattr(module, name, new_module)

#             getattr(module, name).quantize(old_weight)

#             if old_bias is not None:
#               getattr(module, name).bias = old_bias
#         else:
#             # Recursively call the function for nested modules
#             replace_linear_with_target_and_quantize(child,
#                                                     target_class, module_name_to_exclude)


# model_3 = DummyModel()
# replace_linear_with_target_and_quantize(model_3, W8A16LinearLayer, ["lm_head"])
# print(model_3)


"""# L4-C - Building your own Quantizer: Quantize any Open Source PyTorch Model
"""

from helper import plot_results
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize


model_id = "Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))
print(pipe("def hello_world():", max_new_tokens=20,
           do_sample=False)[0]["generated_text"])


print(model.get_memory_footprint()/1e+6)

# for name, param in model.named_parameters():
#     print(f"name : {name} , dtype : {param.dtype}")
    
# print(f"model : {model}")
print("Model before:\n\n", model)

replace_linear_with_target_and_quantize(model,
                                        W8A16LinearLayer, ["lm_head"])
pipe.model
print(pipe("def hello_world():", max_new_tokens=20,
           do_sample=False)[0]["generated_text"])
print("Model After:\n\n", model)


print(model.get_memory_footprint()/1e+6)

for name, param in model.named_parameters():

    print(f"name : {name} , dtype : {param.dtype}")

print(f"model : {model}")


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
img_path = "dinner_with_friends.png"
image = Image.open(img_path).convert("RGB")
image

### Model before Quantization
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
