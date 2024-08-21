
"""Model Name = Salesforce/codegen-350M-mono"""
# from helper import plot_results
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize


# model_id = "Salesforce/codegen-350M-mono"

# model = AutoModelForCausalLM.from_pretrained(model_id,
#                                              torch_dtype=torch.bfloat16,
#                                              low_cpu_mem_usage=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# # print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))
# print(pipe("def hello_world():", max_new_tokens=20,
#            do_sample=False)[0]["generated_text"])


# print(model.get_memory_footprint()/1e+6)

# # for name, param in model.named_parameters():
# #     print(f"name : {name} , dtype : {param.dtype}")

# # print(f"model : {model}")
# print("Model before:\n\n", model)

# replace_linear_with_target_and_quantize(model,
#                                         W8A16LinearLayer, ["lm_head"])
# pipe.model
# print(pipe("def hello_world():", max_new_tokens=20,
#            do_sample=False)[0]["generated_text"])
# print("Model After:\n\n", model)


# print(model.get_memory_footprint()/1e+6)

# for name, param in model.named_parameters():

#     print(f"name : {name} , dtype : {param.dtype}")

# print(f"model : {model}")


"""Model Name = microsoft/Phi-3-mini-4k-instruct"""

# from helper import plot_results
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize
# from huggingface_hub import login

# login(token="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")

# model_id = "microsoft/Phi-3-mini-4k-instruct"

# model = AutoModelForCausalLM.from_pretrained(model_id,
#                                              torch_dtype=torch.bfloat16,
#                                              low_cpu_mem_usage=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# # # print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))
# # print(pipe("def hello_world():", max_new_tokens=20,
# #            do_sample=False)[0]["generated_text"])

# print(model)
# print(model.get_memory_footprint()/1e+6)

# # for name, param in model.named_parameters():
# #     print(f"name : {name} , dtype : {param.dtype}")

# # print(f"model : {model}")
# print("Model before:\n\n", model)

# replace_linear_with_target_and_quantize(model,
#                                         W8A16LinearLayer, ["lm_head"])
# pipe.model
# # print(pipe("def hello_world():", max_new_tokens=20,
# #            do_sample=False)[0]["generated_text"])
# print("Model After:\n\n", model)


# print(model.get_memory_footprint()/1e+6)

# # for name, param in model.named_parameters():

# #     print(f"name : {name} , dtype : {param.dtype}")

# # print(f"model : {model}")


"""Model Name = google-t5/t5-small"""

from helper import plot_results
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize
from huggingface_hub import login



model_id = "google-t5/t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))
# print(pipe("def hello_world():", max_new_tokens=20,
#            do_sample=False)[0]["generated_text"])
input_text  = "Hi how are you?"
inputs = tokenizer.encode(
    "translate English to French: " + input_text, return_tensors="pt")

# Generate text
outputs = model.generate(inputs, max_length=50,
                         num_beams=4, early_stopping=True)

# Decode and return the generated text
outputs =  tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Output==>>", outputs)

print(model)
print(model.get_memory_footprint()/1e+6)

# for name, param in model.named_parameters():
#     print(f"name : {name} , dtype : {param.dtype}")

# print(f"model : {model}")
print("Model before:\n\n", model)

replace_linear_with_target_and_quantize(model,
                                        W8A16LinearLayer, ["lm_head", "1-5"])
# pipe.model
# print(pipe("def hello_world():", max_new_tokens=20,
#            do_sample=False)[0]["generated_text"])
print("Model After:\n\n", model)


print(model.get_memory_footprint()/1e+6)


input_text = "Hi how are you?"
inputs = tokenizer.encode(
    "translate English to French: " + input_text, return_tensors="pt")

# Generate text
outputs = model.generate(inputs, max_length=50,
                         num_beams=4, early_stopping=True)

# Decode and return the generated text
outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Output==>>", outputs)
# for name, param in model.named_parameters():

#     print(f"name : {name} , dtype : {param.dtype}")

# print(f"model : {model}")
