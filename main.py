import requests
import torch
from accelerate import infer_auto_device_map
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"
model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
device_map = infer_auto_device_map(model)
model.to(device_map)

processor = AutoProcessor.from_pretrained(model_id)

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
url = "https://llava-vl.github.io/static/images/view.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=32)
