import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import os

model_id = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically map model layers across available GPUs
)
model.tie_weights()

processor = AutoProcessor.from_pretrained(model_id)

prompt = "<|image|><|begin_of_text|>Is this image rotated or flipped?"

def analyze_image(image_path: str) -> str:
    raw_image = Image.open(image_path)
    inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=32)
    return processor.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':
    local_image_path = os.path.join(os.path.dirname(__file__), "view_upright.jpg")
    result = analyze_image(local_image_path)
    print("\n=============================\n")
    print(result)

    local_image_path = os.path.join(os.path.dirname(__file__), "view_rotated.jpg")
    result = analyze_image(local_image_path)
    print("\n=============================\n")
    print(result)