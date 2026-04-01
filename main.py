# git clone https://github.com/cub-alan/WDR-AI_Sampling
# cd WDR-AI_Sampling    
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt

from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("BioCLIP fully working ✅")