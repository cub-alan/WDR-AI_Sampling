import torch
from PIL import Image
import open_clip

# -----------------------------
# Load model
# -----------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# -----------------------------
# Labels (you MUST define these)
# -----------------------------
labels = [
    "weed",
    "crop",
    "thistle",
    "tree",
    "person"
]

text_tokens = tokenizer(labels).to(device)

# -----------------------------
# Load image
# -----------------------------
image = preprocess(Image.open("Test.jpeg")).unsqueeze(0).to(device)

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Similarity → probabilities
    similarity = image_features @ text_features.T
    probs = similarity.softmax(dim=-1)

# -----------------------------
# Get highest classification
# -----------------------------
top_prob, top_idx = probs[0].topk(1)

print(f"Prediction: {labels[top_idx.item()]}")
print(f"Confidence: {top_prob.item():.4f}")