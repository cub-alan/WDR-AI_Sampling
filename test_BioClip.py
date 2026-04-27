import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# -----------------------------
# Load model
# -----------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

# -----------------------------
# Species list (scientific + common)
# -----------------------------
species = [
    ("Pentaglottis sempervirens", "Green alkanet"),
    ("Geum urbanum", "Herb bennet"),
    ("Elymus repens", "Couch grass"),
    ("Calystegia sepium", "Bindweed"),
    ("Fallopia japonica", "Japanese knotweed"),
    ("Aegopodium podagraria", "Ground elder"),
    ("Oxalis spp.", "Oxalis"),
    ("Ficaria verna", "Lesser celandine"),
    ("Circaea lutetiana", "Enchanter's nightshade"),
    ("Galium aparine", "Cleavers"),
    ("Geranium robertianum", "Herb robert"),
    ("Cardamine hirsuta", "Bittercress"),
    ("Ranunculus repens", "Creeping buttercup"),
    ("Urtica dioica", "Nettles"),
    ("Cirsium arvense", "Creeping thistle"),
    ("Chamerion angustifolium", "Rosebay willowherb"),
    ("Stellaria media", "Common chickweed"),
    ("Equisetum arvense", "Horsetail"),
    ("Poa annua", "Annual meadow grass"),
    ("Rumex obtusifolius", "Docks")
]

# -----------------------------
# Prompt templates (stronger)
# -----------------------------
templates = [
    "a botanical photograph of {}",
    "a close-up of {} leaves",
    "a {} plant growing in the wild",
    "a detailed image of {} foliage",
    "a clear photo of {} showing leaf structure",
    "a photo of {} with visible leaf edges"
    "a {} plant with serrated leaves",
    "a {} plant with jagged leaf edges",
    "a {} plant with smooth rounded leaves"
]

# -----------------------------
# Build text prompts
# -----------------------------
texts = []
label_map = []  # maps each prompt to species index

for i, (sci, common) in enumerate(species):
    name = f"{common} ({sci})"
    for t in templates:
        texts.append(t.format(name))
        label_map.append(i)

# -----------------------------
# Load image
# -----------------------------
image = Image.open("Nettles.jpg").convert("RGB")

# -----------------------------
# Prepare inputs
# -----------------------------
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits_per_image[0].cpu().numpy()  # shape: [N]

# -----------------------------
# Aggregate logits per species
# -----------------------------
scores = np.zeros(len(species))
counts = np.zeros(len(species))

for logit, idx in zip(logits, label_map):
    scores[idx] += logit
    counts[idx] += 1

scores = scores / counts  # average logits per species

# -----------------------------
# Apply softmax AFTER aggregation
# -----------------------------
exp_scores = np.exp(scores)
probs = exp_scores / exp_scores.sum()

# -----------------------------
# Sort + display
# -----------------------------
results = list(zip(
    [f"{common} ({sci})" for sci, common in species],
    probs
))

results = sorted(results, key=lambda x: x[1], reverse=True)

print("\nTop predictions:")
for label, score in results:
    print(f"{label}: {score:.4f}")

print(f"\nFinal Prediction: {results[0][0]}")