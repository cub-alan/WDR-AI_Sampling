import cv2
import numpy as np
from pathlib import Path

from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# -----------------------------
# 1. PATH SETUP
# -----------------------------
BASE_PATH = Path.cwd()
CONFIG_NAME = "sam2.1_hiera_small.yaml" 
CHECKPOINT_PATH = BASE_PATH / "sam2.1_hiera_small.pt"
IMAGE_PATH = BASE_PATH / "Test.jpeg"

# -----------------------------
# 2. DEBUG CHECKS
# -----------------------------
print("Files in directory:")
print(list(BASE_PATH.iterdir()))

print("\nCheckpoint exists:", CHECKPOINT_PATH.exists())
print("Image exists:", IMAGE_PATH.exists())

if not IMAGE_PATH.exists():
    print("\n❌ ERROR: Image not found")
    exit()

# -----------------------------
# 3. CLEAR HYDRA (IMPORTANT)
# -----------------------------
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# -----------------------------
# 4. INITIALISE HYDRA
# -----------------------------
initialize(config_path="configs", version_base=None)

# -----------------------------
# 5. LOAD SAM2 MODEL (CPU FIX)
# -----------------------------
sam2 = build_sam2(
    config_file=CONFIG_NAME,
    ckpt_path=str(CHECKPOINT_PATH),
    device="cpu"   #  prevents CUDA error
)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

# -----------------------------
# 6. LOAD IMAGE
# -----------------------------
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    print("Could not load image")
    exit()

image = cv2.resize(image, (640, 480))

# -----------------------------
# 7. GENERATE MASKS
# -----------------------------
print("\nGenerating masks...")
masks = mask_generator.generate(image)

print(f" SAM2 working! Found {len(masks)} masks")

# -----------------------------
# 8. VISUALISE RESULTS
# -----------------------------
output = image.copy()

for mask in masks:
    segmentation = mask["segmentation"]

    # random colour per mask
    color = np.random.randint(0, 255, size=3)

    output[segmentation] = color

# Blend original + masks
blended = cv2.addWeighted(image, 0.6, output, 0.4, 0)

# -----------------------------
# 9. DISPLAY
# -----------------------------
cv2.imshow("SAM2 Segmentation", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()