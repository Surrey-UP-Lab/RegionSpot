import numpy as np
import cv2
import matplotlib.pyplot as plt
from regionspot.modeling.regionspot import build_regionspot_model
from regionspot import RegionSpot_Predictor
# Function to show masks on an image
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Function to show points on an image
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Function to show bounding boxes on an image
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', lw=2))

# Read image and set up model
image = cv2.imread('assets/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
# Multiple boxes
box_prompt = np.array([[64, 926, 804, 1978], [1237, 490, 1615, 771.], [1510, 64, 1670, 167]]) 
ckpt_path = '/path/to/model_weights.pth'
clip_type = 'CLIP_400M_Large_336'
clip_input_size = 336
custom_vocabulary =  ["Smoothie bowl", "Banana", "Strawberry", "Chia seeds", "Shredded coconut", "Wooden spoons", "Grapefruit", "Goji berries", "Flaxseeds seeds"]

# Build and initialize the model
model, msg = build_regionspot_model(is_training=False, image_size=clip_input_size, clip_type=clip_type, pretrain_ckpt=ckpt_path, custom_vocabulary=custom_vocabulary)

# Create predictor and set image
predictor = RegionSpot_Predictor(model.cuda())
predictor.set_image(image, clip_input_size=clip_input_size)

# Prediction based on box prompt
masks, mask_iou_score, class_score, class_index = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box_prompt,
    multimask_output=False,  
)
# Extract class name and display image with masks and box
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
for idx in range(len(box_prompt)):
    show_mask(masks[idx], ax)  
    show_box(box_prompt[idx], ax)  # Assuming box_prompt contains all your boxes
    # You might want to modify the text display for multiple classes as well
    class_name = custom_vocabulary[int(class_index[idx])]
    ax.text(box_prompt[idx][0], box_prompt[idx][1] - 10, class_name, color='white', fontsize=14, bbox=dict(facecolor='green', edgecolor='green', alpha=0.6))

ax.axis('off')
plt.show()
fig.savefig('result.png')
plt.close(fig)

