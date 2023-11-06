import torch

pretrain_ckpt = './pretrained_model/model_final.pth'
checkpoint = torch.load(pretrain_ckpt, map_location='cpu')

# Remove specific keys from the top-level dictionary
top_level_keys_to_remove = ['trainer', 'iteration']
for key in top_level_keys_to_remove:
    if key in checkpoint:
        del checkpoint[key]

# Remove keys that start with 'clip_model' and 'sam' from the checkpoint's 'model' dictionary
model_keys_to_remove = ['model.clip_model', 'model.sam']
for key in list(checkpoint['model'].keys()):  # Use list to copy keys
    if any(key.startswith(to_remove) for to_remove in model_keys_to_remove):
        print(key)
        del checkpoint['model'][key]

# Save the modified checkpoint back to a file
modified_ckpt_path = './pretrained_model/model_final_modified.pth'
torch.save(checkpoint, modified_ckpt_path)
print(checkpoint['model'].keys())
