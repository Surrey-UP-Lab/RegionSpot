import torch
import numpy as np
import json
import torchvision.transforms.functional as F
from regionspot.modeling.segment_anything.utils.transforms import ResizeLongestSide

NORM_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(1).unsqueeze(2)
NORM_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(1).unsqueeze(2)


def resize_box(after_image_size, befor_image_size, boxes, size=800, max_size=1333): 
    # size can be min_size (scalar) or (w, h) tuple
    #size
    #
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(befor_image_size, size, max_size)

    

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(after_image_size, befor_image_size))
    ratio_width, ratio_height = ratios
    # ratio_width, ratio_height = 1, 1

    scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )

    return scaled_boxes

def resize_and_normalize(image, target_size=(224, 224)):
    resized_image = F.resize(image, target_size)
    device = resized_image.device
    return (resized_image - NORM_MEAN.to(device)) / NORM_STD.to(device)


def get_pred_boxes(pred_results, image_id):
    scores = torch.tensor(pred_results[image_id]['scores'])
    labels = torch.tensor(pred_results[image_id]['labels'])
    boxes = torch.tensor(pred_results[image_id]['boxes'])
   
    return scores, labels, boxes


def prepare_prompt_infer(batched_inputs, num_proposals=None, pred_results=None, target_size=(224,224)):
    boxes_type = 'GT'
    if pred_results is  not None:
        boxes_type = 'PRED_BOX'
    for x in batched_inputs:
        curr_image = x["image"]
        x["curr_image"] = curr_image.clone()
        image_id = x["image_id"]
        image = curr_image.permute(1, 2, 0).to(torch.uint8)
        curr_size = (image.shape[0], image.shape[1])
        
        resized_image = resize_and_normalize(curr_image.cuda() / 255, target_size=target_size)
        x["image"] = torch.as_tensor(ResizeLongestSide(1024).apply_image(np.array(image.cpu())), dtype=torch.float).permute(2, 0, 1).cuda()
        raw_size = (x['height'], x['width'])

        if boxes_type != 'GT':
            scores, gt_label, boxes_prompt = get_pred_boxes(pred_results, str(image_id))
            boxes_prompt = resize_box(curr_size, raw_size, boxes_prompt)
            x['pred_boxes'] = boxes_prompt
            x['scores'] = scores
        else:
            boxes_prompt = x["instances"].gt_boxes.tensor.cpu()
            if len(boxes_prompt) == 0:
                boxes_prompt = torch.tensor([[0, 0, *curr_size]])
        boxes_prompt = ResizeLongestSide(1024).apply_boxes(np.array(boxes_prompt), curr_size)
        x['boxes'] = torch.as_tensor(boxes_prompt, dtype=torch.float).cuda()
        x['resized_image'] = resized_image
        x['original_size'] = curr_size
    return batched_inputs


def prepare_prompt_train(batched_inputs, target_size=(224,224)):
    max_boxes = max(len(x["extra_info"]['mask_tokens']) for x in batched_inputs)
    num_proposals = max(max_boxes, 1)

    for x in batched_inputs:
        raw_image = x["image"]
        image = (x["image"].permute(1, 2, 0)).to(torch.uint8)
        curr_size = (image.shape[0], image.shape[1])
        resized_image = resize_and_normalize(raw_image.cuda() / 255, target_size=target_size)
        input_image = ResizeLongestSide(1024).apply_image(np.array(image.cpu()))
        input_image_torch = torch.as_tensor(input_image, dtype=torch.float).permute(2, 0, 1).cuda()
        x["image"] = input_image_torch
        mask_tokens = x["extra_info"]['mask_tokens'].clone().detach().cuda()
        labels = torch.tensor(x["extra_info"]['classes']).cuda()

        if x['dataset_name'] == 'coco':
            try:
                # Convert labels using the coco_new_dict
                labels = [constants.coco_new_dict[label.item()] for label in labels]
                labels = torch.tensor(labels).cuda()
            except:
                pass
        else:
            # Decrement each label by 1 unless it's zero
            new_labels = [label.item() - 1 if label.item() != 0 else 0 for label in labels]
            labels = torch.tensor(new_labels).cuda()

        num_gt = len(mask_tokens)
        num_repeat = num_proposals // num_gt
        repeat_tensor = [num_repeat] * (num_gt - num_proposals % num_gt) + [num_repeat + 1] * (num_proposals % num_gt)
        repeat_tensor = torch.tensor(repeat_tensor).cuda()
        mask_tokens = torch.repeat_interleave(mask_tokens, repeat_tensor, dim=0)
        labels = torch.repeat_interleave(labels, repeat_tensor, dim=0)

        x['resized_image'] = resized_image
        x['label'] = labels
        x['mask_tokens'] = mask_tokens
        x['original_size'] = curr_size

    return batched_inputs
