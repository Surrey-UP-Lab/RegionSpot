from collections import namedtuple
from .modeling.regionspot import build_regionspot_model
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import json
from detectron2.modeling import META_ARCH_REGISTRY
from .util.postprocessing import segmentation_postprocess

from detectron2.structures import Boxes, Instances
from .util.preprocessing import prepare_prompt_infer, prepare_prompt_train

__all__ = ["RegionSpot"]



@META_ARCH_REGISTRY.register()
class RegionSpot(nn.Module):
    """
    Implement RegionSpot
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.clip_type = cfg.MODEL.CLIP_TYPE
        self.inference_box_type = cfg.MODEL.BOX_TYPE
        self.clip_input_size = cfg.MODEL.CLIP_INPUT_SIZE
        self.clip_target_size = (self.clip_input_size, self.clip_input_size)
        self.model, _ = build_regionspot_model(clip_type = self.clip_type, is_training=cfg.MODEL.TRAINING, image_size=self.clip_input_size)
        self.model.to(self.device)
        if self.inference_box_type != 'GT':
            path = './datasets/glip_results/nms_results_glip_tiny_model_o365_goldg_cc_sbu_lvis_val.json'
            with open(path, 'r') as file:
                self.pred_results = json.load(file)
        else:
            self.pred_results = None
    
    @torch.no_grad()
    def foward_inference(self, batched_inputs, do_postprocess=True):
        with amp.autocast(enabled=True):
            with torch.no_grad():
                logits_per_image, pred_mask = self.model.forward_eval(batched_inputs, multimask_output=False)
        
        image_sizes = [x["original_size"] for x in batched_inputs]
        if self.inference_box_type == 'GT':
            boxes = torch.stack([x["instances"].gt_boxes.tensor for x in batched_inputs], dim=0) #n, n_box, n_token, 256
            if len(boxes[0]) == 0:
                boxes=torch.tensor([[[0,0, image_sizes[0][0], image_sizes[0][1]]]])
        else:
            boxes = torch.stack([x["pred_boxes"] for x in batched_inputs], dim=0) #n, n_box, n_token, 256
            scores = torch.stack([x["scores"] for x in batched_inputs], dim=0)

    
        box_cls = logits_per_image 
        box_pred = boxes 
        if self.inference_box_type == 'GT':
            results = self.inference_gt_box(box_cls, box_pred, pred_mask, image_sizes)
        else:
            results = self.inference_pred_box(box_cls, box_pred, scores, pred_mask, image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = segmentation_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def foward_train(self, batched_inputs):
        with amp.autocast(enabled=True):
            outputs = self.model.forward_train(batched_inputs)
        loss = {'loss': outputs}
        return loss
        
    def forward(self, batched_inputs, do_postprocess=True):
        if not self.training:
            # Prepare Prompt.
            batched_inputs = prepare_prompt_infer(batched_inputs, pred_results = self.pred_results, target_size=self.clip_target_size)

            results = self.foward_inference(batched_inputs)
            return results

        if self.training:
            batched_inputs = prepare_prompt_train(batched_inputs, target_size=self.clip_target_size)
            loss_dict = self.foward_train(batched_inputs)
            return loss_dict
        
    
    
    def inference_gt_box(self, box_cls, box_pred, pred_mask, image_sizes=None):

        device = box_cls.device  # assuming all tensors are on the same device
        results = []

        for logits, boxes, masks, img_size in zip(box_cls, box_pred, pred_mask, image_sizes):
            # Calculate probabilities and flatten them
            probs = F.softmax(logits, dim=-1)
            probs_flattened = probs.view(-1)
            
            # Determine number of top predictions to consider
            top_num = min(900, len(probs_flattened))
            
            # Get top-k values and indices
            topk_probs, topk_indices = torch.topk(probs_flattened, top_num)
            
            # Decode the top-k indices to get corresponding labels and boxes
            topk_labels = topk_indices % logits.shape[1]
            topk_boxes_indices = topk_indices // logits.shape[1]
            
            # Ensure boxes, masks and topk_boxes_indices are on the same device
            topk_boxes_indices = topk_boxes_indices.to(device)
            boxes = boxes.to(device)
            masks = masks.to(device)
            
            # Retrieve predictions using the top-k indices
            boxes_for_topk = boxes[topk_boxes_indices]
            masks_for_topk = masks[topk_boxes_indices]
            scores_for_topk = topk_probs # Modify accordingly if you have another score tensor
            # Create Instances object for top-k predictions
            result = Instances(img_size)
            result.pred_boxes = Boxes(boxes_for_topk)
            result.scores = scores_for_topk
            result.pred_classes = topk_labels
            result.pred_masks = masks_for_topk  # Added masks to the result
            results.append(result)

        return results

    def inference_pred_box(self, box_cls, box_pred, box_score, masks, image_sizes=None):

        results = []

        for i, (logits, box_pred_i, box_score_i, mask_i, img_size) in enumerate(zip(box_cls, box_pred, box_score, masks, image_sizes)):
            
            logits = logits.cuda() 
            box_pred_i = box_pred_i.cuda() 
            box_score_i = box_score_i.cuda()
            
            # Calculate probabilities and flatten them
            probs = F.softmax(logits, dim=-1)
            probs_flattened = probs.view(-1)
            
            # Determine number of top predictions to consider
            top_num = min(900, len(probs_flattened))
            
            # Get top-k values and indices
            topk_probs, topk_indices = torch.topk(probs_flattened, top_num)
            
            # Decode the top-k indices to get corresponding labels and boxes
            topk_labels = topk_indices % logits.shape[1]
            topk_boxes_indices = topk_indices // logits.shape[1]
            
            # Retrieve predictions using the top-k indices
            boxes = box_pred_i[topk_boxes_indices]
            masks = mask_i[topk_boxes_indices]
            scores = box_score_i[topk_boxes_indices] * topk_probs
            
            # Construct result for the current image
            result = Instances(img_size)
            result.pred_boxes = Boxes(boxes)
            result.scores = scores
            result.pred_classes = topk_labels
            result.pred_masks = masks
            results.append(result)

        return results

        
