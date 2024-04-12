import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import itertools
import random
import torch.utils.data as data
from PIL import Image
import os
import os.path
import xml.etree.ElementTree as ET
from torchvision.transforms import ToTensor, Resize
import torchmetrics





ce = nn.CrossEntropyLoss()


def UnetLoss(preds, targets):
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(preds, targets.long())
    acc_Unet = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc_Unet  


import os
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from typing import Any, Tuple, Callable, Optional, List, Dict
import xml.etree.ElementTree as ET
from torchvision.transforms.functional import resize as resize_fn

def preproc_boxes(bbox, image):
    if len(bbox) == 0:
        bbox = torch.zeros(0, 4)
    height = 300
    width = 300
    boxes = bbox.clone()
    boxes[:, 0::2] /= width
    boxes[:, 1::2] /= height

    return boxes


class VOC2012(VisionDataset):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super(VOC2012, self).__init__(root, transforms, transform, target_transform)
        self.year = year
        self.image_set = image_set

        self.images_dir = os.path.join(self.root, f'VOC{year}', 'JPEGImages')
        self.masks_dir = os.path.join(self.root, f'VOC{year}', 'SegmentationClass')
        self.annotations_dir = os.path.join(self.root, f'VOC{year}', 'Annotations')

        self._load_voc_images()

    def _load_voc_images(self):
        splits_dir = os.path.join(self.root, f'VOC{self.year}', 'ImageSets', 'Segmentation', self.image_set + '.txt')
        
        with open(splits_dir, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.images_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(self.masks_dir, x + ".png") for x in file_names]
        self.annotations = [os.path.join(self.annotations_dir, x + ".xml") for x in file_names]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        original_img_size = img.size
        tensor_transform = ToTensor()
        #resize = Resize((300, 300), antialias=True, interpolation=Image.NEAREST)
        # Apply the transform to your images
        img = tensor_transform(img)
        #img = resize(img)
        mask = Image.open(self.masks[index]).convert("L")
        mask = tensor_transform(mask)
        #mask = resize(mask)[0]
        #mask = one_hot_encode_mask(mask, num_classes=21)
        annotation = self._parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        #if annotation['boxes'].size(0) > 0:
            #annotation['boxes'] = _resize_bbox(bbox=annotation['boxes'], orig_size=original_img_size)
        annotation['boxes'] = preproc_boxes(annotation['boxes'], img)
        #print("data before croping")
        #print(f"img shape: {img.shape}, labels shape: {annotation['labels'].shape}, boxes shape: {annotation['boxes'].shape}")

        # augmentations
        img, annotation["boxes"], annotation["labels"], mask = _crop(img, annotation['boxes'], annotation['labels'], mask)
        #print("--------------------")
        #print("data after croping")
        #print(f"img shape: {img.shape}, labels shape: {annotation['labels'].shape}, boxes shape: {annotation['boxes'].shape}")
        #img, annotation['boxes'], annotation['labels'], mask = _mirror(img, annotation['boxes'], annotation['labels'], mask)

        if self.transforms is not None:
            img, target = self.transforms(img, mask, annotation)
        else:
            target = {"masks": mask, "boxes": annotation["boxes"], "labels": annotation["labels"]}

        return img, target

    def __len__(self) -> int:
        return len(self.images)

    def _parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:
        annotation = {}
        for child in node:
            if child.tag == 'object':
                obj_name = child.find('name').text
                bndbox = child.find('bndbox')
                coords = [int(bndbox.find(tag).text) - 1 for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
                annotation.setdefault('boxes', []).append(coords)
                annotation.setdefault('labels', []).append(obj_name)
        
        # Convert annotations to tensors
        annotation['boxes'] = torch.tensor(annotation.get('boxes', []), dtype=torch.float32)
        annotation['labels'] = torch.tensor([self._class_to_idx(label) for label in annotation.get('labels', [])], dtype=torch.int64)
        
        return annotation

    def _class_to_idx(self, label: str) -> int:
        classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        return classes.index(label)

    @property
    def classes(self) -> List[str]:
        return ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        return len(self.ids)


def match_with_iou(pred_boxes, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
    iou_metric = torchmetrics.detection.iou.IntersectionOverUnion(box_format='xyxy', iou_threshold=iou_threshold)
    batch_size = pred_boxes.size(0)
    
    batch_matched_idxs = []
    batch_matched_labels = []

    #print(f"pred_boxes: {pred_boxes.shape}, pred_scores: {pred_scores.shape}, gt_boxes: {gt_boxes.shape}, gt_labels: {gt_labels.shape}")
    

        #print(f"pred_boxes: {pred_boxes[i].shape}, pred_scores: {pred_scores[i].shape}, gt_boxes: {gt_boxes[i].shape}, gt_labels: {gt_labels[i].shape}")
    preds = [{'boxes': pred_boxes, 'labels': torch.argmax(pred_scores, dim=1)}]
    targets = [{'boxes': gt_boxes, 'labels': gt_labels}]
    
    iou_dict = iou_metric(preds, targets)
    iou_values = iou_dict['iou']  # Assuming single-image processing for now

    # Determine matches based on IoU threshold
    max_iou, max_indices = torch.max(iou_values, dim=0)
    matched = max_iou >= iou_threshold

    matched_idxs = max_indices[matched]
    matched_labels = gt_labels[matched_idxs]

    batch_matched_idxs.append(matched_idxs)
    batch_matched_labels.append(matched_labels)
    matched_boxes = gt_boxes[matched_idxs]

    
    
    return batch_matched_idxs, batch_matched_labels


import torch
import itertools
import torchmetrics
from torchvision.ops import box_iou, box_convert

grids = (38, 19, 10, 5, 3, 1)
steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)]
aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))


class MultiBoxEncoder(object):
    def __init__(self, cfg):
        self.variance = torch.tensor(cfg['variance'])

        grids = cfg['grids']
        steps = [s / 300 for s in cfg['steps']]
        sizes = [s / 300 for s in cfg['sizes']]
        aspect_ratios = cfg['aspect_ratios']

        default_boxes = []
        for k, grid in enumerate(grids):
            for v, u in itertools.product(range(grid), repeat=2):
                cx = (u + 0.5) * steps[k]
                cy = (v + 0.5) * steps[k]

                # Adding square box
                s = sizes[k]
                default_boxes.append((cx, cy, s, s))

                # Adding next square box
                s = (sizes[k] * sizes[k + 1]) ** 0.5
                default_boxes.append((cx, cy, s, s))

                # Adding aspect ratio boxes
                s = sizes[k]
                for ar in aspect_ratios[k]:
                    default_boxes.extend([
                        (cx, cy, s * (ar ** 0.5), s / (ar ** 0.5)),
                        (cx, cy, s / (ar ** 0.5), s * (ar ** 0.5))
                    ])

        default_boxes = torch.tensor(default_boxes).clamp(0, 1)
        self.default_boxes = default_boxes  # Shape: [num_priors, 4]

    def encode(self, boxes, labels, threshold=0.2):
        '''
        Match each anchor box with the ground truth box of the highest IoU overlap, 
        encoding the bounding boxes and class labels.
        '''
        if len(boxes) == 0:
            # Return zeros if there are no boxes
            return (torch.zeros(self.default_boxes.size(), dtype=torch.float),
                    torch.zeros(self.default_boxes.size(0), dtype=torch.long))

        print(f"default_boxes: {self.default_boxes[:1]}, boxes: {boxes[:1]}, labels: {labels[:1]}")

        # Convert boxes to point form and compute IoU
        iou = box_iou(box_convert(self.default_boxes, in_fmt='cxcywh', out_fmt='xyxy'), boxes)

        # Match each anchor box to the ground truth box with the highest IoU overlap
        gt_idx = iou.argmax(1)
        iou, _ = iou.max(1)
        matched_boxes = boxes[gt_idx]
        matched_labels = labels[gt_idx]

        # Encode matched boxes (location targets)
        loc = torch.cat([
            ((matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2 - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_boxes[:, 2:]),
            torch.log((matched_boxes[:, 2:] - matched_boxes[:, :2]) / self.default_boxes[:, 2:]) / self.variance[1]
        ], dim=1)

        # Class targets (background is class 0, hence add 1 to labels)
        conf = matched_labels
        conf[iou < threshold] = 0  # Set background for low IoU

        return loc, conf

def test_anchor_generation():
    cfg = {
        'grids': (38, 19, 10, 5, 3, 1),
        'steps': [8, 16, 32, 64, 100, 300],
        'sizes': [30, 60, 111, 162, 213, 264, 315],
        'aspect_ratios': ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        'variance': [0.1, 0.2]
    }
    encoder = MultiBoxEncoder(cfg)
    assert encoder.default_boxes.size(0) > 0, "No default boxes generated."
    print("finished")

def test_encoding_known_ground_truth():
    cfg = {
        'grids': (38, 19, 10, 5, 3, 1),
        'steps': [8, 16, 32, 64, 100, 300],
        'sizes': [30, 60, 111, 162, 213, 264, 315],
        'aspect_ratios': ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        'variance': [0.1, 0.2]
    }
    encoder = MultiBoxEncoder(cfg)
    
    # Known ground truth boxes (in cxcywh format) and labels
    boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])  # Example box
    labels = torch.tensor([1])  # Example label
    
    loc, conf = encoder.encode(boxes, labels)
    
    assert loc.size(0) == encoder.default_boxes.size(0), "Location targets size mismatch."
    assert conf.size(0) == encoder.default_boxes.size(0), "Confidence scores size mismatch."
    assert (conf > 0).any(), "No positive matches found."
    print("finished")

def test_integration_with_mock_model():
     # Your configuration dictionary
    cfg = {
        'grids': (38, 19, 10, 5, 3, 1),
        'steps': [8, 16, 32, 64, 100, 300],
        'sizes': [30, 60, 111, 162, 213, 264, 315],
        'aspect_ratios': ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        'variance': [0.1, 0.2]
    }
    encoder = MultiBoxEncoder(cfg)
    
    # Mock model output
    pred_boxes = torch.rand(8732, 4)  # Random predictions
    pred_labels = torch.randint(21, (8732,))  # Random labels for 20 classes
    
    # Known ground truth for testing
    gt_boxes = torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]])
    gt_labels = torch.tensor([1, 2])
    
    loc, conf = encoder.encode(gt_boxes, gt_labels)

    localization_loss_fn = torch.nn.SmoothL1Loss()
    classification_loss_fn = torch.nn.CrossEntropyLoss()

    loc_loss = localization_loss_fn(pred_boxes, loc)
    #conf_loss = classification_loss_fn(pred_labels, conf)
    print(f"loc_loss: {loc_loss}")
    
    assert loc.size(0) == pred_boxes.size(0), "Mismatch in number of encoded boxes."
    assert conf.size(0) == pred_boxes.size(0), "Mismatch in number of confidence scores."
    print("finished")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------  Utility Functions  ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import torch
import itertools
import math

cfg = {
    'grids': (38, 19, 10, 5, 3, 1),
    'steps': [8, 16, 32, 64, 100, 300],
    'sizes': [30, 60, 111, 162, 213, 264, 315],
    'aspect_ratios': ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
    'variance': [0.1, 0.2]
}

def point_form(boxes):
    """ Convert boxes to (xmin, ymin, xmax, ymax) representation."""
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                      boxes[:, :2] + boxes[:, 2:]/2), 1)

def bbox_iou(box_a, box_b):
    """ Compute the intersection over union of two set of boxes."""
    box_a = box_a.to("cuda")
    box_b = box_b.to("cuda")
    inter = torch.min(box_a[:, None, 2:], box_b[:, 2:]) - torch.max(box_a[:, None, :2], box_b[:, :2])
    inter = torch.clamp(inter, min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0)

    union = area_a + area_b - inter_area
    return inter_area / union

class MultiBoxEncoder_GPT(object):

    def __init__(self, cfg):
        self.variance = cfg['variance']

        default_boxes = []
        for k, f in enumerate(cfg['grids']):
            for i, j in itertools.product(range(f), repeat=2):
                cx = (j + 0.5) / f  # Normalizing cx and cy to [0, 1]
                cy = (i + 0.5) / f

                for size in (cfg['sizes'][k], math.sqrt(cfg['sizes'][k] * cfg['sizes'][k + 1])):
                    default_boxes.append([cx, cy, size / 300, size / 300])  # Assuming image size of 1024 for normalization

                for ar in cfg['aspect_ratios'][k]:
                    default_boxes.extend([
                        [cx, cy, size * math.sqrt(ar) / 300, size / math.sqrt(ar) / 300],
                        [cx, cy, size / math.sqrt(ar) / 300, size * math.sqrt(ar) / 300]
                    ])

        self.default_boxes = torch.clip(torch.tensor(default_boxes, dtype=torch.float32), 0, 1)
        self.default_boxes = (self.default_boxes).to("cuda")

    def encode(self, boxes, labels, threshold=0.5):
        # Ensure boxes and default_boxes are in the same scale
        if boxes.size(0) == 0:
            return torch.zeros_like(self.default_boxes), torch.zeros(self.default_boxes.size(0), dtype=torch.int64)

        iou = bbox_iou(point_form(self.default_boxes), boxes)

        # Match logic
        best_gt_iou, best_gt_idx = iou.max(1)
        
        # Ensure each ground truth box matches with at least one default b
        
        conf = torch.zeros_like(best_gt_idx, dtype=torch.int64)
        #conf = torch.zeros_like(self.default_boxes.size(0), dtype=torch.int64)

        # Set label for matches above threshold; labels are incremented by 1 to reserve 0 for background
        above_threshold = best_gt_iou > threshold
        conf[above_threshold] = (labels[best_gt_idx[above_threshold]]).long()


        matches = boxes[best_gt_idx]

        loc = torch.cat([
            (matches[:, :2] + matches[:, 2:])/2 - self.default_boxes[:, :2],
            torch.log((matches[:, 2:] - matches[:, :2]) / self.default_boxes[:, 2:])
        ], 1)
        loc[:, :2] /= self.variance[0]
        loc[:, 2:] /= self.variance[1]

        return loc, conf
    
    def decode(self, loc):
        """
        Decode location predictions using default boxes (anchors).
        Args:
            loc (Tensor): Location predictions for bounding boxes,
                          shape: [num_boxes, 4], format: [Δcx, Δcy, Δw, Δh].
        Returns:
            boxes (Tensor): Decoded box predictions, format: [x_min, y_min, x_max, y_max].
        """
        # Split the default boxes into center coordinates (cxcy) and sizes (wh)
        cxcy = self.default_boxes[:, :2]
        wh = self.default_boxes[:, 2:]

        # Decode locations from predictions
        boxes = torch.cat([
            cxcy + loc[:, :2] * self.variance[0] * wh,  # Δcx, Δcy to cx, cy
            wh * torch.exp(loc[:, 2:] * self.variance[1])  # Δw, Δh to w, h
        ], dim=1)

        # Convert cx, cy, w, h to x_min, y_min, x_max, y_max
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes



import torch.nn.functional as F

"""
class SSD_loss(nn.Module):

    def __init__(self, num_classes=21, neg_radio=3):
        super(SSD_loss, self).__init__()
        self.num_classes = num_classes
        self.neg_radio = neg_radio

    def forward(self, pred_loc, pred_label, gt_loc, gt_label):

        pos_idx = gt_label > 0
        pred_loc_pos = pred_loc[pos_idx]
        gt_loc_pos = gt_loc[pos_idx]

        loc_loss = F.smooth_l1_loss(pred_loc_pos, gt_loc_pos, reduction='sum')

        cls_loss = F.cross_entropy(pred_label, gt_label, reduction='sum')

        N = pos_idx.long().sum(dtype=torch.float)

        loc_loss /= N
        cls_loss /= N

        return loc_loss, cls_loss"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def hard_negative_mining(confidence_loss, pos_mask, neg_ratio):
    """
    Performs hard negative mining.

    Parameters:
    - confidence_loss: Tensor, the loss for each example.
    - pos_mask: Tensor, a mask of positive examples.
    - neg_ratio: int, the ratio of negatives to positives.

    Returns:
    - mask: Tensor, a mask to select examples for the classification loss.
    """
    # Flatten the confidence_loss and pos_mask to simplify processing
    confidence_loss_flat = confidence_loss.view(-1).to(device)
    pos_mask_flat = pos_mask.view(-1).to(device)

    # Number of hard negatives to select
    num_pos = pos_mask_flat.sum(dim=0, keepdim=True)
    num_neg = neg_ratio * num_pos

    # Set confidence loss for pos to -inf so they are not selected as hard negatives
    confidence_loss_flat[pos_mask_flat] = float('-inf')
    # Sort by loss
    _, idx = confidence_loss_flat.sort(descending=True)
    _, rank = idx.sort()

    neg_mask_flat = rank < num_neg

    return neg_mask_flat.view(pos_mask.size())



class SSD_loss(nn.Module):
    def __init__(self, num_classes=21, neg_ratio=3):
        super(SSD_loss, self).__init__()
        self.num_classes = num_classes
        self.neg_ratio = neg_ratio

    def forward(self, pred_loc, pred_label, gt_loc, gt_label):
        pos_mask = gt_label > 0
        pos_mask = pos_mask.to(device)
        pred_loc = pred_loc.to(device)
        pred_label = pred_label.to(device)
        gt_loc = gt_loc.to(device)
        gt_label = gt_label.to(device)

        local_loss = torch.nn.SmoothL1Loss(reduction='sum')
        class_loss = torch.nn.CrossEntropyLoss(reduction='sum')

        # Localization Loss
        #loc_loss = torch.nn.smooth_l1_loss(pred_loc[pos_mask], gt_loc[pos_mask], reduction='sum')
        loc_loss = local_loss(pred_loc[pos_mask], gt_loc[pos_mask])

        # Classification Loss
        num_boxes, _ = pred_label.size()
        conf_loss = F.cross_entropy(pred_label.view(-1, self.num_classes), gt_label.view(-1), reduction='none')
        

        # Hard Negative Mining
        neg_mask = hard_negative_mining(conf_loss, pos_mask, self.neg_ratio)
        neg_mask = neg_mask.to(device)

        # Selecting positives and hard negatives
        pos_or_neg = pos_mask | neg_mask
        #cls_loss = F.cross_entropy(pred_label[pos_or_neg], gt_label[pos_or_neg], reduction='sum')
        cls_loss = class_loss(pred_label[pos_or_neg], gt_label[pos_or_neg])

        # Normalize the losses
        num_pos = pos_mask.sum().float()  # Ensuring float division
        loc_loss /= num_pos.clamp(min=1.0)
        cls_loss /= num_pos.clamp(min=1.0)

        return loc_loss, cls_loss
    

def simple_voc_eval(pred_boxes, pred_scores, gt_boxes, ovthresh=0.1):
    """
    Compute the average precision and recall at a single IoU threshold (ovthresh).
    
    Parameters:
    - pred_boxes : Tensor of predicted bounding boxes, shape [N, 4]
    - pred_scores : Tensor of scores for the predicted boxes, shape [N]
    - gt_boxes : Tensor of ground truth bounding boxes, shape [M, 4]
    - ovthresh : IoU threshold to consider a predicted box as a true positive
    
    Returns:
    - precision, recall, average precision
    """
    
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    
    # Sort predictions by scores (descending)
    _, sorted_idx = pred_scores.sort(descending=True)
    sorted_pred_boxes = pred_boxes[sorted_idx]

    # Calculate IoUs between all predictions and ground truth boxes
    ious = bbox_iou(sorted_pred_boxes, gt_boxes)
    
    # Determine the best ground truth match for each prediction
    max_ious, max_iou_idxs = ious.max(1)
    tp = torch.zeros_like(pred_scores)
    fp = torch.zeros_like(pred_scores)
    
    detected = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=gt_boxes.device)
    
    for i in range(max_ious.size(0)):
        if max_ious[i] > ovthresh:
            if not detected[max_iou_idxs[i]]:
                tp[i] = 1  # True positive
                detected[max_iou_idxs[i]] = True
            else:
                fp[i] = 1  # False positive (duplicate detection)
        else:
            fp[i] = 1  # False positive (IoU threshold not met)
    
    # Calculate precision and recall
    fp = torch.cumsum(fp, dim=0)
    tp = torch.cumsum(tp, dim=0)
    npos = gt_boxes.size(0)
    
    recall = tp / npos
    precision = tp / (tp + fp)
    ap = torch.trapz(precision, recall)  # Compute area under curve as approximation of AP
    
    return precision[-1], recall[-1], ap

from torchvision.ops import nms  # PyTorch NMS operation

def apply_nms(pred_boxes, pred_scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to prediction boxes.
    
    Parameters:
    - pred_boxes : Tensor of predicted bounding boxes, shape [N, 4]
    - pred_scores : Tensor of confidence scores for the predicted boxes, shape [N]
    - iou_threshold : IoU threshold for NMS
    
    Returns:
    - indices of boxes that are kept after NMS
    """
    keep = nms(pred_boxes, pred_scores, iou_threshold)
    return keep

def _crop(image, boxes, labels, mask, outpute_size=(300, 300)):
    """
    Crop the image and adjust the bounding boxes, labels, and segmentation mask accordingly.
    """
    height, width = image.shape[-2:]
    if boxes.nelement() == 0:
        image = torchvision.transforms.functional.resize(image, outpute_size)
        mask = torchvision.transforms.functional.resize(mask, outpute_size)
        return image, boxes, labels, mask

    while True:
        mode = random.choice([
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ])

        if mode is None:
            image = torchvision.transforms.functional.resize(image, outpute_size)
            mask = torchvision.transforms.functional.resize(mask, outpute_size)
            return image, boxes, labels, mask

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            new_width = random.randrange(int(0.3 * width), width)
            new_height = random.randrange(int(0.3 * height), height)

            if new_height / new_width < 0.5 or 2 < new_height / new_width:
                continue

            left = random.randrange(width - new_width)
            top = random.randrange(height - new_height)
            roi = torch.tensor([left, top, left + new_width, top + new_height], dtype=torch.float32)

            iou = bbox_iou(boxes, roi.unsqueeze(0))
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image = torchvision.transforms.functional.crop(image, top, left, new_height, new_width)
            mask = torchvision.transforms.functional.crop(mask, top, left, new_height, new_width)

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask_indices = (centers > roi[:2]) & (centers < roi[2:])
            mask_indices = mask_indices.all(dim=1)
            boxes = boxes[mask_indices]
            labels = labels[mask_indices]

            boxes[:, :2] = torch.max(boxes[:, :2], roi[:2])
            boxes[:, :2] -= roi[:2]
            boxes[:, 2:] = torch.min(boxes[:, 2:], roi[2:])
            boxes[:, 2:] -= roi[:2]
            
            image = torchvision.transforms.functional.resize(image, outpute_size)
            mask = torchvision.transforms.functional.resize(mask, outpute_size)

            return image, boxes, labels, mask

import random
        

def _mirror(image, boxes, label, mask):
    # image shape: [channels, height, width]
    _, height, width = image.shape

    if random.randrange(2):  # Randomly decide to mirror the image
        image = image[:, :, ::-1]  # Flip image horizontally
        mask = mask[:, :, ::-1]    # Flip mask horizontally
        label = label.copy()       # Ensure label is copied correctly (if needed)
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2] - 1  # Adjust bounding boxes

    return image, boxes, label, mask
