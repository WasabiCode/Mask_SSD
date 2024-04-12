import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from Y_Net_utilities import VOC2012, UnetLoss, MultiBoxEncoder_GPT, SSD_loss, simple_voc_eval
from torch.nn.utils.rnn import pad_sequence
from Y_Net import Y_Net
import wandb
import torchmetrics
import torch.nn.functional as F
import os
import torch.multiprocessing as mp

# Set start method to 'spawn'
mp.set_start_method('spawn', force=True)


cfg = {
        'grids': (38, 19, 10, 5, 3, 1),
        'steps': [8, 16, 32, 64, 100, 300],
        'sizes': [30, 60, 111, 162, 213, 264, 315],
        'aspect_ratios': ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        'variance': [0.1, 0.2]
    }
mb = MultiBoxEncoder_GPT(cfg)
SSD_loss = SSD_loss()


torch.cuda.empty_cache()



def custom_collate_fn(batch):
    imgs, targets = zip(*batch)  # Unzip the batch
    
    # Stack images as they should all be the same size after your transformations
    
    imgs = torch.stack(imgs, dim=0)
    
    # Initialize lists to hold padded targets
    padded_boxes = []
    padded_masks = []
    padded_labels = []
    
    # Determine the maximum sizes needed for padding
    max_boxes = max(len(target['boxes']) for target in targets)
    max_masks = max(len(target['masks']) for target in targets)
    max_labels = max(len(target['labels']) for target in targets)
    
    for target in targets:
        # Pad boxes
        num_boxes = len(target['boxes'])
        padded_box = torch.cat([target['boxes'], torch.zeros(max_boxes - num_boxes, 4)])
        padded_boxes.append(padded_box)
        
        # Pad masks
        num_masks = len(target['masks'])
        padded_mask = torch.cat([torch.Tensor(mask[None]) for mask in target['masks']]).long()
        padded_masks.append(padded_mask)
        
        # Pad labels
        num_labels = len(target['labels'])
        padded_label = torch.cat([target['labels'], torch.zeros(max_labels - num_labels)])
        padded_labels.append(padded_label)
        
    # Stack padded components to form batch-wise tensors
    padded_boxes = torch.stack(padded_boxes, dim=0)
    padded_masks = torch.stack(padded_masks, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    
    # Update targets for the batch
    batch_targets = {'boxes': padded_boxes, 'masks': padded_masks, 'labels': padded_labels}
    
    return imgs, batch_targets

def validate(model, val_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    val_seg_loss = 0
    val_loc_loss = 0
    val_cls_loss = 0
    # Initialize any other metrics here
    batch_val_total_loss = []
    batch_val_seg_loss = []
    batch_val_box_loss = []
    batch_val_cls_loss = []


    
    with torch.no_grad():
        for (images, targets) in val_dataloader:
            images = images.to(device)
            boxes = targets['boxes'].to(device)
            masks = targets['masks'].to(device)
            labels = targets['labels'].to(device)

            segmentation_output, (loc_preds, conf_preds) = model(images)

            for idx in range(images.size(0)):
                pred_boxes = loc_preds[idx]
                pred_labels = conf_preds[idx]
            


                gt_boxes_img = boxes[idx].to(device)
                gt_labels_img = labels[idx].to(device)

            


                encoded_boxes, encoded_labels = mb.encode(gt_boxes_img, gt_labels_img)



                localization_loss, classification_loss = SSD_loss(pred_boxes, pred_labels, encoded_boxes, encoded_labels)


                gt_masks = masks[idx]
                segmentation_loss, _ = UnetLoss(segmentation_output[idx].unsqueeze(0), gt_masks.unsqueeze(0))



                total_batch_loss = segmentation_loss + localization_loss + classification_loss

                batch_val_total_loss.append(total_batch_loss)
                batch_val_seg_loss.append(segmentation_loss)
                batch_val_box_loss.append(localization_loss)
                batch_val_cls_loss.append(classification_loss)

    # Compute average losses
    val_loss /= len(val_dataloader)
    val_seg_loss /= len(val_dataloader)
    val_loc_loss /= len(val_dataloader)
    val_cls_loss /= len(val_dataloader)
    # Compute averages of other metrics here

    print(f"Validation Loss: {val_loss}, Seg Loss: {val_seg_loss}, Loc Loss: {val_loc_loss}, Cls Loss: {val_cls_loss}")
    # Log or print other metrics here
    return val_loss, val_seg_loss, val_loc_loss, val_cls_loss


if __name__ == "__main__":
    #torch.set_num_threads(1)
    wandb.init(
    # set the wandb project where this run will be logged
    project="Mask_SSD",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "SSD_U_Net",
    "dataset": "VOC2012",
    "epochs": 100,
    }
)


    # parameters
    num_epochs = 100
    learning_rate = 1e-3
    batch_size = 16
    model = Y_Net(num_classes=21, bboxes=[4, 6, 6, 6, 4, 4])
    dataset_root = "/OV-data/mapy/vasek/VOCdevkit"
    saving_path = "/OV-data/mapy/vasek/model/Y_Net_3_adam.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    train_dataset = VOC2012(root=dataset_root, image_set='train', download=False)
    print("lenght of dataset: ", len(train_dataset))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)


    

    print("starting training..........")



    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        seg_loss = 0
        loc_loss = 0
        cls_loss = 0
        seg_accuracy = 0

        for (images, targets) in dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            boxes = targets['boxes'].to(device)
            masks = targets['masks'].to(device)
            labels = targets['labels'].to(device) # Prints device of a tensor


            #print(f"images: {images.shape}, boxes: {boxes.shape}, masks: {masks.shape}, labels: {labels.shape}")
            
            # Forward pass
            segmentation_output, (loc_preds, conf_preds) = model(images)
            #print(images.size(0))



            batch_total_losses = []
            batch_seg_loss = []
            batch_box_loss = []
            batch_cls_loss = []
            bach_seg_accuracy = []
            for idx in range(images.size(0)):
                pred_boxes = loc_preds[idx]
                pred_labels = conf_preds[idx]
                #print(f"pred_boxes: {pred_boxes.shape}, pred_labels: {pred_labels.shape}")


                gt_boxes_img = boxes[idx].to(device)
                gt_labels_img = labels[idx].to(device)

                #print(f"pred_boxes: {pred_boxes[:1]}, pred_labels: {pred_labels[:1]}, gt_boxes_img: {gt_boxes_img[:1]}, gt_labels_img: {gt_labels_img[:1]}")
                #print(f"pred_boxes: {pred_boxes.shape}, pred_labels: {pred_labels.shape}")

                #print(f"gt_boxes_img: {gt_boxes_img.shape}, gt_labels_img: {gt_labels_img.shape}")

                #match_idxs, matched_labels = match_with_iou(pred_boxes, pred_labels, gt_boxes_img, gt_labels_img)


                encoded_boxes, encoded_labels = mb.encode(gt_boxes_img, gt_labels_img)

                #mean average precision calculations




                #print(f"encoded_boxes: {encoded_boxes.shape}, encoded_labels: {encoded_labels.shape}")



                localization_loss, classification_loss = SSD_loss(pred_boxes, pred_labels, encoded_boxes, encoded_labels)

                #print(f"localization_loss: {localization_loss}")
                #print("---------------------------------------------") 
                #print(f"classification_loss: {classification_loss}")
                #print("---------------------------------------------")


                gt_masks = masks[idx]
                segmentation_loss, _ = UnetLoss(segmentation_output[idx].unsqueeze(0), gt_masks.unsqueeze(0))
                #print(f"segmentation_loss: {segmentation_loss}")
                #print("---------------------------------------------")


                total_batch_loss = segmentation_loss + localization_loss + classification_loss
                #print(f"total_loss: {total_loss}")
                #print("---------------------------------------------")
                #print("---------------------------------------------")
                batch_total_losses.append(total_batch_loss)
                batch_seg_loss.append(segmentation_loss)
                batch_box_loss.append(localization_loss)
                batch_cls_loss.append(classification_loss)
                #bach_seg_accuracy.append(segmentation_accuracy)

            total_loss_batch = sum(batch_total_losses) / len(batch_total_losses)
            seg_loss_batch = sum(batch_seg_loss) / len(batch_seg_loss)
            loc_loss_batch = sum(batch_box_loss) / len(batch_box_loss)
            cls_loss_batch = sum(batch_cls_loss) / len(batch_cls_loss)
            #seg_accuracy_batch = sum(bach_seg_accuracy) / len(bach_seg_accuracy)
            #print(f"total_loss_batch: {total_loss_batch}")
            #print(f"seg_loss_batch: {seg_loss_batch}")
            #print(f"loc_loss_batch: {loc_loss_batch}")
            #print(f"cls_loss_batch: {cls_loss_batch}")
            #print(f"seg_accuracy: {seg_accuracy_batch}")

            #print("before backward")
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            seg_loss += seg_loss_batch.item()
            loc_loss += loc_loss_batch.item()
            cls_loss += cls_loss_batch.item()
            #seg_accuracy += seg_accuracy_batch

            

            # Logging, validation, and checkpoint saving logic goes here
        print(f"Epoch [{epoch+1}/{num_epochs}], Total_Loss: {total_loss/len(dataloader):.4f}, seg_loss: {seg_loss/len(dataloader):.4f}, loc_loss: {loc_loss/len(dataloader):.4f}, cls_loss: {cls_loss/len(dataloader):.4f}")
        wandb.log({"total_loss": (total_loss/len(dataloader)), "segmentation_loss": (seg_loss/len(dataloader)), "localization_loss": (loc_loss/len(dataloader)), "classification_loss": (cls_loss/len(dataloader))})
    print("finished training")
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    torch.save(model.state_dict(), saving_path)
    print("model saved")
