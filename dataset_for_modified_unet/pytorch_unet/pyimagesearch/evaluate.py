import torch
#import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import config
import os
from imutils import paths
from modified_unet import modified_UNet, encoder, decoder,Block

class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_folder = os.path.join(root_dir, 'images')
        self.mask_folder = os.path.join(root_dir, 'labels')
        self.grad_cam_folder = os.path.join ( root_dir, 'grad_cam_images' )

        self.image_list = sorted(os.listdir(self.image_folder))
        self.mask_list = sorted(os.listdir(self.mask_folder))
        self.grad_cam_list = sorted (os.listdir (self.grad_cam_folder))
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_folder, 
                                self.image_list[idx])
        mask_path = os.path.join(self.mask_folder, 
                                self.mask_list[idx])
        grad_cam_path = os.path.join(self.grad_cam_folder,
                                    self.grad_cam_list[idx] )
        image = Image.open( img_path ).convert('RGB')
        mask = Image.open( mask_path ).convert('L')  # Assuming masks are grayscale
        grad_cam_img = Image.open ( grad_cam_path ).convert('RGB')
        #grad_cam_img = image
        # Apply transformations if provided
        if self.transform:
            image = self.transform( image )
            grad_cam_img = self.transform (grad_cam_img)
            mask = self.transform( mask )

        return image, grad_cam_img , mask

if __name__ == "__main__":
    def calculate_iou(pred_mask, true_mask):
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        union = torch.logical_or(pred_mask, true_mask).sum()
        iou = intersection / union.float()
        return iou.item()

    def calculate_dice(pred_mask, true_mask):
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        dice_coefficient = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum())
        return dice_coefficient.item()

    transforms = transforms.Compose([
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    testImages = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    testMasks = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    validation_dataset = \
        CustomSegmentationDataset( 
            r"dataset_for_modified_unet\pytorch_unet\dataset\test", 
            transforms )

    validation_loader = DataLoader( validation_dataset , shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
    print("here1")
    unet = modified_UNet()
    unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
    print("here2")
    unet.eval()
    total_iou = 0.0
    total_dice = 0.0
    num_samples = len(validation_dataset)

    print ( num_samples )

    with torch.no_grad():
        for entry in validation_loader:
            # Assuming your model outputs segmentation masks
            outputs = unet( entry[0], entry[1])

            # Convert logits to binary masks using a threshold or softmax
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()

            # Assuming targets are binary masks as well
            true_masks = entry[2].float()

            # Calculate IoU and Dice for each sample
            for i in range(len(entry[0])):
                iou = calculate_iou(pred_masks[i], true_masks[i])
                dice = calculate_dice(pred_masks[i], true_masks[i])

                total_iou += iou
                total_dice += dice

                print ( i )

    # Calculate average IoU and Dice across all samples
    average_iou = total_iou / num_samples
    average_dice = total_dice / num_samples

    print(f'Average IoU: {average_iou:.4f}')
    print(f'Average Dice Coefficient: {average_dice:.4f}')
