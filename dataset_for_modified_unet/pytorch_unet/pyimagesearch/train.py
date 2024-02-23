import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import config
from modified_unet import modified_UNet
import time
from model import UNet
import numpy as np
import cv2
import torch
import psutil

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

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

transforms = transforms.Compose([
    transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		            config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

train_root_dir = \
    r"SkinLesionSegmentation\dataset_for_modified_unet\pytorch_unet\dataset\train"

test_root_dir = \
    r"SkinLesionSegmentation\dataset_for_modified_unet\pytorch_unet\dataset\test"

trainDS = CustomSegmentationDataset ( train_root_dir , transforms )
testDS = CustomSegmentationDataset ( test_root_dir , transforms )

trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count(), drop_last=True)

testLoader = DataLoader(testDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count(),drop_last=True)

unet = modified_UNet()
#unet = UNet()

lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

if __name__ == "__main__":
    H = {"train_loss": [], "test_loss": []}
    print("[INFO] training the network...")
    startTime = time.time()

    for e in tqdm(range(config.NUM_EPOCHS)):
        # Get CPU usage
        cpu_usage = psutil.cpu_percent()
        print("CPU Usage: {}%".format(cpu_usage))

        # Get memory usage
        memory_usage = psutil.virtual_memory().percent
        print("Memory Usage: {}%".format(memory_usage))
        print("Epoch : ", e)
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        for (i,(x1,x2,y)) in enumerate(trainLoader):
            # send the input to the device
            #print ( x1.shape, x2.shape, y.shape )
            (x1,x2, y) = (x1.to(config.DEVICE),
                        x2.to(config.DEVICE),
                        y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = unet(x = x1, grad_cam_tensor_x = x2)
            #pred = unet( x1 )
            loss = lossFunc(pred, y)
            print ( i )
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward(retain_graph=True)
            #clip_grad_norm_(value_model.parameters(), clip_grad_norm)
            # loss.backward()
            opt.step()
            # loss.detach()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
            
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for index,(x1, x2 , y) in enumerate(testLoader):
                # send the input to the device
                (x1, x2 , y) = (x1.to(config.DEVICE), 
                        x2.to(config.DEVICE),
                        y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                #print ( "x1.shape : ", x1.shape, "x2.shape : ", x2.shape, "y.shape : ", y )
                pred = unet(x = x1, grad_cam_tensor_x = x2)
                #pred = unet(x1)
                totalTestLoss += lossFunc(pred, y)
                #print ( index )
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)
    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH)