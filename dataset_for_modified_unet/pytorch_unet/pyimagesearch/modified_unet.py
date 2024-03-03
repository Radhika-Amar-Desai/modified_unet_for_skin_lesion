#import the necessary packages
import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torchsummary import summary
#from utils import get_decoder_features
from model import UNet
from PIL import Image
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
import time
from model import UNet
import numpy as np
import cv2
import torch
import psutil
import gc

#print("modified_unet import successful")
model = UNet()
model = torch.load ( config.PARALLEL_MODEL_PATH )

enc_output = []
dec_output = []

def get_decoder_features ( x ):

    encoder = model.encoder
    decoder = model.decoder

    #Encoder ( including bottleneck )
    for idx , enc_block in enumerate ( encoder.encBlocks ):
        
        if idx < len ( encoder.encBlocks ) - 1:
            x = enc_block ( x )
            x = encoder.pool ( x )
            enc_output.append ( x )
        
        else :
            x = enc_block ( x )
            enc_output.append ( x )

    #Decoder
    enc_features = enc_output[::-1][1:]

    for idx , dec_block in enumerate ( decoder.dec_blocks ):
        #dec_input.append ( x )
        x1 = decoder.upconvs[ idx ] ( x )
        encFeat = decoder.crop ( enc_features[ idx ] , x1 )
        x2 = torch.cat ( [ x1 , encFeat ] , dim = 1 )
        x3 = dec_block ( x2 )
        dec_output.append ( x3 )
        x = x3

    return dec_output

class Block(Module):
		def __init__(self, inChannels, outChannels):
			super().__init__()
			# store the convolution and RELU layers
			self.conv1 = Conv2d(inChannels, outChannels, 3)
			self.relu = ReLU()
			self.conv2 = Conv2d(outChannels, outChannels, 3)
		def forward(self, x):
			# apply CONV => RELU => CONV block to the inputs and return it
			return self.conv2(self.relu(self.conv1(x)))
		
class encoder(Module):
		def __init__(self, channels=(3, 16, 32, 64)):
			super().__init__()
			#store the encoder blocks and maxpooling layer
			self.encBlocks = ModuleList(
				[Block(channels[i], channels[i + 1])
					for i in range(len(channels) - 1)])
			self.pool = MaxPool2d(2)
		def forward(self, x):
			#initialize an empty list to store the intermediate outputs
			blockOutputs = []
			#loop through the encoder blocks
			for block in self.encBlocks:
				#pass the inputs through the current encoder block, store
				#the outputs, and then apply maxpooling on the output
				x = block(x)
				blockOutputs.append(x)
				x = self.pool(x)
			#return the list containing the intermediate outputs
			return blockOutputs

class decoder(Module):
		def __init__(self, channels=(64, 32, 16)):
			super().__init__()
			#initialize the number of channels, upsampler blocks, and
			#decoder blocks
			self.channels = channels
			self.upconvs = ModuleList(
				[ConvTranspose2d(64, 32, 2, 2),
				ConvTranspose2d(64, 16, 2, 2)])
			self.dec_blocks = ModuleList(
				[Block(channels[i] , channels[i + 1])
					for i in range(len(channels) - 1)])
		def forward(self, x, grad_cam_x, encFeatures):
			#loop through the number of channels
			decoder_features = get_decoder_features(grad_cam_x)
			for i in range(len(self.channels) - 1):
				#pass the inputs through the upsampler blocks
				x = self.upconvs[i](x)
				#crop the current features from the encoder blocks,
				#concatenate them with the current upsampled features,
				#and pass the concatenated output through the current
				#decoder block
				#print ( "After upconv : ", x.shape )
				encFeat = self.crop( encFeatures[i], x )
				x = torch.cat([x, encFeat], dim=1)
				#print ( "Decoder-encoder  : ", x.shape )
				grad_cam_x = decoder_features[i]
				grad_cam_x = grad_cam_x.cpu().detach()
				grad_cam_x.requires_grad = False
                #x = self.crop( x, grad_cam_x )
				#print ( "x.shape" , x.shape )
				#print ( "grad_cam_x.shape", grad_cam_x.shape )
				#x = torch.cat([x, grad_cam_x], dim=1)
				x = self.dec_blocks[i](x)
				#print ( "After Dec_block : ", x.shape )
				x = self.crop( x, grad_cam_x )
				x = torch.cat([x, grad_cam_x], dim=1)
				#print ( "Decoder block output : ", x.shape )
			#return the final decoder output
			return x

		def crop(self, encFeatures, x):
			#grab the dimensions of the inputs, and crop the encoder
			#features to match the dimensions
			#print ( "encFeatures.shape : ", encFeatures.shape )
			#print ( "x.shape : ", x.shape )
			(_, _, H, W) = x.shape
			encFeatures = CenterCrop([H, W])(encFeatures)
			#return the cropped features
			return encFeatures

class modified_UNet(Module):
		def __init__(self, encChannels=(3, 16, 32, 64),
			decChannels=(64, 32, 16),
			nbClasses=1, retainDim=True,
			outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
			super().__init__()
			#initialize the encoder and decoder
			self.Encoder = encoder(encChannels)
			self.Decoder = decoder(decChannels)
			#initialize the regression head and store the class variables
			self.head = Conv2d(decChannels[-2], 
					        nbClasses, 1)
			self.retainDim = retainDim
			self.outSize = outSize

		def forward(self, x, grad_cam_tensor_x):
				#grab the features from the encoder
				encFeatures = self.Encoder(x)
				#pass the encoder features through decoder making sure that
				#their dimensions are suited for concatenation
				decFeatures = self.Decoder(
					x = encFeatures[::-1][0], 
					grad_cam_x = grad_cam_tensor_x,
					encFeatures = encFeatures[::-1][1:])
				#pass the decoder features through the regression head to
				#obtain the segmentation mask
				map = self.head(decFeatures)
				#check to see if we are retaining the original output
				#dimensions and if so, then resize the output to match them
				if self.retainDim:
					map = F.interpolate(map, self.outSize)
				#return the segmentation map
				return map

class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_folder = os.path.join(root_dir, 
										'images')
        self.mask_folder = os.path.join(root_dir, 
										'labels')
        self.grad_cam_folder = os.path.join ( root_dir, 
											'grad_cam_images' )

        self.image_list = sorted(os.listdir(self.image_folder))
        self.mask_list = sorted(os.listdir(self.mask_folder))
        self.grad_cam_list = sorted (os.listdir (self.grad_cam_folder))
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #Load image and mask
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
        #Apply transformations if provided
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
    r"dataset_for_modified_unet\pytorch_unet\dataset\train"

test_root_dir = \
    r"dataset_for_modified_unet\pytorch_unet\dataset\test"

trainDS = CustomSegmentationDataset ( train_root_dir , transforms )
testDS = CustomSegmentationDataset ( test_root_dir , transforms )

trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count(), drop_last=True)

testLoader = DataLoader(testDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count(),drop_last=True)

unet = modified_UNet().to(config.DEVICE)
# t = torch.randn(1,3,128,128)
# unet( t , t )

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
            loss.backward()
            opt.step()
            loss.cpu().detach().numpy()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # # Clear cache
        del x1, x2, y, pred, loss
        gc.collect()
		
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
                test_loss = lossFunc(pred, y)
                totalTestLoss += test_loss
                test_loss.cpu().detach().numpy()
                #print ( index )
            del x1 , x2, y, pred, test_loss
            gc.collect()
			
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

# model_for_seg = modified_unet()
# t = torch.randn ( 1, 3, 128, 128)
# grad_cam_t = torch.randn ( 1, 3, 128, 128 )
# print ( model_for_seg ( t, grad_cam_t ) )
