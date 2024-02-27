#import necessary libraries
import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
import torchvision.transforms as transforms
import cv2
from torch.nn import functional as F
import torch
from torchsummary import summary
import model
from model import UNet
import numpy as np
from PIL import Image
# import GradCAM

import os
import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow logging level to suppress warnings
# warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings

#Load Model
# model = model.UNet()
# model = torch.load ( config.PARALLEL_MODEL_PATH )

outSize = \
    (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)

#Customizing forward function to return decoder features
enc_output = []
dec_input = []
dec_output = []

def convert_img_to_tensor ( image = None ):

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.INPUT_IMAGE_HEIGHT, 
                               config.INPUT_IMAGE_WIDTH))
    image = image.astype("float32") / 255.0

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(config.DEVICE)
    
    return image

def get_decoder_features ( model , x ):

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

def get_prediction ( model , image ):
    image = convert_img_to_tensor ( image )
    model.eval()
    with torch.no_grad():
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)
        predMask = cv2.resize ( predMask , ( 150,150 ) )
    return predMask

# def get_grad_cam_saliency_map ( image = None ):

#     grad_cam_img = GradCAM.apply_gradcam_on_custom_model ( image )

#     return grad_cam_img

# org_image = Image.open ( r"dataset_for_modified_unet\pytorch_unet\dataset\train\grad_cam_images\IMD003_0.jpg")
# image = get_prediction ( model, org_image )

# cv2.imshow ( "Image" , image )
# cv2.waitKey( 0 )

# t = torch.randn (1,3,128,128)
# decoder_input = get_decoder_features ( UNet() , t ) 
# print ( [ x.shape for x in decoder_input ] )