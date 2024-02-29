import utils
from model import UNet
import config
import cv2
import torch
import os

def save_decoder_outputs_from_parallel_model ( 
    image_folder : str, image_file : str, 
    tensor_folder : str
 ):
    if not os.path.exists ( tensor_folder ): 
        os.makedirs ( tensor_folder )
    image_file_path = os.path.join ( image_folder, 
                                    image_file )

    #tensor_folder = r"dataset_for_modified_unet\pytorch_unet\dataset\train\decoder_tensors"
    tensor_file = image_file.split(".")[0] + ".pt"
    tensor_file_path = os.path.join ( tensor_folder, 
                                    tensor_file )

    image = cv2.imread ( image_file_path )

    decoder_features = \
        utils.get_decoder_features( utils.model, 
                utils.convert_img_to_tensor ( image ) )

    torch.save ( decoder_features, tensor_file_path )

def folder_save_decoder_outputs_from_parallel_model (
    image_folder : str, tensor_folder : str
):
    if not os.path.exists ( tensor_folder ): 
        os.makedirs ( tensor_folder )
    for image_file in os.listdir ( image_folder ):
        save_decoder_outputs_from_parallel_model (
            image_folder = image_folder,
            image_file = image_file,
            tensor_folder = tensor_folder
        )

def rotate_and_save_image ( image_file_path : str ):
    image = cv2.imread ( image_file_path )
    def rotate_image ( image ):
        return [ image, 
                cv2.rotate ( image, cv2.ROTATE_180 ),
                cv2.rotate ( image, cv2.ROTATE_90_CLOCKWISE ),
                cv2.rotate ( image, cv2.ROTATE_90_COUNTERCLOCKWISE) ]

    image_folder =\
        "\\".join(image_file_path.split("\\")[:-1])
    image_file = image_file_path.split("\\")[-1]
    image_name = image_file.split(".")[0] 
    extension = image_file.split(".")[-1]

    for i in range ( 4 ):
        new_image_file_name =\
            image_name + "_" + str(i) + "." + extension
        
        new_file_path = os.path.join ( image_folder,
                                    new_image_file_name )

        print ( new_file_path )

        cv2.imwrite ( new_file_path, 
                    rotate_image ( image ) [ i ] )

def folder_rotate_and_save_image ( image_folder : str ):
    for image_file in os.listdir ( image_folder ):
        image_file_path = \
            os.path.join ( image_folder, image_file )
        rotate_and_save_image ( image_file_path )

#folder_rotate_and_save_image ( r"dataset_for_modified_unet\pytorch_unet\dataset\train\labels" )

image_folder = \
    "dataset_for_modified_unet/pytorch_unet/dataset/train/grad_cam_images"
tensor_folder = \
    "dataset_for_modified_unet/pytorch_unet/dataset/train/decoder_tensors"

folder_save_decoder_outputs_from_parallel_model ( 
    image_folder = image_folder,
    tensor_folder = tensor_folder)

# tensor_file = \
#     r"dataset_for_modified_unet\pytorch_unet\dataset\train\decoder_tensors\IMD003_0.pt"
# print ( torch.load ( tensor_file )[1].shape )
