# import the necessary packages
import torch
import os
# base path of the dataset
DATASET_PATH = \
    r'SkinLesionSegmentation\dataset_for_unet3\train'    
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = \
    "dataset_for_modified_unet\pytorch_unet\dataset\test\images"    

MASK_DATASET_PATH = \
    "dataset_for_modified_unet\pytorch_unet\dataset\test\labels"

GRAD_CAM_IMAGE_DATASET_PATH = \
    "dataset_for_modified_unet\pytorch_unet\dataset\test\grad_cam_images"

OUTPUT_IMAGE_PATH = \
    "dataset_for_modified\pytorch_unet\output2"

MODEL_PATH = \
    "dataset_for_modified_unet\\pytorch_unet\\output2\\unet_tgs_salt.pth"

# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 1
# define the input image dimensions
INPUT_IMAGE_WIDTH = 150
INPUT_IMAGE_HEIGHT = 150
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = \
    'C:\\Users\\97433\\unet\\SkinLesionSegmentation\\dataset_for_modified_unet\\pytorch_unet\\output'
# define the path to the output serialized model, model training
# plot, and testing image paths
#MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = "dataset_for_modified_unet\pytorch_unet\output\plot.png"
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
PARALLEL_MODEL_PATH = \
    r"dataset_for_modified_unet\pytorch_unet\output\unet_tgs_salt.pth"