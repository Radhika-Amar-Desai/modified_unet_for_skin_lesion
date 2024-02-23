import os
import shutil
import model
import torch
import config
from torchsummary import summary

def train_test_data_split ( root_dir_src : str, 
                            root_dir_des : str,
                            train_test_data_split = 0.15 ):
    
    images_folder_path = os.path.join ( root_dir_src , 
                                       "augmented_images" )
    labels_folder_path = os.path.join ( root_dir_src ,
                                        "augmented_labels" )

    images_folder_content = \
        os.listdir ( images_folder_path )
    
    num_of_train_images = \
        int ( len ( images_folder_content ) *\
        ( 1 - train_test_data_split ) )
    
    train_folder_data = images_folder_content [ 0 : num_of_train_images ]
    test_folder_data = images_folder_content [ num_of_train_images : ]

    train_folder_images_path = os.path.join ( root_dir_des, 
                                       r"train/images")
    train_folder_labels_path = os.path.join ( root_dir_des, 
                                       r"train/labels")

    test_folder_images_path = os.path.join ( root_dir_des,
                                      r"test/images" )
    test_folder_labels_path = os.path.join ( root_dir_des, 
                                       r"test/labels")
    
    os.makedirs ( train_folder_images_path , exist_ok = True )
    os.makedirs ( test_folder_images_path , exist_ok = True )

    os.makedirs ( train_folder_labels_path , exist_ok = True )
    os.makedirs ( test_folder_labels_path , exist_ok = True )
    
    for file in train_folder_data:
        
        src_image_file = os.path.join ( images_folder_path , file )
        des_image_file = os.path.join ( train_folder_images_path , file )
        
        shutil.move ( src_image_file , des_image_file )

        src_label_file = os.path.join ( labels_folder_path , file )
        des_label_file = os.path.join ( train_folder_labels_path , file )
        
        shutil.move ( src_label_file , des_label_file )

    for file in test_folder_data:
        
        src_image_file = os.path.join ( images_folder_path , file )
        des_image_file = os.path.join ( test_folder_images_path , file )
        
        shutil.move ( src_image_file , des_image_file )

        src_label_file = os.path.join ( labels_folder_path , file )
        des_label_file = os.path.join ( test_folder_labels_path , file )
        
        shutil.move ( src_label_file , des_label_file )
    
    print ( "Done :)" )

print ( len ( os.listdir(r"SkinLesionSegmentation\dataset_for_modified_unet\pytorch_unet\dataset\train\grad_cam_images") ) )

# model = model.UNet()
# model = torch.load ( config.PARALLEL_MODEL_PATH )
# summary ( model , (3,128,128) )