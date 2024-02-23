from keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import keras
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import os
import shutil
from copy import deepcopy
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow logging level to suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore")

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        
        gradModel = Model(
            
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, tf.argmax(predictions[0])]
        
        grads = tape.gradient(loss, convOutputs)
        
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        (w, h) = (image.shape[2], image.shape[1])
        
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)

def apply_gradcam_on_custom_model ( image : list ,
                                    layer_name = "max_pooling2d_2",
                                    model_path = 
                                    "C:\\Users\\97433\\unet\\custom_model.h5" ):

    # print(image_path.split("\\")[-1])
    # image = cv2.imread( image_path )
    image = np.array(image)  
    image = cv2.resize(image, (150, 150))
    org = deepcopy( image )
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)

    model = keras.models.load_model ( model_path )

    preds = model.predict(image) 
    #print ( preds.tolist() )
    i = np.argmax(preds[0])

    icam = GradCAM(model, i, layer_name ) 
    heatmap = icam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (150,150))

    # image = cv2.imread(image_path)  
    # image = cv2.resize(image, (150, 150))
    # print(heatmap.shape, image.shape)

    (heatmap, output) = icam.overlay_heatmap(heatmap, org, alpha=0.5)

    resized_img = cv2.resize ( output , (765,574) )

    return resized_img

# def get_grad_cam_images ( root_dir :str, grad_cam_dir : str , 
#                         des_dir : str ):
    
#     folder_content = os.listdir ( root_dir )

#     for file in folder_content:
        
#         src_image_path = os.path.join ( grad_cam_dir , file )
#         des_image_path = os.path.join ( des_dir , file )

#         shutil.move ( src_image_path , des_image_path )
    
#     print ("Done :)")

# get_grad_cam_images ( des_dir = r"SkinLesionSegmentation\dataset_for_modified_unet\pytorch_unet\dataset\test\grad_cam_images",
#                     root_dir = r"SkinLesionSegmentation\dataset_for_modified_unet\pytorch_unet\dataset\test\images",
#                     grad_cam_dir = r"SkinLesionSegmentation\dataset_for_modified_unet\gradcam_augmented_images")
