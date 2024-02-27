# import the necessary packages
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
#from torchsummary import summary
#from utils import get_decoder_features
#from model import UNet

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
	
class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(int(channels[i] * 3/2), channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, grad_cam_x, 
				decoder_output1,decoder_output2,
				encFeatures):
		# loop through the number of channels
		#decoder_features = get_decoder_features(UNet(), grad_cam_x)
		decoder_output = [ decoder_output1, decoder_output2 ]
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			
			#print ( "After upconv : ", x.shape )
			encFeat = self.crop( encFeatures[i], x )
			x = torch.cat([x, encFeat], dim=1)
			#print ( "Decoder-encoder  : ", x.shape )
			grad_cam_x = decoder_output[i]
			x = self.crop( x, grad_cam_x )
			#print ( "x.shape" , x.shape )
			#print ( "grad_cam_x.shape", grad_cam_x.shape )
			x = torch.cat([x, grad_cam_x], dim=1)
			#print (	"Decoder-another decoder : ", x.shape )
			x = self.dec_blocks[i](x)
			#print ( "Decoder block output : ", x.shape )
		
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		# print ( "encFeatures.shape : ", encFeatures.shape )
		# print ( "x.shape : ", x.shape )
		#print ( "x.shape : " , x.shape )
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures

class modified_UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64),
		 decChannels=(64, 32, 16),
		 nbClasses=1, retainDim=True,
		 outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize

	def forward(self, x, grad_cam_tensor_x, 
			decoder_output1, decoder_output2):
			# grab the features from the encoder
			encFeatures = self.encoder(x)
			# pass the encoder features through decoder making sure that
			# their dimensions are suited for concatenation
			decFeatures = self.decoder(
				x = encFeatures[::-1][0], 
				grad_cam_x = grad_cam_tensor_x,
				decoder_output1 = decoder_output1,
				decoder_output2 = decoder_output2,
				encFeatures = encFeatures[::-1][1:])
			# pass the decoder features through the regression head to
			# obtain the segmentation mask
			map = self.head(decFeatures)
			# check to see if we are retaining the original output
			# dimensions and if so, then resize the output to match them
			if self.retainDim:
				map = F.interpolate(map, self.outSize)
			# return the segmentation map
			return map

model_for_seg = modified_UNet()
t = torch.randn ( 1, 3, 128, 128)
grad_cam_t = torch.randn ( 1, 3, 128, 128 )

t1 = torch.randn ( 1, 32, 56, 56 )
t2 = torch.randn ( 1, 16, 108, 108 )

decoder_output = [ t1 , t2 ]
# print ( model_for_seg ( t, grad_cam_t, t1, t2 ) )

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 16, 126, 126]             448
#               ReLU-2         [-1, 16, 126, 126]               0
#             Conv2d-3         [-1, 16, 124, 124]           2,320
#              Block-4         [-1, 16, 124, 124]               0
#          MaxPool2d-5           [-1, 16, 62, 62]               0
#             Conv2d-6           [-1, 32, 60, 60]           4,640
#               ReLU-7           [-1, 32, 60, 60]               0
#             Conv2d-8           [-1, 32, 58, 58]           9,248
#              Block-9           [-1, 32, 58, 58]               0
#         MaxPool2d-10           [-1, 32, 29, 29]               0
#            Conv2d-11           [-1, 64, 27, 27]          18,496
#              ReLU-12           [-1, 64, 27, 27]               0
#            Conv2d-13           [-1, 64, 25, 25]          36,928
#             Block-14           [-1, 64, 25, 25]               0
#         MaxPool2d-15           [-1, 64, 12, 12]               0
#           Encoder-16  [[-1, 16, 124, 124], [-1, 32, 58, 58], [-1, 64, 25, 25]]               0
#   ConvTranspose2d-17           [-1, 32, 50, 50]           8,224
#            Conv2d-18           [-1, 32, 48, 48]          18,464
#              ReLU-19           [-1, 32, 48, 48]               0
#            Conv2d-20           [-1, 32, 46, 46]           9,248
#             Block-21           [-1, 32, 46, 46]               0
#   ConvTranspose2d-22           [-1, 16, 92, 92]           2,064
#            Conv2d-23           [-1, 16, 90, 90]           4,624
#              ReLU-24           [-1, 16, 90, 90]               0
#            Conv2d-25           [-1, 16, 88, 88]           2,320
#             Block-26           [-1, 16, 88, 88]               0
#           Decoder-27           [-1, 16, 88, 88]               0
#            Conv2d-28            [-1, 1, 88, 88]              17
# ================================================================
# Total params: 117,041
# Trainable params: 117,041
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.19
# Forward/backward pass size (MB): 12197.77
# Params size (MB): 0.45
# Estimated Total Size (MB): 12198.41
# ----------------------------------------------------------------