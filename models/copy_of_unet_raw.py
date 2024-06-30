# -*- coding: utf-8 -*-


# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
import os
import random
import cv2
# load all images in a directory into memory
#sz = 512
def load_tarimages(path, size=(256,256)):
    tar_list= list()
    # enumerate filenames in directory, assume all are images
    
    for filename in path:
        # load and resize the image  
            pixels = load_img(filename, target_size=size)
        # convert to numpy array
            tar = img_to_array(pixels)      
            tar_list.append(tar)
        
    return asarray(tar_list)

def load_srcimages(path, size=(1280,256)):
    src_list= list()
    # enumerate filenames in directory, assume all are images
    
    for filename in path:
        # load and resize the image  
            pixels = load_img(filename, target_size=size)
        # convert to numpy array
            src = img_to_array(pixels)
            src_list.append(src)
        
    return asarray(src_list)    
   
# dataset path
path1 = 'E:\\Narges\\hesam\\dataset\\sensor2'
path2 = 'E:\\Narges\\hesam\\dataset\\init2'
#path1 = "/content/drive/My Drive/Datasets/mice_sparse/sensor2/"
#path2 =  "/content/drive/My Drive/Datasets/mice_sparse/init2/"
input_img_paths = sorted(
    [
        os.path.join(path1, fname)
        for fname in os.listdir(path1)
        
    ]
)
target_img_paths = sorted(
    [
        os.path.join(path2, fname)
        for fname in os.listdir(path2)
        
    ]
)

val_samples =205
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]





# load dataset
src_images = load_srcimages(train_input_img_paths)
tar_images = load_tarimages(train_target_img_paths)

val_src_images = load_srcimages(val_input_img_paths)
val_tar_images = load_tarimages(val_target_img_paths)


# save as compressed numpy array
filename1 = 'maps_512.npz'
savez_compressed(filename1, src_images, tar_images)
print('Saved dataset: ', filename1)
filename2 = 'maps_5120.npz'
savez_compressed(filename2, val_src_images, val_tar_images)
print('Saved dataset: ', filename2)
#%%
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
#from keras.models import tf.keras.Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Resizing

def define_encoder_block(layer_in, n_filters,s, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (3,3), strides=(s,s), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g


def decoder_block(layer_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	#g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g 

def define_unet(image_shape1=(1280,256,3)):
		# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image1 = Input(shape=image_shape1)

	# encoder model
	e11 = define_encoder_block(in_image1, 64, 1, batchnorm=False)
	e12 = define_encoder_block(e11, 128, 2)
	e13 = define_encoder_block(e12, 256, 2)
	e14 = define_encoder_block(e13, 512, 2)
	e15 = define_encoder_block(e14, 512, 2) 
	e16 = Conv2D(512, (20,3), strides=(5,1),padding='same', kernel_initializer=init)(e15)
	e16 = Activation('relu')(e16)
 
	r1 = Resizing(256,256)(e11)
	r2 = Resizing(128,128)(e12)
	r3 = Resizing(64,64)(e13)
	r4 = Resizing(32,32)(e14)

 
	d1 = decoder_block(e16, 256)
	c2 = Concatenate()([r4, d1])
	d2 = decoder_block(c2, 128)
	c3 = Concatenate()([r3, d2]) 
	d3 = decoder_block(c3, 64)
	c4 = Concatenate()([r2, d3]) 
	d4 = decoder_block(c4, 32)
	c5 = Concatenate()([r1, d4])
	d5 = Conv2DTranspose(3, (3,3), padding='same', kernel_initializer=init)(c5)
	out_image = Activation('sigmoid')(d5) 
	# define model
	model = Model(in_image1, out_image)
	opt = Adam(lr=0.0002, beta_1=0.5)	
	model.compile(optimizer=opt, loss='mse') 
	return model

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = X1/255
	X2 = X2/255
	return [X1, X2]


# select a batch of random samples, returns images and target
def generate_samples(dataset, n_samples):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	return X1, X2

def summarize_performance(step, u_model, dataset):
	
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	u_model.save(filename2)
	print('>Saved: %s' % ( filename2))
 

# train  model
def train(u_model, dataset, n_epochs=80, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = u_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_real, Y_real = generate_samples(dataset, n_batch)
		# update discriminator for real samples
		u_loss = u_model.train_on_batch(X_real, Y_real)

		# summarize performance
		print('>%d, d1[%.3f]' % (i+1, u_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, u_model, dataset)

# load image data
dataset = load_real_samples('maps_512.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
u_model = define_unet(image_shape)

# train model
train(u_model, dataset)
#%%
from tensorflow.keras.models import load_model
import skimage
from PIL import Image, ImageOps 
import numpy as np      
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
ssim_in=[]
psnr_in=[]
ssim_out=[]
psnr_out=[]
mse_in = []
mse_out = []

[X1, X2] = load_real_samples('maps_5120.npz')
model = load_model('model_032800.h5')
ix=[]
sz=256
for iix in range(205):
    ix.append(iix) 
    src_image, tar_image = X1[ix], X2[ix]
    gen_image = model.predict(src_image)
    #src_image = (src_image+1)/2
    #tar_image = (tar_image+1)/2
    #gen_image = (gen_image+1)/2
    image_array1 = np.asarray(src_image)
    image_array2 = np.asarray(gen_image)
    image_array3 = np.asarray(tar_image)
    #image_array1= np.reshape(image_array1,(sz,sz,3))
    image_array2= np.reshape(image_array2,(sz,sz,3))
    image_array3= np.reshape(image_array3,(sz,sz,3))
    value1 = ssim(image_array2,image_array3,multichannel=True)
    value2 = psnr(image_array2,image_array3)
    value3 = skimage.metrics.mean_squared_error(image_array2,image_array3)
    #value4 = ssim(image_array1,image_array3,multichannel=True)
    #value5 = psnr(image_array1,image_array3)
    #value6 = skimage.metrics.mean_squared_error(image_array1,image_array3)

    ssim_out.append(value1)
    psnr_out.append(value2)
    mse_out.append(value3)
    #ssim_in.append(value4)
    #psnr_in.append(value5)
    #mse_in.append(value6)
    ix=[]

#print(np.mean(ssim_in))
#print(np.mean(psnr_in))
#print(np.mean(mse_in))
print(np.mean(ssim_out))
print(np.mean(psnr_out))
print(np.mean(mse_out))
#%%
# example of loading a pix2pix model and using it for image to image translation
#from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import cv2
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = X1  / 255
	X2 = X2 / 255
	return [X1, X2]

# plot source, generated and target images
def plot_images(gen_img, tar_img):
	#src_img = cv2.resize(src_img,(256,256))
	images = vstack((gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	#images = (images + 1) / 2.0
	titles = ['Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 2, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()

# load dataset
[X1, X2] = load_real_samples('maps_5120.npz')
print('Loaded', X1.shape, X2.shape)
# load model
#model = load_model('model_008000.h5')
# select random example
#ix = randint(0, len(X1), 1)
ix =[8] 
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(gen_image, tar_image)