

# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
import os
import random
# load all images in a directory into memory
sz = 256
def load_tarimages(path, size=(sz,sz)):
    tar_list= list()
    # enumerate filenames in directory, assume all are images
    
    for filename in path:
        # load and resize the image  
            pixels = load_img(filename, target_size=size)
        # convert to numpy array
            tar = img_to_array(pixels)      
            tar_list.append(tar)
        
    return asarray(tar_list)

def load_srcimages(path, size=(sz,sz)):
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
#path1 = 'D:\\hesam\\mouse_data\\recon3'
#path2 = 'D:\\hesam\\mouse_data\\init3'

path1 = 'D:\\hesam\\vessel\\recon'
path2 = 'D:\\hesam\\vessel\\init'
#path1 = "/content/drive/My Drive/Datasets/vessels/vessel_recon/"
#path2 = "/content/drive/My Drive/Datasets/vessels/vessel_init/"
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

val_samples = 200
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
#FD_Unet-------------------------
import tensorflow as tf
def convv1(layer_in, F):
  init = RandomNormal(stddev=0.02)
  g = Conv2D(F, (1,1), padding='same', kernel_initializer=init)(layer_in)
  g = BatchNormalization()(g)
  g = Activation('relu')(g)
  return g

def dense_block(layer_in, k, F):
  init = RandomNormal(stddev=0.02)
  g = Conv2D(F, (1,1), padding='same', kernel_initializer=init)(layer_in)
  g = BatchNormalization()(g)
  g = Activation('relu')(g)
  g = Conv2D(k, (3,3), padding='same', kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g1 = Activation('relu')(g)

  g=Concatenate()([layer_in,g1])

  g = Conv2D(F, (1,1), padding='same', kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g = Activation('relu')(g)
  g = Conv2D(k, (3,3), padding='same', kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g2 = Activation('relu')(g)

  g=Concatenate()([layer_in,g1, g2])

  g = Conv2D(F, (1,1), padding='same', kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g = Activation('relu')(g)
  g = Conv2D(k, (3,3), padding='same', kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g3 = Activation('relu')(g)

  g=Concatenate()([layer_in,g1, g2, g3])

  g = Conv2D(F, (1,1), padding='same', kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g = Activation('relu')(g)
  g = Conv2D(k, (3,3), padding='same', kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g4 = Activation('relu')(g)

  g=Concatenate()([layer_in,g1, g2, g3, g4])
  return g

def define_encoder_block(layer_in, n_filters):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	
	g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = Activation('relu')(g)
	return g


def decoder_block(layer_in, skip_in, n_filters):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout

    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g  

def FD_Unet(image_shape=(sz,sz,3)):
  init = RandomNormal(stddev=0.02)
  # define the standalone generator model
  image_shape=(sz,sz,3)
    # weight initialization
    # image input
  in_image = tf.keras.Input(shape=image_shape) 
  e1 = define_encoder_block(in_image, 32) 
  e1 = dense_block(e1,8, 32)
  m1 = MaxPooling2D(pool_size=(2, 2))(e1)  

  e2 = dense_block(m1,16, 64)
  m2 = MaxPooling2D(pool_size=(2, 2))(e2)

  e3 = dense_block(m2,32, 128)
  m3 = MaxPooling2D(pool_size=(2, 2))(e3)

  e4 = dense_block(m3,64,256)
  m4 = MaxPooling2D(pool_size=(2, 2))(e4)

  e5 = dense_block(m4,128,512)

  d4 = decoder_block(e5, e4, 512)
  d4 = convv1(d4,256)
  d4 = dense_block(d4,64,256)  

  d3 = decoder_block(d4, e3, 256)
  d3 = convv1(d3,128)
  d3 = dense_block(d3,32,128)  

  d2 = decoder_block(d3, e2, 128)
  d2 = convv1(d2,64)
  d2 = dense_block(d2,16,64) 


  d1 = decoder_block(d2, e1, 64)
  d1 = convv1(d1,32)
  d1 = dense_block(d1,8,32)     

  out = Conv2D(3, (1,1), padding='same', kernel_initializer=init)(d1)
  out = Activation('sigmoid')(out)
  opt = Adam(lr=0.0002, beta_1=0.5)
  model = Model(in_image, out)
  model.compile(optimizer=opt, loss='mse')

  return model
#%%
from numpy import load
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
#from tensorflow.keras.models import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D
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
u_model = FD_Unet(image_shape)

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
model = load_model('model_018000.h5')
ix=[]
sz=256
for iix in range(200):
    ix.append(iix) 
    src_image, tar_image = X1[ix], X2[ix]
    gen_image = model.predict(src_image)

    image_array1 = np.asarray(src_image)
    image_array2 = np.asarray(gen_image)
    image_array3 = np.asarray(tar_image)
    image_array1= np.reshape(image_array1,(sz,sz,3))
    image_array2= np.reshape(image_array2,(sz,sz,3))
    image_array3= np.reshape(image_array3,(sz,sz,3))

    value1 = ssim(image_array2,image_array3,multichannel=True)
    value2 = psnr(image_array2,image_array3)
    value3 = skimage.metrics.mean_squared_error(image_array2,image_array3)
    value4 = ssim(image_array1,image_array3,multichannel=True)
    value5 = psnr(image_array1,image_array3)
    value6 = skimage.metrics.mean_squared_error(image_array1,image_array3)

    ssim_out.append(value1)
    psnr_out.append(value2)
    mse_out.append(value3)
    ssim_in.append(value4)
    psnr_in.append(value5)
    mse_in.append(value6)
    ix=[]

print(np.mean(ssim_in))
print(np.mean(psnr_in))
print(np.mean(mse_in))
print(np.mean(ssim_out))
print(np.mean(psnr_out))
print(np.mean(mse_out))
#%%
# example of loading a pix2pix model and using it for image to image translation
from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint

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
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	#images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
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
model = load_model('model_054000.h5')
# select random example
#ix = randint(0, len(X1), 1)
ix =[8] 
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)