

# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
import os
import random
#import cv2
# load all images in a directory into memory
def load_tarimages(path, size=(256,256)):
    tar_list= list()
    # enumerate filenames in directory, assume all are images
    
    for filename in path:
        # load and resize the image  
            pixels = load_img(filename, target_size=size)
        # convert to numpy array
            tar = img_to_array(pixels)
            #tar = cv2.resize(tar,(256,256))
      
            tar_list.append(tar)
        
    return asarray(tar_list)

def load_srcimages(path, size=(256,256)):
    src_list= list()
    # enumerate filenames in directory, assume all are images
    
    for filename in path:
        # load and resize the image  
            pixels = load_img(filename, target_size=size)
        # convert to numpy array
            src = img_to_array(pixels)
            #src = cv2.resize(src,(256,256))
            src_list.append(src)
        
    return asarray(src_list)    
   
# dataset path
path1 = 'D:\\hesam\\mouse_data\\recon3'
path2 = 'D:\\hesam\\mouse_data\\init3'
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

val_samples = 500
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
# load the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the dataset
data = load('maps_512.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Loaded: ', src_images.shape, tar_images.shape)
# plot source images
n_samples =3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()

import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
def perceptual_loss(img_true, img_generated):
    image_shape = (512,512,3)
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_block3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_block3.trainable = False
    loss_block2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
    loss_block2.trainable = False
    loss_block1 = Model(input=vgg.input, outputs = vgg.get_layer('block1_conv2').output)
    loss_block1.trainable = False
    return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2*K.mean(K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5*K.mean(K.square(loss_block3(img_true) - loss_block3(img_generated)))
#%%
# example of pix2pix gan for satellite to map image-to-image translation
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
from tensorflow.keras.layers import Dropout, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
import tensorflow as tf
# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = tf.keras.Input(shape=image_shape)
	# target image input
	in_target_image = tf.keras.Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	s=3
	d = Conv2D(64, (s,s), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (s,s), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (s,s), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (s,s), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (s,s), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (s,s), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	#model.compile(loss='mse', optimizer=opt, loss_weights=[0.5])
	return model





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
sz=256
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
  out = Activation('tanh')(out)
  opt = Adam(lr=0.0002, beta_1=0.5)
  model = Model(in_image, out)
  model.compile(optimizer=opt, loss='mse')

  return model
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = tf.keras.Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	#model.compile(loss=['mse', 'mae'], optimizer=opt, loss_weights=[1,100])

	return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=80, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)

# load image data
dataset = load_real_samples('maps_512.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = FD_Unet(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)



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
	X1 = (X1-127.5) /127.5
	X2 = (X2-127.5) /127.5

	return [X1, X2]

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
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
model = load_model('model_065600.h5')
# select random example
#ix = randint(0, len(X1), 1)
ix =[40] 
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)
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
model = load_model('model_045000.h5')
ix=[]
sz=256
for iix in range(500):
    ix.append(iix) 
    src_image, tar_image = X1[ix], X2[ix]
    gen_image = model.predict(src_image)
    src_image = (src_image+1)/2
    tar_image = (tar_image+1)/2
    gen_image = (gen_image+1)/2
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