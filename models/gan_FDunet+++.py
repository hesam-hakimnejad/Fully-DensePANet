

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
#path1 = 'D:\\hesam\\phantom\\recon'
#path2 = 'D:\\hesam\\phantom\\init'

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
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid

def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]


    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)    
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])
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
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
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

# define an encoder block



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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianDropout

smooth = 1
dropout_rate = 0.5
act = "relu"
bn_axis = 3
def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    #x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    #x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x


def UNetPlusPlus(img_rows, img_cols, color_type=1, num_class=3, deep_supervision=False):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    #global bn_axis
    #if K.image_dim_ordering() == 'tf':
      #bn_axis = 3
      #img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    #else:
      #bn_axis = 1
      #img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    conv1_1 = dense_block(conv1_1,8, 32)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)
    #pool1 = attach_attention_module(pool1, 'cbam_block')

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    conv2_1 = dense_block(conv2_1,16, 64)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)
    #pool2 = attach_attention_module(pool2, 'cbam_block')

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])
    
    
    
    conv11_11 = standard_unit(conv1_2 , stage='111', nb_filter=nb_filter[0])
    pool11 = MaxPooling2D((2, 2), strides=(2, 2), name='pool11')(conv11_11)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    conv3_1 = dense_block(conv3_1,32, 128)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)
    #pool3 = attach_attention_module(pool3, 'cbam_block')

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])


    conv22_22 = standard_unit(conv2_2 , stage='222', nb_filter=nb_filter[1])
    pool22 = MaxPooling2D((2, 2), strides=(2, 2), name='pool22')(conv22_22)




    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])
    
    
    
    conv44_44 = standard_unit(conv1_3 , stage='44', nb_filter=nb_filter[0])
    pool44 = MaxPooling2D((2, 2), strides=(2, 2), name='pool44')(conv44_44)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    conv4_1= dense_block(conv4_1,64,256)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)
    #pool4 = attach_attention_module(pool4, 'cbam_block')
    
    

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])
    
    
    conv33_33 = standard_unit(conv3_2 , stage='303', nb_filter=nb_filter[2])
    pool33 = MaxPooling2D((2, 2), strides=(2, 2), name='pool33')(conv33_33)
    

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])
    
    
    
    conv55_55 = standard_unit(conv2_3 , stage='55', nb_filter=nb_filter[1])
    pool55 = MaxPooling2D((2, 2), strides=(2, 2), name='pool55')(conv55_55)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])


    conv66_66 = standard_unit(conv1_4, stage='66', nb_filter=nb_filter[0])
    pool66 = MaxPooling2D((2, 2), strides=(2, 2), name='pool66')(conv66_66)


    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])
    conv5_1 = dense_block(conv5_1,128,512)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])
    conv4_2 = dense_block(conv4_2,64,256)  


    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])
    conv3_3 = dense_block( conv3_3,32,128)  


    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])
    conv2_4 = dense_block(conv2_4,16,64) 


    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])
    conv1_5 = dense_block(conv1_5,8,32)     


    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='tanh', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='tanh', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='tanh', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='tanh', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(img_input, [nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(optimizer=opt, loss='mse')
    else:
        model = Model(img_input, nestnet_output_4)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(optimizer=opt, loss='mse')
    return model




# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
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
		if (i+1) % (bat_per_epo *5) == 0:
			summarize_performance(i, g_model, dataset)

# load image data
dataset = load_real_samples('maps_512.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = UNetPlusPlus(256,256,3)
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
model = load_model('model_016000.h5')
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
model = load_model('model_022500.h5')
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