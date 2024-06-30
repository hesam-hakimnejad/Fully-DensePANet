
# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
import os
import random
import numpy
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
        src_list.append(src)
    return asarray(src_list)

def load_senimages(path, size=(1280,256)):
    sen_list= list()
    # enumerate filenames in directory, assume all are images
    
    for filename in path:
    # load and resize the image  
        pixels = load_img(filename, target_size=size)
    # convert to numpy array
        sen = img_to_array(pixels)
        sen_list.append(sen)       
    return asarray(sen_list)  
# dataset path
path3 = 'D:\\hesam\\vessel\\recon'
path4 = 'D:\\hesam\\vessel\\init'
path5 = 'D:\\hesam\\vessel\\sensor'
#path3 = "/content/drive/My Drive/Datasets/vessels/vessel_recon/"
#path4 = "/content/drive/My Drive/Datasets/vessels/vessel_init/"
#path5 = "/content/drive/My Drive/Datasets/vessels/vessel_sensor/"
input_img_paths = sorted(
    [
        os.path.join(path3, fname)
        for fname in os.listdir(path3)
        
    ]
)
target_img_paths = sorted(
    [
        os.path.join(path4, fname)
        for fname in os.listdir(path4)
        
    ]
)

sensor_img_paths = sorted(
    [
        os.path.join(path5, fname)
        for fname in os.listdir(path5)
        
    ]
)
val_samples = 200
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
random.Random(1337).shuffle(sensor_img_paths)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
train_sensor_img_paths = sensor_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]
val_sensor_img_paths = sensor_img_paths[-val_samples:]

# load dataset
src_images = load_srcimages(train_input_img_paths)
tar_images = load_tarimages(train_target_img_paths)
sen_images = load_senimages(train_sensor_img_paths)

val_src_images = load_srcimages(val_input_img_paths)
val_tar_images = load_tarimages(val_target_img_paths)
val_sen_images = load_senimages(val_sensor_img_paths)


# save as compressed numpy array

filename1 = 'maps_64.npz'
savez_compressed(filename1, src_images, sen_images, tar_images)
print('Saved dataset: ', filename1)
filename2 = 'maps_640.npz'
savez_compressed(filename2, val_src_images,val_sen_images, val_tar_images)
print('Saved dataset: ', filename2)# load, split and scale the maps dataset ready for training
#%%
# load the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the dataset
data = load('maps_64.npz')
src_images, sen_images, tar_images = data['arr_0'], data['arr_1'], data['arr_2']
print('Loaded: ', src_images.shape, sen_images.shape, tar_images.shape)
# plot source images
n_samples =3
for i in range(n_samples):
    pyplot.subplot(3, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
    pyplot.subplot(3, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(sen_images[i].astype('uint8'))
 
for i in range(n_samples):
    pyplot.subplot(3, n_samples, 1 +2*(n_samples) + i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].astype('uint8')) 
pyplot.show()
#%%
...
from instancenormalization import InstanceNormalization
# define layer
layer = InstanceNormalization(axis=-1)
...
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
# -----------------------------------------------------------------------------l1 InstanceNormalization CBAM-----------------------------------------------------------
# example of defining a u-net encoder-decoder generator model
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
from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.layers import Resizing
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Resizing

# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image =  tf.keras.Input(shape=image_shape)
    # target image input
    in_target_image =  tf.keras.Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # C128
    d = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # C256
    d = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d =InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # C512
    d = Conv2D(512, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # second last output layer
    d = Conv2D(512, (3,3), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # patch output
    d = Conv2D(1, (3,3), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    #model.compile(loss='mse', optimizer=opt, loss_weights=[0.5])
    return model
def define_encoder_block(layer_in, n_filters,s, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (3,3), strides=(s,s), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g =InstanceNormalization(axis=-1)(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = InstanceNormalization(axis=-1)(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    #g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g 

def define_generator(image_shape1=(1280,256,3),image_shape2=(256,256,3)):
      # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image1 =  tf.keras.Input(shape=image_shape1)
    in_image2 =  tf.keras.Input(shape=image_shape2)

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


 
    e21 = define_encoder_block(in_image2, 64, 1, batchnorm=False)
    e22 = define_encoder_block(e21, 128, 2)
    e23 = define_encoder_block(e22, 256, 2)
    e24 = define_encoder_block(e23, 512, 2)
    e25 = define_encoder_block(e24, 512, 2) 
 



    c1 = Concatenate()([e25, e16])
    d1 = decoder_block(c1, 256)
    c2 = Concatenate()([e24, r4, d1])
    d2 = decoder_block(c2, 128)
    c3 = Concatenate()([e23, r3, d2]) 
    d3 = decoder_block(c3, 64)
    c4 = Concatenate()([e22, r2, d3]) 
    d4 = decoder_block(c4, 32)
    c5 = Concatenate()([e21, r1, d4])
    d5 = Conv2DTranspose(3, (3,3), padding='same', kernel_initializer=init)(c5)
    out_image = Activation('tanh')(d5) 
    # define model
    model = Model([in_image1,in_image2], out_image)
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
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape1, image_shape2):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src1 =  tf.keras.Input(shape=image_shape1)
    in_src2 =  tf.keras.Input(shape=image_shape2)

    # connect the source image to the generator input
    gen_out = g_model([in_src1, in_src2])
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src2, gen_out])
    # src image as input, generated image and classification output
    model = Model([in_src1, in_src2], [dis_out, gen_out])
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
    X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5

    return [X1, X2, X3]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB, trainC = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2, X3 = trainA[ix], trainB[ix], trainC[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2, X3], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples1, samples2, patch_shape):
    # generate fake instance
    X = g_model.predict([samples1, samples2])
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB, X_realC], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realB, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_realC = (X_realB + 1) / 2.0
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
        pyplot.imshow(X_realC[i])
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
    trainA,  trainB, trainC = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB, X_realC], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model,X_realB, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realC], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch([X_realB,X_realA], [y_real, X_realC])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo * 5) == 0:
            summarize_performance(i, g_model, dataset)

# load image data
dataset = load_real_samples('maps_64.npz')
print('Loaded', dataset[0].shape, dataset[1].shape,  dataset[2].shape )
# define input shape based on the loaded dataset
image_shape1 = dataset[1].shape[1:]
image_shape2 = dataset[0].shape[1:]

# define the models
d_model = define_discriminator(image_shape2)
g_model = define_generator(image_shape1, image_shape2)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape1, image_shape2)
# train model
train(d_model, g_model, gan_model, dataset)
#%%
#-----------------------------------------------------------l2 instancenormalization cbam-------------------------------------------------
# example of defining a u-net encoder-decoder generator model
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
from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.layers import Resizing
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Resizing

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
    d = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # C128
    d = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # C256
    d = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # C512
    d = Conv2D(512, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = attach_attention_module(d, 'cbam_block')

    # second last output layer
    d = Conv2D(512, (3,3), padding='same', kernel_initializer=init)(d)
    d =InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (3,3), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    #model.compile(loss='mse', optimizer=opt, loss_weights=[0.5])
    return model
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

def decoder_block(layer_in, n_filters,s, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3,3), strides=(s,s), padding='same', kernel_initializer=init)(layer_in)
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

def define_generator(image_shape1=(1280,256,3),image_shape2=(256,256,3)):
      # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image1 = Input(shape=image_shape1)
    in_image2 = Input(shape=image_shape2)

    # encoder model
    e11 = define_encoder_block(in_image1, 64, 1, batchnorm=False)
    e11 = define_encoder_block(e11, 64,1, batchnorm=False)
    e12 = define_encoder_block(e11, 128, 1)
    e12 = define_encoder_block(e12, 128, 2)
    e13 = define_encoder_block(e12, 256, 1)
    e13 = define_encoder_block(e13, 256, 2)
    e14 = define_encoder_block(e13, 512, 1)
    e14 = define_encoder_block(e14, 512, 2)
    e15 = define_encoder_block(e14, 512, 1) 
    e15 = define_encoder_block(e15, 512, 2)
    e16 = Conv2D(512, (20,3), strides=(5,1),padding='same', kernel_initializer=init)(e15)
    e16 = Activation('relu')(e16)
 
    r1 = Resizing(256,256)(e11)
    r2 = Resizing(128,128)(e12)
    r3 = Resizing(64,64)(e13)
    r4 = Resizing(32,32)(e14)


 
    e21 = define_encoder_block(in_image2, 64, 1, batchnorm=False)
    e21 = define_encoder_block(e21, 64, 1, batchnorm=False)
    e22 = define_encoder_block(e21, 128, 1)
    e22 = define_encoder_block(e22, 128, 2)
    e23 = define_encoder_block(e22, 256, 1)
    e23 = define_encoder_block(e23, 256, 2)
    e24 = define_encoder_block(e23, 512, 1)
    e24 = define_encoder_block(e24, 512, 2)
    e25 = define_encoder_block(e24, 512, 1) 
    e25 = define_encoder_block(e25, 512, 2)
 



    c1 = Concatenate()([e25, e16])
    d1 = decoder_block(c1, 256,1)
    d1 = decoder_block(d1, 256,2)
    c2 = Concatenate()([e24, r4, d1])
    d2 = decoder_block(c2, 128,1)
    d2 = decoder_block(d2, 128,2)
    c3 = Concatenate()([e23, r3, d2]) 
    d3 = decoder_block(c3, 64,1)
    d3 = decoder_block(d3, 64,2)
    c4 = Concatenate()([e22, r2, d3]) 
    d4 = decoder_block(c4, 32,1)
    d4 = decoder_block(d4, 32,2)
    c5 = Concatenate()([e21, r1, d4])
    d5 = Conv2DTranspose(3, (3,3), padding='same', kernel_initializer=init)(c5)
    out_image = Activation('tanh')(d5) 
    # define model
    model = Model([in_image1,in_image2], out_image)
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
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape1, image_shape2):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src1 = Input(shape=image_shape1)
    in_src2 = Input(shape=image_shape2)

    # connect the source image to the generator input
    gen_out = g_model([in_src1, in_src2])
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src2, gen_out])
    # src image as input, generated image and classification output
    model = Model([in_src1, in_src2], [dis_out, gen_out])
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
    X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5

    return [X1, X2, X3]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB, trainC = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2, X3 = trainA[ix], trainB[ix], trainC[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2, X3], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples1, samples2, patch_shape):
    # generate fake instance
    X = g_model.predict([samples1, samples2])
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB, X_realC], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realB, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_realC = (X_realB + 1) / 2.0
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
        pyplot.imshow(X_realC[i])
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
    trainA,  trainB, trainC = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB, X_realC], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model,X_realB, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realC], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch([X_realB,X_realA], [y_real, X_realC])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)

# load image data
dataset = load_real_samples('maps_64.npz')
print('Loaded', dataset[0].shape, dataset[1].shape,  dataset[2].shape )
# define input shape based on the loaded dataset
image_shape1 = dataset[1].shape[1:]
image_shape2 = dataset[0].shape[1:]

# define the models
d_model = define_discriminator(image_shape2)
g_model = define_generator(image_shape1, image_shape2)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape1, image_shape2)
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
    X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5

    return [X1, X2, X3]

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
[X1, X2, X3] = load_real_samples('maps_640.npz')
print('Loaded', X1.shape, X2.shape)
# load model
model = load_model('model_000820.h5')
# select random example
#ix = randint(0, len(X1), 1)
ix =[2] 
src_image,sen_image, tar_image = X1[ix], X2[ix], X3[ix]
# generate image from source
gen_image = model.predict([sen_image,src_image])
# plot all three images
plot_images(src_image, gen_image, tar_image)

#%%
import skimage
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps 
import numpy as np      
from skimage.metrics import structural_similarity as ssim
#from skimage.measure import compare_psnr as psnr
ssim_in=[]
psnr_in=[]
ssim_out=[]
psnr_out=[]
mse_out=[]
mse_in=[]
[X1, X2, X3] = load_real_samples('maps_640.npz')
cust = {'InstanceNormalization': InstanceNormalization}

model = load_model('model_009000.h5',cust)
ix=[]
sz=256
for iix in range(200):
    ix.append(iix) 
    src_image,sen_image, tar_image = X1[ix], X2[ix], X3[ix]
    gen_image = model.predict([sen_image,src_image])
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
    value2 = skimage.metrics.peak_signal_noise_ratio(image_array2,image_array3)
    value3 = skimage.metrics.mean_squared_error(image_array2,image_array3)

    value4 = ssim(image_array1,image_array3,multichannel=True)
    value5 = skimage.metrics.peak_signal_noise_ratio(image_array1,image_array3)
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
import cv2