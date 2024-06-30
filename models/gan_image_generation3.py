

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
path1 = 'D:\\hesam\\vessel\\recon'

input_img_paths = sorted(
    [
        os.path.join(path1, fname)
        for fname in os.listdir(path1)
        
    ]
)


random.Random(1337).shuffle(input_img_paths)
train_input_img_paths = input_img_paths[0:2000]

# load dataset
src_images = load_srcimages(train_input_img_paths)



# save as compressed numpy array
filename1 = 'maps_512.npz'
savez_compressed(filename1, src_images)
print('Saved dataset: ', filename1)
#%%
# load the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the dataset
data = load('maps_512.npz')
src_images = data['arr_0']
print('Loaded: ', src_images.shape)
# plot source images
n_samples =3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i].astype('uint8'))
# plot target image

pyplot.show()
#%%

# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot
from numpy import load

# define the standalone discriminator model
def define_discriminator(in_shape=(256,256,3)):
	model = Sequential()
	model.add(Conv2D(512, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(256, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 256 * 16 *16
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((16, 16, 256)))

	#model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	#model.add(LeakyReLU(alpha=0.2))
	
	#model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
	#model.add(LeakyReLU(alpha=0.2))
 
	model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
 
	model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
 
	model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
 
	model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
 
	model.add(Conv2D(3, (7,7), activation='sigmoid', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
'''
# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X
'''  
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X = data['arr_0']
	X = X.astype('float32')

	# scale from [0,255] to [-1,1]
	X = X / 255

	return X
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
#def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	#x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	#x_input = x_input.reshape(n_samples, latent_dim)
	#return x_input

    
    
import numpy
import numpy as np
from numpy import load
npz = load('SC4001E0.npz')
signal = npz['x']
signal=numpy.reshape(signal,(2523000,1))
b1=signal-np.min(signal)
u=np.max(signal)-np.min(signal)
b=b1/u
b=b-.5
signal=b/.5
 
def generate_latent_points(latent_dim, n_samples):
    z=np.zeros((n_samples, latent_dim))

    for i in range(n_samples):
        c=random.randint(0, 2523000-latent_dim)
        b2=signal[c:c+latent_dim]

        b2=numpy.reshape(b2,(latent_dim,))
        z[i,:]=b2
    return z



 

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=10):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	#save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=4):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights•
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 5 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

# size of the latent space
latent_dim = 250
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples('maps_512.npz')
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
#%%	
from numpy import asarray
from matplotlib import pyplot

from tensorflow.keras.models import load_model
latent_dim = 250

# load model
model = load_model('generator_model_080.h5')
# all 0s
#vector = asarray([[0.3 for _ in range(250)]])
vector= generate_latent_points(latent_dim, 1)
# # generate image
X = model.predict(vector)
# plot the result
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()

#%%
from tensorflow.keras.models import load_model
import skimage
from PIL import Image, ImageOps 
import numpy as np      
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
ssim_in=[]

def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1 = data['arr_0']
    # scale from [0,255] to [-1,1]
    X1 = X1/255

    return X1
X1 = load_real_samples('maps_512.npz')
model = load_model('generator_model_080.h5')
ix=[]
sz=256
vector= generate_latent_points(latent_dim, 1)
for iix in range(2000):
    ix.append(iix) 
    src_image = X1[ix]
    vector= generate_latent_points(latent_dim, 1)
    gen_image = model.predict(vector)
    image_array1 = np.asarray(src_image)
    image_array2 = np.asarray(gen_image)
    image_array1= np.reshape(image_array1,(sz,sz,3))
    image_array2= np.reshape(image_array2,(sz,sz,3))
    value1 = ssim(image_array2,image_array1,multichannel=True)


    ssim_in.append(value1)

    ix=[]

print(np.mean(ssim_in))
#%%
from matplotlib import pyplot
zz=generate_latent_points(500,1)

pyplot.plot(zz[0,:])
pyplot.show()



