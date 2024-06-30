# for Ynet 

from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(1024,2048)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :1024], pixels[:, 1024:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]


# load all images in a directory into memory
def load_images2(path, size=(1024,1024)):
	graph_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		graph_img = pixels
		graph_list.append(graph_img)
		
	return asarray(graph_list)
# dataset path
path1 = './test1024/'
path2 = './Ytest1024/'
# load dataset
[src_images, tar_images] = load_images(path1)
graph_images = load_images2(path2)
print('Loaded: ', src_images.shape, tar_images.shape, graph_images.shape)
# save as compressed numpy array
filename = 'Ymaps_10240.npz'
savez_compressed(filename, src_images, tar_images, graph_images)
print('Saved dataset: ', filename)

