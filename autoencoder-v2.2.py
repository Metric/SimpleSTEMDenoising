
from gc import callbacks
from genericpath import exists
from pickle import NONE
from random import seed
import tensorflow as tf
from PIL import Image, ImageDraw
import os
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, InputLayer, AveragePooling2D, UpSampling2D
import numpy as np

mode = 0 #1 = train, 0 = infer
model_file = "" #do not include the .index extension; as this is added automatically.
infer_dir = ""
train_dir = ""
results_dir = "results"
tmp_dir = "tmp" #used to temp storage of noised images
minimum_loss = 0.0122 # adjust as necessary to prevent overtraining, or set to 0 to disable this check
image_size = 512 # this is a helper; changing will build the proper neural net layers to fit 
                 # maintain a number that is a multiple of 2: 128, 256, 512, 1024, 2048, etc.
                 # do not go below 128
noise_type = 0 #0 = uniform, 1 = normal, 2 = poisson (lambda 0.45)
include_average_pooling = 0 #a value greater than 0 will include an average pooling layer after the last 8x8 layer
                            #a proper UpSampling2D will be included in the decoder as well
average_pooling_layer_size = 2 #the size of the average pooling downsampling

#note the images loaded to infer must be grayscale mode

#note the images loaded for training must be in RGB mode
#this allows a quick an efficient method of creating a cropped part with scan noise
#that is copied with np.resize() and returned via tf.random.uniform() etc.

def list_dir(folder):
    files = []
    paths = []
    for file in os.listdir(folder):
        files.append(file)
        paths.append(os.path.join(folder, file))
    
    return files, paths

def load_img(path):
    image = Image.open(path)
    image = image.resize((image_size,image_size))
    image = np.array(image)
    image = np.reshape(image, (image_size, image_size, 1))
    image = image.astype(np.float32)
    image /= 255.0
    return image

def create_dataset(folder, ntype=0):
    i = 0

    images = []
    noises = []

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        image = Image.open(img_path)
        image = image.resize((image_size, image_size))
        image = np.array(image)

        # we use resize instead of reshape here
        # this is so a cropped part is copied 
        # in the newer part of the multi array
        # this is more efficient than trying to do this manually
        # as it also helps create horizontal artifacts
        # since the horizontal pixel offset is slightly off
        # this helps with scanline removal / denoising
        image = np.resize(image, (image_size, image_size, 1))
        image = image.astype(np.float32)
        image /= 255.0

        noised = image

        # now we noise and due to how the noising works it returns
        # primarily the cropped part that was copied in the 
        # above array section

        if ntype == 0:
            noised = image + 0.5 * tf.random.uniform(shape=image.shape)
        elif ntype == 1:
            noised = image + 0.5 * tf.random.normal(shape=image.shape)
        elif ntype == 2:
            noised = image + 0.5 * tf.random.poisson(shape=image.shape, lam=0.45)

        noised = tf.clip_by_value(
            noised, clip_value_min=0., clip_value_max=1.)

        noise_sample = tf.keras.preprocessing.image.array_to_img(noised)
        noise_sample.save(os.path.join(tmp_dir, "nsample" + str(i) + ".bmp"))

        i = i + 1

        images.append(image)
        noises.append(noised)

    return images, noises

if mode == 1:
    x_train, x_test = create_dataset(train_dir, noise_type)


class NoiseReducer(tf.keras.Model):
    def __init__(self):

        super(NoiseReducer, self).__init__()

        encoding_list = []
        
        denom = 2
        rsize = image_size / denom
        nsize = rsize

        encoding_list.append(InputLayer(input_shape=(image_size, image_size, 1)))

        while nsize >= 8:
            if nsize >= 64:
                encoding_list.append(Conv2D(nsize, 1, activation='relu', padding='same'))
            elif nsize >= 16:
                encoding_list.append(Conv2D(nsize, 3, activation='relu', padding='same'))
            elif nsize >= 8:
                encoding_list.append(Conv2D(nsize, 3, activation='swish', padding='same')) 
                if include_average_pooling > 0:
                    encoding_list.append(AveragePooling2D((average_pooling_layer_size,average_pooling_layer_size)))
                    
                break
            nsize = nsize / denom

        self.encoder = tf.keras.Sequential(encoding_list)

        decoding_list = []
        nsize = 8

        while nsize <= rsize:
            if nsize <= 8:
                if include_average_pooling > 0:
                    decoding_list.append(UpSampling2D((average_pooling_layer_size, average_pooling_layer_size), interpolation='bilinear'))
                    
                decoding_list.append(Conv2DTranspose(nsize, 3, activation='swish', padding='same'))
            elif nsize <= 32:
                decoding_list.append(Conv2DTranspose(nsize, 3, activation='relu', padding='same'))
            else:
                decoding_list.append(Conv2DTranspose(nsize, 1, activation='relu', padding='same'))
            nsize = nsize * denom

        decoding_list.append(Conv2DTranspose(1, 1, activation='swish', padding='same'))

        self.decoder = tf.keras.Sequential(decoding_list)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class haltCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        if(float(logs.get('loss')) <= minimum_loss):
            print("\n\n\nReached minimum loss value so cancelling training!\n\n\n")
            self.model.stop_training = True

autoencoder = NoiseReducer()
autoencoder.compile(optimizer='adam', loss='logcosh')

if exists(model_file + ".index"):
    autoencoder.load_weights(model_file)
else:
    print("model file not found")

callbacks = []

if mode == 1:
    if minimum_loss > 0:
        callbacks.append(haltCallback())

    batch_size = 4
    if image_size > 1024:
        batch_size = 2
    elif image_size == 1024:
        batch_size = 3

    steps = 12 / batch_size

    autoencoder.fit(tf.convert_to_tensor(x_test[:12]), 
                    tf.convert_to_tensor(x_train[:12]),
                    batch_size=batch_size,
                    epochs=1,
                    steps_per_epoch=steps,
                    shuffle=True,
                    callbacks=callbacks)

    autoencoder.save_weights(model_file)

names, paths = list_dir(infer_dir)

i = 0
for p in paths:
    img = load_img(p)
    imgs = []
    imgs.append(img)
    encoded_imgs = autoencoder.encoder(tf.convert_to_tensor(imgs))
    decoded_imgs = autoencoder.decoder(encoded_imgs)
    result = decoded_imgs[0]
    result_image = tf.keras.preprocessing.image.array_to_img(result)
    result_image.save(results_dir + "/" + names[i])
    i = i + 1
