import os
import math
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import h5py


from keras.layers import Input,BatchNormalization,Conv2DTranspose
from keras.models import Model, Sequential, load_model
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation 
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
from keras.callbacks import TensorBoard
from keras.utils import to_categorical


import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

K.set_image_dim_ordering('th')

# Deterministic output.
# np.random.seed(1000)


class DCGAN(object):

    def __init__(self,*L):

        self.runNo = L[2]
        self.GLR = L[0]
        self.DLR = L[1]
        self.latentSize = 100
        self.decay = 6e-8    


        self.NAME = "LOGS/MNIST-G:{}-D:{}-R:{}".format(self.GLR,self.DLR,self.runNo)#,int(time.time()))

        (X_train, Y_train), (_, _) = mnist.load_data()

        # Reshape : (28,28,1) and normalize to [0,1] range from [0,255]
        imageSize   = X_train.shape[1]
        X_train     = np.reshape(X_train, [-1, imageSize, imageSize, 1])
        X_train     = X_train.astype('float32') / 255
        numLabels   = np.amax(Y_train) + 1
        Y_train     = to_categorical(Y_train)


        inputsGen  = Input(shape=(self.latentSize, ) ,name='gen_input')
        inputsDisc = Input(shape=(imageSize, imageSize, 1), name='disc_input')
        labels     = Input(shape=(numLabels, ), name='class_labels')

        self.X_train = X_train
        self.Y_train = Y_train
        self.numLabels = numLabels
        self.batchSize = 64

        # define the optimiser as an Adam optimiser, beta values control the decay rate
        # potentoally change this to RMSProp
        # optimizerG = Adam(self.GLR, 0.5)
        # optimizerD = Adam(self.DLR, 0.5)
        

        self.discriminator = self.build_discriminator(inputsDisc, labels, imageSize)
        optimizerD = RMSprop(lr=self.DLR, decay=self.decay)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizerD,metrics=['accuracy'])
        self.discriminator.summary()

        self.generator = self.build_generator(inputsGen, labels, imageSize)
        # self.generator.compile(loss='binary_crossentropy', optimizer=optimizerG)
        self.generator.summary()

        optimizerG = RMSprop(lr=self.GLR, decay=self.decay)
        self.discriminator.trainable = False
        
        outputs = self.discriminator([self.generator([inputsGen, labels]), labels])

        self.stack = self.build_stack([inputsGen,labels],outputs)
        self.stack.compile(loss='binary_crossentropy', optimizer=optimizerG, metrics=['accuracy'])

        

    def build_generator(self,inputs, labels, imageSize):
        imageResize = imageSize // 4

        x = concatenate([inputs,labels],axis=1)

        x = Dense(7*7*128)(x)
        x = Reshape((7, 7,128))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(128, kernel_size=[5,5], strides=[2,2], padding='same',input_shape=(7,7,128),data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(64, kernel_size=[5,5], strides=[2,2], padding='same',input_shape=(14,14,128),data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(32, kernel_size=[5,5], strides=[1,1], padding='same',input_shape=(28,28,2),data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, kernel_size=[5,5], strides=[1,1], padding='same', input_shape=(28,28,2),activation='sigmoid',data_format='channels_last')(x)

        generator = Model([inputs, labels], x, name='generator')

        return generator

    def build_discriminator(self,inputs, labels, imageSize):

        y = Dense(imageSize * imageSize)(labels)
        y = Reshape((imageSize, imageSize, 1))(y)
        x = concatenate([inputs,y])

        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=32, kernel_size=[5,5], strides=[2,2], padding='same', input_shape=(28,28,2),data_format='channels_last')(x)
        
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=64, kernel_size=[5,5], strides=[2,2], padding='same', input_shape=(14,14,32),data_format='channels_last')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=128, kernel_size=[5,5], strides=[2,2], padding='same', input_shape=(7,7,64), data_format='channels_last')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=256, kernel_size=[5,5], strides=[1,1], padding='same',input_shape=(4,4,128), data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        discriminator = Model([inputs, labels], x, name='discriminator')

        return discriminator

    def build_stack(self,ganInput,ganOutput):

        model = Model(inputs=ganInput, outputs=ganOutput, name="dcgan")

        return model

    def train(self,totalEpochs, batchSize,save_interval,callback):

        # noise vector of 16 by latent size 16 as batchsize = 64 / 4 image reduction
        noiseInput = np.random.uniform(-1.0, 1.0, size=[16, self.latentSize])
        # one-hot label the noise will be conditioned to
        noiseClass = np.eye(self.numLabels)[np.arange(0, 16) % self.numLabels]
        # number of elements in train dataset
        trainSize = self.X_train.shape[0]
        print(self.NAME," Hot Labels for Generated Images: ",np.argmax(noiseClass, axis=1))

        
        for epoch in tqdm(range(totalEpochs)):

            # randomly pick a real sample of size batchSize from the dataset 
            # with corresponding labels and images
            randIndexes= np.random.randint(0, trainSize, size=self.batchSize)
            realImages  = self.X_train[randIndexes]
            realLabels  = self.Y_train[randIndexes]

            # random noise sampled from the dataset, with card of the batch size
            noise = np.random.uniform(-1.0, 1.0, size=[self.batchSize, self.latentSize])
            # hot label to indicate digit
            fakeLabels = np.eye(self.numLabels)[np.random.choice(self.numLabels,self.batchSize)]
            # make the generator make a prediction based on the noise and labels above
            fakeImages = self.generator.predict([noise, fakeLabels])

            # concantenate the synthetic and real images/labels to train on discriminator
            allImages = np.concatenate((realImages, fakeImages))
            allLabels = np.concatenate((realLabels, fakeLabels))

            # Smooth Labelling 
            yDis = np.ones([2 * self.batchSize, 1])
            yDis[batchSize:, :] = 0.0
            dloss= self.discriminator.train_on_batch([allImages, allLabels], yDis)

            # now train the stacked network, since discriminator weights frozen this 
            # only affects the generator, sample new noise and switch labels to fool
            # the discriminator in adversarial game
            # conitional aspect provides hot label (0,0,0,1,0,0,0,0) to indicate digit
            noise = np.random.uniform(-1.0, 1.0, size=[self.batchSize, self.latentSize])
            fakeLabels = np.eye(self.numLabels)[np.random.choice(self.numLabels,self.batchSize)]
            yGen = np.ones([batchSize, 1])
           
            gloss = self.stack.train_on_batch([noise, fakeLabels], yGen)

            print ('epoch: %d, [Discriminator :: loss,acc: %f ; %f ], [ Generator :: loss,acc: %f ; %f ]' % (epoch, dloss[0],dloss[1], gloss[0],gloss[1]))
            if epoch % save_interval:
                plot_images(self.generator,epoch,noiseClass,noiseInput,0,self.GLR,self.DLR,self.runNo)
                write_log(callback, ['loss/g_loss'], [gloss[0]], epoch)
                write_log(callback, ['loss/d_loss'], [dloss[0]], epoch)
                write_log(callback, ['acc/d_accuracy'], [dloss[1]], epoch)

        self.generator.save(self.NAME + ".h5")

            
def plot_images(generator,epoch,noiseClass,noiseInput,isTest,GLR,DLR,runNo):

    if isTest:
        NAME = "TEST/TEST-MNIST-G:{}-D:{}-R:{}".format(GLR,DLR,runNo)#,int(time.time()))
    else: 
        NAME = "LOGS/MNIST-G:{}-D:{}-R:{}".format(GLR,DLR,runNo)#,int(time.time()))

    if not os.path.exists(NAME):
            os.makedirs(NAME)
    filename = os.path.join(NAME, "%05d.png" % epoch)
    images = generator.predict([noiseInput, noiseClass])
    print(NAME , " labels for generated images: ", np.argmax(noiseClass, axis=1))
    plt.figure(figsize=(2.2, 2.2))
    numImages = images.shape[0]
    imageSize = images.shape[1]
    rows = int(math.sqrt(noiseInput.shape[0]))
    for i in range(numImages):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [imageSize, imageSize])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    plt.close()

def test_generator(generator,classLabel,GLR,DLR,runNo):
    noiseInput = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    print(classLabel)
    if classLabel is None :
        numLabels = 10
        noiseClass = np.eye(numLabels)[np.random.choice(numLabels, 16)]
    else:
        noiseClass = np.zeros((16, 10))
        noiseClass[:,classLabel] = 1
        step = classLabel

    print("Plotting the image")

    plot_images(generator,step,noiseClass,noiseInput,1,GLR,DLR,runNo)

if __name__ == '__main__':
    print("Enter Learning Rate for Generator and Discriminator respectively")
    GLR = float(input())
    DLR = float(input())
    runNo = 1
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        filename = "MNIST-G.{}-D.{}-R.{}".format(GLR,DLR,runNo)
        generator = load_model(filename)
        classLabel = None
        if args.digit is not None:
            classLabel = args.digit
        test_generator(generator, classLabel,GLR,DLR,runNo)
    else:
        L = [GLR,DLR,runNo]
        gan = DCGAN(*L)

        callback = TensorBoard(gan.NAME)
        callback.set_model(gan.stack)

        gan.train(1, gan.batchSize, 100, callback)
    
