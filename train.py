#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import settings
settings.init()


import keras
import keras.applications as kapp
from keras.utils import multi_gpu_model

import six
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, serialize
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from keras import regularizers
from keras import losses
from keras.layers.advanced_activations import LeakyReLU

from os import listdir
from os.path import isfile, join
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

from my_classes import DataGenerator

import random
import threading
from itertools import islice
import matplotlib.pyplot as plt
from operator import add

from PIL import Image
#import keyboard  # using module keyboard
from getkey import getkey, keys
from AppKit import NSSound
#prepare sound:
sound = NSSound.alloc()
sound.initWithContentsOfFile_byReference_('/System/Library/Sounds/Ping.aiff', True)
def beep():
    global sound
    sound.stop() #rewind
    sound.play()

labelsb = dict()
labels = dict()

imagesCombiner = dict()
imagesBlocker = 0

imagesCombinerLoad = dict()
imagesBlockerLoad = 0

def showPlots():
    global historyAvg
    print("plots:")
    plt.plot(list( map(add, historyAvg['mean_squared_error'][3:], historyAvg['val_mean_squared_error'][3:])))
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train + test'], loc='upper left')
    plt.show()
    plt.plot(list( map(add, historyAvg['mean_absolute_error'][3:], historyAvg['val_mean_absolute_error'][3:])))
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train + test'], loc='upper left')
    plt.show()
    
    plt.plot(historyAvg['mean_squared_error'][3:])
    plt.plot(historyAvg['val_mean_squared_error'][3:])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(historyAvg['mean_absolute_error'][3:])
    plt.plot(historyAvg['val_mean_absolute_error'][3:])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history['mean_squared_error'][3:])
    plt.plot(history['val_mean_squared_error'][3:])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('batches')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history['mean_absolute_error'][3:])
    plt.plot(history['val_mean_absolute_error'][3:])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('batches')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def createBatch(data, SIZE=1000):
    endList = list()
    smallDict = dict()
    i = 0
    for key in data:
        if i%SIZE != SIZE-1:
            smallDict[key] = data[key]
        else:
            endList.append(smallDict)
            smallDict = dict()
        i+=1
    return endList

def loadIMGS(paths):
    imagesL = dict()
    for name in paths:
        filename = paths[name]
        if os.path.isfile(filename):
            try:
                imagea = load_img(filename, target_size=(256, 256))
            except:
                # doesn't exist
                print(filename, "doesn't exist")
            else:
                # exists
                try:
                    image = img_to_array(imagea)
                except:
                    print(filename, " not loaded")
                else:
                    # reshape data for the model
                    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    # prepare the image for the VGG model
                    image = preprocess_input(image)
                    imagesL[name] = image
    return imagesL

j = 0
def loadPhotosNamesInCategory(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    global labels
    global labelsb
    global j
    images = dict()
    all_photos = []
    try:
        all_photos_file = open(directory + "/" + className + "/" + className + ".txt","r+")
        for line in all_photos_file:
            all_photos.append(line)
    except:
        all_photos = listdir(directory + "/" + className)
    all_photos_size = len(all_photos)
    print("\t", all_photos_size)
    random.shuffle(all_photos)
    for name in all_photos[:min(30000, all_photos_size)]:
        if name[0] != '.':
            # load an image from file
            filename = directory + '/' + className + '/' + name
            classNameInt = className.split("k")[1:]
            label = [float(classNameInt[0])-float(classNameInt[1]), float(classNameInt[3])-float(classNameInt[2])]
            labelsb[i*10000000+j] =  label
            labels[i*10000000+j] = label
            images[i*10000000+j] = filename
            j += 1
            
    imagesCombiner.update(images)
    imagesBlocker += 1



def load_photos(directory, names = False):
    global imagesBlocker
    global imagesCombiner
    
    #imagesCombiner.clear()
    threads = []
    images = dict()
    i = 0
    print(listdir(directory))
    for className in listdir(directory):
        if className[0] != '.':
            #            print("class: ", className, i, names)
            if ".png" in className and names == False:
                imagesBlocker = 0
                threads.append(threading.Thread(target=loadThisPhoto, args=(directory, className, i)))
            else:
                if names == True:
                    print(className, "\t\tloading", end="\t")
                    loadPhotosNamesInCategory(directory, className, i)
                    print(className, "\t\tloaded")
                else:
                    threads.append(threading.Thread(target=loadPhotosInCategory, args=(directory, className, i)))
            i += 1
    if names != True:
        print("class state: \tname\tnr of images (", i, ")")
        for thread in threads:
            thread.start()
            
        while imagesBlocker != i:
            time.sleep(2.0)
            print(imagesBlocker, i)
        for thread in threads:
            i -= 1
            thread.join()
    images = imagesCombiner
    print("All loaded")
    keys =  list(imagesCombiner.keys())
    random.shuffle(keys)
    for key in keys:
        images[key] = imagesCombiner[key]
    return images


analisedData = None


def breakTraining():
    while settings.stopTraining != True:  # making a loop
        time.sleep(3)
        key = getkey()
        try:  # used try so that if user pressed other than the given key error will not be shown
            if key == 'q' or key == 'p':  # if key 'q' is pressed
                if key == 'q':
                    settings.stopTraining = True
                    print('\tStopping!\t')
                if key == 'p':
                    settings.shouldShowPlots = True
                    settings.showPlots()
                    print('\tPlots comming!\t')
            else:
                if key != None:
                    print('press q to stop or p to show plots')
                pass
        except:
            print('press q to stop or p to show plots')


keyboardStop = threading.Thread(target=breakTraining)
keyboardStop.start()
print(settings.directory)

#(imagesTrain, imagesTest) = chunks(images, int(len(images)*995/1000))
imagesTrain = load_photos(settings.directory, names = True).copy()
imagesCombiner.clear()
imagesTest = load_photos(settings.directoryval, names = True).copy()
imagesCombiner.clear()

#imagesChunks =  createBatch(imagesTrain, 4*(int(settings.batch_size*1.25))+1)


print('Loaded Images: %d / %d' % (int(len(imagesTrain)), int(len(imagesTest))))


# Generators
training_generator = DataGenerator(imagesTrain, labels)
validation_generator = DataGenerator(imagesTest, labels)
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
model_path = os.path.join(settings.save_dir, settings.model_name)
#keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


epoch_start = 0
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
model_path = os.path.join(settings.save_dir, settings.model_name)

settings.model = Sequential()

settings.model.add(Conv2D(32, (3, 3), padding='same',
                          input_shape=(256, 256, 3)))
settings.model.add(LeakyReLU(alpha=0.01))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))

settings.model.add(Conv2D(32, (3, 3), padding='same'))
settings.model.add(LeakyReLU(alpha=0.01))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))

settings.model.add(Conv2D(128, (3, 3), padding='same'))
settings.model.add(LeakyReLU(alpha=0.01))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))

#settings.model.add(Conv2D(128, (3, 3), padding='same'))
#settings.model.add(LeakyReLU(alpha=0.01))
#settings.model.add(MaxPooling2D(pool_size=(4, 4)))

settings.model.add(Flatten())
settings.model.add(Dense(320))
settings.model.add(LeakyReLU(alpha=0.1))
settings.model.add(Dropout(0.4))

settings.model.add(Dense(320))
settings.model.add(LeakyReLU(alpha=0.1))
settings.model.add(Dropout(0.4))

settings.model.add(Dense(14))
settings.model.add(LeakyReLU(alpha=0.1))
settings.model.add(Dropout(0.2))
settings.model.add(Dense(2))
settings.model.add(Activation('linear'))
settings.model.summary()

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = 'adam'#keras.optimizers.rmsprop(lr=0.0000001, decay=1e-6)


#settings.model.load_weights(settings.save_dir+"/load.hdf5")         #, by_name=True)

#settings.model = multi_gpu_model(settings.model, gpus=2)


settings.model.compile(loss='mean_squared_error',
                         optimizer=opt,
                         metrics=['categorical_accuracy', 'mean_squared_error', 'mean_absolute_error'])
settings.model.save(model_path)
filepath=settings.save_dir+"/weights-improvementn-{epoch:02d}-{mean_absolute_error:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='mean_absolute_error', verbose=1, save_best_only=True, mode='min')
webpage = RemoteMonitor(root='http://gameaiupdate.duszekjk.com', path="/")
callbacks_list = [checkpoint, webpage]
historyAvg = []
if(os.path.isdir(settings.directory)):
    History = settings.model.fit_generator(generator=training_generator, steps_per_epoch=1500,
                                           validation_data=validation_generator,
                                           use_multiprocessing=False,
                                           workers=1, epochs=settings.epochs, verbose = 1, callbacks=callbacks_list, initial_epoch = epoch_start)
else:
    print("No directory Error")
    beep()
beep()
settings.model.save(model_path)



stopTraining = True


print("additional tests:")
listOfTests = load_photos(settings.directorytest, names = True)
myTest = loadIMGS(listOfTests)
print('Loaded Images Test: %d' % int(len(myTest)))
(my_x_test, my_y_test, my_z_test) = np.array(list(myTest.values())).reshape(-1,256,256,3), np.array([labels[x] for x in list(myTest.keys())]), np.array([labelsb[x] for x in list(myTest.keys())])
my_x_test = my_x_test.astype('float32')
my_x_test /= 255.0


classes = settings.model.predict(my_x_test, batch_size=16)
j = 0
przysp = 0
skręcanie = 0

if len(classes[0]) == 2:
    przysp += classesProbs[0] - my_y_test[0]
    skręcanie += classesProbs[1] - my_y_test[1]
    for classesProbs in classes:
        print(my_y_test[j], classesProbs)
        j += 1
    
    

else:
    
    arrayX = "["
    arrayY = "["
    arrayZ = "["
    files = ""
    for classesProbs in classes:
        
        trueA = (int(((1+my_y_test[j][0])/2.0)*100.0)-30.0)/10.0
        trueB = round(((1+my_y_test[j][1])/2.0), 2)
        predA = (int(((1+classesProbs[0])/2.0)*100.0)-30.0)/10.0
        predB = round(max(((1+classesProbs[1])/2.0), 0.1), 2)
        trueC = round(((1+my_z_test[j])/2.0)*1000)
        predC = trueC
        if len(classesProbs) > 2:
            trueC = round(((1+my_y_test[j][2])/2.0)*1000)
            predC = round(int(((1+classesProbs[2])/2.0)*1000.0), 2)
        
        print("\tfile: ", str(int(trueC))+str(int(100+trueA*10) + 1000 * int(trueB * 100)), "\ttrue:\t", trueA, trueB, trueC, "\tprediction:\t", predA, predB, predC, "\t = ", (abs(trueA - predA)*1000.0//10)/100.0, (abs(trueB - predB)*1000.0//10)/100.0, abs(trueC-predC), "raw pred:", classesProbs)
        arrayX += str(predA)+", "
        arrayY += str(predB)+", "
        arrayZ += str(predC)+", "
        files += "f\t" + str(int(trueC))+str(int(100+trueA*10) + 1000 * int(trueB * 100)) +"\t"+ str(int(predC))+str(int(100+predA*10) + 1000 * int(predB * 100)) + "\n"
        j += 1
    
    print(arrayX)
    print(arrayY)
    print(arrayZ)
    print(files)

