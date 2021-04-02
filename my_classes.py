import numpy as np
import keras
from os import listdir
from os.path import isfile, join
import os
import time

import settings
import random


from scipy import ndimage, misc
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
import keras.applications as kapp
from keras.models import load_model
from os.path import isfile, join
import json

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=settings.batch_size, dim=(1, 2,128,205), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        #        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        #        return int(min(int(np.floor(len(self.list_IDs) / self.batch_size)),(((3+settings.saveNow)//3)**3)*40))
        return min(int(np.floor(len(self.list_IDs[0]) / self.batch_size)), settings.maxEpochSize)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #        print(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        keyList = list(self.list_IDs[0].keys())
#        keyListb = list(self.list_IDs[1].keys())
        list_IDs_tempa = dict()
        i = 0
        for k in indexes:
            list_IDs_tempa[keyList[k]] = self.list_IDs[0][keyList[k]]
        list_IDs_tempb = dict()
        i = 0
#        for k in indexes:
#            list_IDs_tempb[keyList[k]] = self.list_IDs[1][keyList[k]]
        X, y = self.__data_generation(list_IDs_tempa)
#        time.sleep(0.1)
        return X, y
    def on_epoch_end(self, epoch=int(settings.saveNow), logs=None):
        'Updates indexes after each epoch'
        os.system("caffeinate -u -t 36000 &")
        #        settings.saveNow = epoch
        settings.saveNow = 0.5 + settings.saveNow
        print("epoch end: ", settings.saveNow, epoch)
        self.indexes = np.arange(len(self.list_IDs[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
#            while(abs(np.average(np.array(self.labels[self.list_IDs[0][self.indexes[:settings.maxEpochSize*settings.batch_size]]]))) > 0.3):
#                np.random.shuffle(self.indexes)
#                print(".", abs(np.average(np.array(self.labels[self.indexes[:settings.maxEpochSize*settings.batch_size]]))), end=" ")


    def __data_generation(self, list_IDs_temp):
#        print(list_IDs_temp)
#        randkod = random.randint(1000, 9999)
        imagesLoadeda = self.loadIMGS(list_IDs_temp)
        x_trainal = []
        y_trainl = []
        for aa in list(imagesLoadeda.keys()):
            try:
                b = imagesLoadeda[aa]
                x_trainal.append(imagesLoadeda[aa])
                y_trainl.append(aa)
            except:
                error = 1
#                print(aa)
            
        (x_traina, y_train) = np.array(x_trainal).reshape(-1,128,205,3), np.array([self.labels[x] for x in y_trainl])
        
        x_traina = x_traina.astype('float32')
        x_traina /= 255.0
#        x_trainb = x_trainb.astype('float32')
#        x_trainb /= 255.0
#        print(x_traina.shape, x_trainb.shape)
        
#        return np.stack((x_traina, x_trainb)).reshape(-1,2,128,205,3), y_train
        return x_traina, y_train
    
    
    def loadIMGS(self, paths):
        imagesL = dict()
        ii = 0
        for name in paths:
#            ii += 1
            filename = paths[name]
            #            print(filename, self.labels[name])
            if os.path.isfile(filename):
                try:
                    imagea = load_img(filename, target_size=(128, 205))
                except:
                    # doesn't exist
                    print(filename, "doesn't exist")
                    os.system("rm -f \""+filename+"\"")
                else:
                    # exists
                    try:
                        image = img_to_array(imagea)
                    except:
                        print(filename, " not loaded")
                        os.system("rm -f \""+filename+"\"")
                    else:
                        
#                        if(random.getrandbits(1)):
#                            image = np.flip(image, 1)
#                        if(random.getrandbits(1)):
#                            if(random.getrandbits(1)):
#                                if(random.getrandbits(1)):
#                                    if(random.getrandbits(1)):
#                                        image = gaussian_filter(image, sigma=(3))
                        fil = "f"
                        if "val" not in filename:
                            if(random.getrandbits(1) and random.getrandbits(1)):
                                if(random.getrandbits(1)):
                                    change = (1.0*random.randint(-5000,5000))/5000.0
                                    image = np.clip((image * (1+(change/3.0))), 0, 255)
                                    fil += " c "
                                if(random.getrandbits(1)):
                                    change = (1.0*random.randint(-1000,1000))/1000.0
                                    image = np.clip((image + change*35.0), 0, 255)
                                    fil += " b "
                                if(random.getrandbits(1)):
                                    change = (1.0*random.randint(-1000,1000))/1000.0
                                    image = np.clip(ndimage.rotate(image, change*10.0, reshape=False), 0, 255)
                                    fil += " r "
                                if(random.getrandbits(1)):
                                    changel = (1.0*random.randint(-1000,1000))/1000.0
                                    change = (np.random.rand(image.shape[0], image.shape[1], image.shape[2])*2.0-1.0)*changel
                                    image = np.clip((image + change*35.0), 0, 255)
                                    fil += " n "
                        
                        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                        # prepare the image for the VGG model

#                        imgprev = Image.fromarray(image[0, :, :, :].astype(np.uint8), 'RGB')
#                        imgprev.save('/Users/jacekkaluzny/Pictures/aug/a'+fil+'a'+str(random.randint(10000, 99999))+'.png')
                        image = preprocess_input(image)
                        imagesL[name] = image
    #                    print(filename, "is damaged exist")
        return imagesL
    
    def createBatch(data, SIZE=1000):
        endList = list()
        smallDict = dict()
        i = 0
        for key in data:
            #        print("key:", key)
            if i%SIZE != SIZE-1:
                smallDict[key] = data[key]
            else:
                endList.append(smallDict)
                smallDict = dict()
            i+=1
        return endList


