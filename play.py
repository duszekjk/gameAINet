from PIL import ImageGrab, Image
import PIL
import io
import os
import sys
import numpy as np
import keyboard
import keys
keys.init()


import record


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import settings
settings.init()
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from PIL import Image
import keras
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, serialize
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import losses
from keras.layers.advanced_activations import LeakyReLU
import random
from mss import mss
import time
import threading

from keras.models import load_model

#
#opt = keras.optimizers.rmsprop(lr=0.00004, decay=1e-6)
#model.compile(loss='mean_squared_error',
#              optimizer=opt,
#              metrics=['accuracy'])

nkr = 1
lri = -211.0
lle = 178.0
images_old = [12345]
images_old_start = False
timelast = 0.0
predictionlast = np.array([1.0,0.0])

def drive():
    global model
    global images_old
    global images_old_start
    global lastPressedKeys
    global nrOfFrame
    global nkr
    global lri
    global lle
    global timelast
    global predictionlast
    path = "/Users/jacekkaluzny/Documents/carai/"
    timeA = time.time()
    waiting = True
    while waiting:
        files = os.listdir(path)
        for file in files:
            if file[-3:] == "png":
                time.sleep(0.03)
                images = load_img(path+file, target_size=(128, 205))
                os.remove(path+file)
                waiting = False
                break
        time.sleep(0.01)
            
    timeB = time.time()
    
    image = img_to_array(images)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    image /= 255.0
    
    timeD = time.time()
    print("screenshot: ", timeB-timeA)
    
    timeStart = time.time()
    prediction = model.predict(image,verbose=1)
#        prediction = (2.0*predictionlast + prediction)/3.0
    predictionlast = prediction
    timeEnd = time.time()
    timechange = time.time() - timelast
    timelast = time.time()
    predictionString = ""
    for p in prediction[0]:
        predictionString += str(p)+","
    print("------------------------------------------------------------------------------------------")
    print("\t\t", predictionString, "\t\t\t in time: ", timeEnd-timeStart ," frame time:",timechange)
    print("------------------------------------------------------------------------------------------")
#    beep()
    nrOfFrame += 1
    saveFile = open("/Users/jacekkaluzny/Documents/carai/action.txt", "w+")
    saveFile.write(predictionString)
    saveFile.close()
#    thread1 = threading.Thread(target = pressKey, args = (prediction, ))
#    thread1.start()
def pressKey(predictionl):
    global lastPressedKeys
    global nrOfFrame
    global nkr
    global lri
    global lle
    frameNow = nrOfFrame
    próg = 50.0
    i = 0.0
    while(frameNow == nrOfFrame):
        if i <= 8.0:
            prediction = [[(predictionl[0][0]*(8-i)+predictionl[0][2]*i)/8.0, (predictionl[0][1]*(8-i)+predictionl[0][3]*i)/8.0],]
        else:
            prediction = [[predictionl[0][2], predictionl[0][3]],]
        i += 1.0
#        prediction[0][1] -= 0.3
        if(random.getrandbits(1)):
            próg = (1.0*random.randint(30, 99))/100.0
        else:
            próg = (1.0*random.randint(40, 90))/100.0
#        nkr += 1
        newPressedKeys = np.array([0, 0, 0, 0, 0, 0, 0])
#        for p in range(0, len(prediction[0])):
#            prediction[0][p] += random.randint(0, 60) - 30
        if(prediction[0][0] > próg):
            print("up")
            newPressedKeys[0] = 1
        else:
            if(-1.0*prediction[0][0] > próg):
                newPressedKeys[1] = 1
                print("down")
#        lri += prediction[0][2]
        if(prediction[0][1] > próg):
#            if(prediction[0][2] - (lri/nkr) > próg):
            newPressedKeys[3] = 1
            print("left")
        else:
#        lle += prediction[0][3]
#        if(prediction[0][2] + 40.0 - (lri/nkr) < prediction[0][3] - (lle/nkr)):
            if(-1.0*prediction[0][1] > próg):
                newPressedKeys[2] = 1
                print("right")
    # -7 -43.23562299548172 -194.61520547941913
    #    if(prediction[0][4] > 10):
    #        newPressedKeys[4] = 1
    #        print("enter")
    #n
    #    if(prediction[0][5] > 10):
    #        newPressedKeys[6] = 1
    #        print("esc")1

#        if(prediction[0][6] > 63 and prediction[0][6] > prediction[0][0]):
#            newPressedKeys[6] = 1
#            print("r")
        change = newPressedKeys - lastPressedKeys
        lastPressedKeys = newPressedKeys
        for keyChange, keyName in zip(change, keys.names):
            if keyChange > 0:
                keyboard.press(keyName)
            if keyChange < 0:
                keyboard.release(keyName)
        time.sleep(0.01)



def setDrvieMode(self, state = "ai"):
    global driveMode
    if state == "ai":
        driveMode = True
    else:
        if driveMode:
            for key in keys.names:
                keyboard.release(key)
            lastPressedKeys = np.array([0, 0, 0, 0, 0, 0, 0])
        driveMode = False
    
    print(state)

nrOfFrame = 0
lastPressedKeys = np.array([0, 0, 0, 0, 0, 0, 0])
print("press n to start")

driveMode = True
keyboard.add_hotkey('1', setDrvieMode, args=("me"))
#keyboard.add_hotkey('up', setDrvieMode, args=("me"))
#keyboard.add_hotkey('down', setDrvieMode, args=("me"))
#keyboard.add_hotkey('left', setDrvieMode, args=("me"))
#keyboard.add_hotkey('right', setDrvieMode, args=("me"))
keyboard.add_hotkey('2', setDrvieMode, args=("ai"))


try:
#    model = load_model(settings.save_dir+"/"+settings.model_name)
    model = load_model(settings.save_dir+"/"+"load.hdf5")
except:
    print("no model")
    driveMode = False
else:
    model.summary()

from AppKit import NSSound
#prepare sound:
sound = NSSound.alloc()
sound.initWithContentsOfFile_byReference_('/System/Library/Sounds/Ping.aiff', True)
#rewind and play whenever you need it:

def beep():
    global sound
    sound.stop() #rewind
    sound.play()
beep()
keyboard.wait('n')

beep()
threads = []
#os.system("./gameAINet.xcodeproj/CARLA_0.8.2/CarlaUE4.sh")
while(1):
    driveMode = True
    if driveMode:
        print("d")
#        drive(rr)
        thread = threading.Thread(target=drive)
        threads.append(thread)
        while(len(threads) > 1):
            threads[0].join()
            threads.pop(0)
        time.sleep(0.01)
        print("threads: ", len(threads))
        thread.start()
    else:
        print("r")
        thread = threading.Thread(target=record.start)
        #        threads.append(thread)n1
        time.sleep(0.03)
        thread.start()



# predicting multiple images at once
#img = image.load_img('test2.jpg', target_size=(img_width, img_height))
#y = image.img_to_array(img)
#y = np.expand_dims(y, axis=0)
#2
## pass the list of multiple images np.vstack()
#images = np.vstack([x, y])
#classes = model.predict_classes(images, batch_size=10)
#
## print the classes, the images belong to
#print classes
#print classes[0]
#print classes[0][0]
