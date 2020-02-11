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
from keras.preprocessing import image
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
def drive():
    global model
    global lastPressedKeys
    global nrOfFrame
    global nkr
    global lri
    global lle
#    screen = ImageGrab.grab()

    sct = mss()
    for num, monitor in enumerate(sct.monitors[1:], 1):
        sct_img = sct.grab(monitor)
        screen = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        screen = screen.resize((256, 256), resample=PIL.Image.BICUBIC)
#        time.sleep(0.15)
#        screen.save(settings.save_dir+"/lastframe"+str(random.randint(0, 60))+".png")
        print("screen saved", time.time(), sct_img.size, screen.size)
        break
    
    if(nrOfFrame%30 == 0):
        for key in keys.names:
            keyboard.release(key)
        lastPressedKeys = np.array([0, 0, 0, 0, 0, 0, 0])

    
    
    x = image.img_to_array(screen)[:, :, :3]
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
#    image = load_img(screen)
    images = preprocess_input(images)
    images /= 255.0
#    classes = model.predict_classes(images, batch_size=1)
    prediction = model.predict(images,verbose=1)
    print("------------------------------------------------------------------------------------------")
    print("\t\t", prediction)
    print("------------------------------------------------------------------------------------------")
    beep()
    nrOfFrame += 1
    thread1 = threading.Thread(target = pressKey, args = (prediction, ))
    thread1.start()
def pressKey(predictionl):
    global lastPressedKeys
    global nrOfFrame
    global nkr
    global lri
    global lle
    frameNow = nrOfFrame
    próg = 50.0
    while(frameNow == nrOfFrame):
        prediction = predictionl
#        prediction[0][1] -= 0.3
        próg = (3*próg+(2.0*random.randint(80, 105)))/500.0
#        nkr += 1
        print(prediction, próg, (lri/nkr), (lle/nkr))
        newPressedKeys = np.array([0, 0, 0, 0, 0, 0, 0])
#        for p in range(0, len(prediction[0])):
#            prediction[0][p] += random.randint(0, 60) - 30
        if(prediction[0][0] > próg - 0.7):
            print("up")
            newPressedKeys[0] = 1
        else:
            if(-1.0*prediction[0][0] > próg + 0.6):
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
        print(change, próg)
        for keyChange, keyName in zip(change, keys.names):
            if keyChange > 0:
                keyboard.press(keyName)
            if keyChange < 0:
                keyboard.release(keyName)
        time.sleep(0.05)



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
    model = load_model(settings.save_dir+"/"+settings.model_name)
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
    if driveMode:
        print("d")
#        drive(rr)
        thread = threading.Thread(target=drive)
        threads.append(thread)
        time.sleep(0.1)
        while(len(threads) > 1):
            threads[0].join()
            threads.pop(0)
        print("threads: ", len(threads))
        thread.start()
    else:
        print("r")
#        thread = threading.Thread(target=record.start)
#        #        threads.append(thread)n1
#        time.sleep(0.05)
#        thread.start()



# predicting multiple images at once
#img = image.load_img('test2.jpg', target_size=(img_width, img_height))
#y = image.img_to_array(img)
#y = np.expand_dims(y, axis=0)
#
## pass the list of multiple images np.vstack()
#images = np.vstack([x, y])
#classes = model.predict_classes(images, batch_size=10)
#
## print the classes, the images belong to
#print classes
#print classes[0]
#print classes[0][0]
