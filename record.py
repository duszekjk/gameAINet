#!/usr/bin/env python3
from PIL import ImageGrab, Image
import PIL
import keyboard
import random
import os
import time
import numpy as np
import keys
keys.init()
import time
import threading
from mss import mss


import settings
settings.init()
#os.system("cd /Volumes/Flash⚡️/carGame")
#os.system("ls")
images = []
plik = open(settings.directory+"opis.csv")

def start():
    global images
    global plik
#    screen = ImageGrab.grab()
    sct = mss()
    for num, monitor in enumerate(sct.monitors[1:], 1):
        sct_img = sct.grab(monitor)
        screen = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        screen = screen.resize((256, 256), resample=PIL.Image.BICUBIC)
        nameNewImg = settings.directory+"images/s"+str(time.time())+str(random.randint(1000000,9999999))+".png"
        screen.save(nameNewImg)
        time.sleep(0.01)
#        if(keys.className()[1] == "1" and random.randint(0,100) > 80):
#            break
#        if(keys.className()[0:8] == "k0k0k0k0" and random.randint(0,100) > 66):
#            break
#        if(keys.className()[0:8] == "k1k0k0k0" and random.randint(0,100) > 66):
#            break
        if(len(images) > 1):
            plik.write("\n, "+keys.className()+", "+images[0]+", "+images[3])
            screen.save(keys.className())
            images[3] = images[2]
            images[2] = images[1]
            images[1] = images[0]
            images[0] = nameNewImg
            print("screen saved", time.time(), sct_img.size, images)
        break
#    os.system("screencapture -t jpg -x -m "+keys.className())

def init():
    keyboard.wait('s')
    threads = []
    while(1):
        thread = threading.Thread(target=start)
#        threads.append(thread)
        time.sleep(0.02)
        thread.start()

