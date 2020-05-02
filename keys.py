import keyboard
import string
import random
import os
from os import listdir
from os.path import isfile, join


def init():
    global names
    global states
    global className
    names = ['up', 'down', 'right', 'left', 'r', 'space', 'k']
def states():
    global names
    statesArray = []
    for name in names:
        statesArray.append(keyboard.is_pressed(name))
def className(textBefore = "", textAfter = ".png"):
    global names
#    if(os.path.exists("/Volumes/Flash⚡️/icon.ICNS") and os.path.isfile("/Volumes/Flash⚡️/icon.ICNS")):
#        statesString = "/Volumes/Flash⚡️/carGame2/photos/"
#    else:
#        if(os.path.exists("/Volumes/Flash⚡️ 1/icon.ICNS") and os.path.isfile("/Volumes/Flash⚡️ 1/icon.ICNS")):
#            statesString = "/Volumes/Flash⚡️ 1/carGame2/photos/"
#        else:
#            if(os.path.exists("/Volumes/Flash⚡️ 2/icon.ICNS") and os.path.isfile("/Volumes/Flash⚡️ 2/icon.ICNS")):
#                statesString = "/Volumes/Flash⚡️ 2/carGame2/photos/"
    statesString = ""
    for name in names:
        if(keyboard.is_pressed(name)):
            statesString += "k1"
        else:
            statesString += "k0"
    try:
        os.makedirs(statesString)
    except OSError:
        print ("error saving")
#    statesString += "/"
#    statesString += ''.join(random.choices(string.ascii_uppercase + string.digits, k=32))
    return textBefore + statesString + textAfter
