from keras.models import Sequential
import os
from os import listdir
from os.path import isfile, join
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
def init():
    global batch_size
    global maxEpochSize
    global removeEveryNBatch
    global num_classes
    global imagesNumberInDirectory
    global epochs
    global data_augmentation
    global num_predictions
    global save_dir
    global saveNow
    global directory
    global directoryval
    global directorytest
    global model_name
    global model
    global stopTraining
    global shouldShowPlots
    global history
    global historyAvg
    
    
    batch_size = 256
    num_classes = 1000.0#len(listdir(directory))
    epochs = 70
    data_augmentation = False
    maxEpochSize = 5000
    
    
    save_dir = os.path.join('/Users/jacekkaluzny/Documents/🛍moje programy/gameAINet', 'saved_models')
    saveNow = 0
    
    if(os.path.exists("/Volumes/NN/network-50.icns") and os.path.isfile("/Volumes/NN/network-50.icns")):
        directory = '/Volumes/NN/car/carGame2/photos/'
    else:
        if(os.path.exists("/Volumes/Flash⚡️ 1/icon.ICNS") and os.path.isfile("/Volumes/Flash⚡️ 1/icon.ICNS")):
            directory = '/Volumes/Flash⚡️ 1/carGame2/photos/'
        else:
            if(os.path.exists("/Volumes/Flash⚡️ 2/icon.ICNS") and os.path.isfile("/Volumes/Flash⚡️ 2/icon.ICNS")):
                directory = '/Volumes/Flash⚡️ 2/carGame2/photos/'
            else:
                directory = '/Users/jacekkaluzny/Pictures/carGame2/photos/'

#    directory = '/Volumes/Flash⚡️/carGame2/photos'
    directoryval = directory+'val/'
    directorytest = directory+'test/'
    directory = directory+'train/'

    
    model_name = 'selfDrivingFastE.h5'
    model = Sequential()
    
    stopTraining = False
    shouldShowPlots = False
    
    
    history = None
    historyAvg = dict()
    
    
    historyAvg['mean_squared_error'] = []
    historyAvg['val_mean_squared_error'] = []
    historyAvg['mean_absolute_error'] = []
    historyAvg['val_mean_absolute_error'] = []
    def showPlots():
        historyAvg = self.historyAvg
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
