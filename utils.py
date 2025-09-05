import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import  Dense, Input, Dropout, MaxPooling2D, Conv2D, Concatenate, Embedding, Reshape, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import random
def importdata(path):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data= pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    print(data.head())
    print(data['center'][0])
    print(getname(data['center'][0]))
    data['center'] = data['center'].apply(getname)
    print(data.shape[0])
    return data

def getname(filepath):
    return filepath.split('\\')[-1]

def balance_data(data,display=True):
    nBin = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data['steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)
 
    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
     hist, _ = np.histogram(data['steering'], (nBin))
     plt.bar(center, hist, width=0.06)
     plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
     plt.show()
    return data 


def load_data(path,data):
    images_path=[]
    steering=[]
    for i in range (len(data)):
        index_data=data.iloc[i]
        images_path.append(os.path.join(path,'IMG',index_data[0]))
        steering.append(float(index_data[3]))

    images_path=np.asarray(images_path)
    steering=np.asarray(steering)

    return images_path,steering

def augment_image(image_path,steering):
    img=mpimg.imread(image_path)
    #pan
    if np.random.rand()<0.5:
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    #zoom
    if np.random.rand() < 0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)

    #brightness
    if np.random.rand() < 0.5:
        brightness=iaa.Multiply((0.2,1.2))
        img = brightness.augment_image(img)

    #flip
    if np.random.rand() < 0.5:
        img=cv2.flip(img,1)
        steering=-steering
     #blur image with random kernel
    if np.random.rand() < 0.5:  
        kernel_size=random.randint(1,5)
        if kernel_size%2!=1:
            kernel_size+=1
        img=cv2.GaussianBlur(img,(kernel_size,kernel_size),0)    
    #brightness
    if np.random.rand() < 0.5:  
        img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        brightness=np.random.uniform(0.5,1.1)
        img[:,:,2]=img[:,:,2]*brightness
        img=cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

      #rotate
    if np.random.rand() < 0.5:
        rotate = random.uniform(-1, 1)
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotate, 1)
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0])) 
    #img shift
    if np.random.rand() < 0.5:
        trans_range=80
        shift_x=trans_range*np.random.uniform()-trans_range/2
        shofy_y=40*np.random.uniform()-40/2
        m=np.float32([[1,0,shift_x],[0,1,shofy_y]])
        img=cv2.warpAffine(img,m,(img.shape[1],img.shape[0]))
        steering+=shift_x/trans_range*2*0.2          


    return img,steering



'''imgre,st=augment_image('test.jpg',0)
plt.imshow(imgre)
plt.show()'''

def  preprocessing(img):
    #will crop the image
    img=img[60:135,:,: ]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255

    return img

'''imgre=preprocessing(mpimg.imread('test.jpg'))
plt.imshow(imgre)
plt.show()'''

def batch_size(images_path,steering_list,batch_size,train_flag):
    while True:
        img_batch=[]
        steeering_batch=[]
        for i in range (batch_size):
            index=np.random.randint(0,len(images_path)-1)
            if train_flag:
                img,steering=augment_image(images_path[index],steering_list[index])
            else:
               img=mpimg.imread(images_path[index])
               steering=steering_list[index]
            img=preprocessing(img)
            img_batch.append(img)
            steeering_batch.append(steering)
        yield(np.asarray(img_batch),np.asarray(steeering_batch))

def Model():
    model= Sequential()

    #layer 1
    model.add(Conv2D(24,(5,5),activation='elu',strides=(2,2), padding='valid',
              kernel_initializer='he_normal',input_shape=(66,200,3)))
     #he_normal initialize the weights by drawing random values from a gaussian distribution
     # with a mean of 0 and standard deviation calculated based on the size of th weight tensor
     #layer2
    model.add(Conv2D(36,(5,5),activation='elu',strides=(2,2), padding='valid',
              kernel_initializer='he_normal'))
    #layer 3
    model.add(Conv2D(48,(5,5),activation='elu',strides=(2,2), padding='valid',
              kernel_initializer='he_normal'))
    #layer 4
    model.add(Conv2D(64,(3,3),activation='elu',padding='valid',
              kernel_initializer='he_normal'))
    #layer 5
    model.add(Conv2D(64,(3,3),activation='elu',padding='valid',
              kernel_initializer='he_normal'))
    
    model.add(Flatten())    
    # fully connected neural network
    #fc1
    model.add(Dense(100,activation='elu',kernel_initializer='he_normal'))
    #fc2
    model.add(Dense(50,activation='elu',kernel_initializer='he_normal'))
    #fc3
    model.add(Dense(10,activation='elu',kernel_initializer='he_normal'))
    #output layer
    model.add(Dense(1,kernel_initializer='he_normal'))

    #compile the model
    model.compile(optimizer=Adam(lr=0.000001),loss='mean_squared_error',metrics='MSE')
  
    return model
