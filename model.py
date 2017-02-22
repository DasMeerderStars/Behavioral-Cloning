import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines=[]
with open('./data/driving_log.csv') as csvfile:
    csvfile.readline()
    reader=csv.reader(csvfile)
    for line in reader:
         lines.append(line)
print(len(lines))
images=[]
measurements=[]
for line in lines:
    for i in range(3):
        #print(len(line))
        #print(line[3])
        source_path=line[0]
        tokens=source_path.split('/')
        filename=tokens[-1]
        local_path="./data/IMG/"+filename
        #print(local_path)
        image=cv2.imread(local_path)
        images.append(image)
    correction=0.2
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
        #print(images)
        #print(len(measurements))

augmented_images=[]
augmented_measurements=[]
for image, measurement in zip (images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image=cv2.flip(image,1)
    flipped_measurement=measurement*-1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)


#print(images[0].shape())
#print(len(measurements))
X_train=np.asarray(augmented_images)
y_train=np.asarray(augmented_measurements)
print(X_train.shape)
X_train, y_train = shuffle(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size=0.20,random_state=42)
#X_train=np.asarray(images)
#y_train=np.asarray(measurements)

def generator(X, y, batch_size):
    while True:
        # X, y = shuffle(X, y)
        smaller =len(X)
        iterations = int(smaller/batch_size)
        for i in range(iterations):
            start, end = i * batch_size, (i + 1) * batch_size
            yield X[start:end], y[start:end]



import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D

train_generator = generator(X_train, y_train,batch_size=128)
validation_generator = generator(X_valid,y_valid, batch_size=128)

model=Sequential()
model.add(Lambda(lambda x:x/255-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
#model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
#model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(X_train),validation_data=validation_generator,nb_val_samples=len(y_valid), nb_epoch=5)
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3)
model.save('model.h5')







    
