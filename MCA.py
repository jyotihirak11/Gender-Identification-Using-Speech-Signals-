#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D,MaxPooling2D
from glob import glob
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#Librosa for audio
import librosa as lr
#And the display module for visualization
import librosa.display


# In[2]:



data=pd.read_csv('E:/AI/Gender Recognition of Speaker/training samples/voices/voicesample1.csv')
data.head(5)
#E:\AI\Gender Recognition of Speaker\training samples\voices


# In[3]:


#Display No. of rows and columns 
data.shape


# In[4]:


#read data
data_dir = 'E:/AI/Gender Recognition of Speaker/training samples/voices'
audio_files = glob(data_dir + '/*.flac')
#files = librosa.util.find_files('E:/AI/Gender Recognition of Speaker/LibriSpeech/train-clean-100', ext='flac')
print(len(audio_files))


# In[5]:


# Load the audio as a waveform `y`
# Store the sampling rate as `sr`
y,sr=lr.load(audio_files[5], duration=2.97)
print(y)
print(sr)
plt.plot(y)
# Let's make and display a mel-scaled power (energy-squared) spectrogram
ps=librosa.feature.melspectrogram(y=y, sr=sr)
print(ps)
ps.shape


# In[6]:


# Display the spectrogram on a mel scale
librosa.display.specshow(ps, y_axis='mel', x_axis='time')


# In[7]:


y,sr=lr.load(audio_files[16], duration=2.97)
print(y)
print(sr)
plt.plot(y)
# Let's make and display a mel-scaled power (energy-squared) spectrogram
ps=librosa.feature.melspectrogram(y=y, sr=sr)
print(ps)


# In[8]:


librosa.display.specshow(ps, y_axis='mel', x_axis='time')


# In[9]:


D=[] #DataSet
y,sr=lr.load(audio_files[0], duration=2.97)
ps=librosa.feature.melspectrogram(y=y, sr=sr)
D.append((ps,audio_files[0]))
print(D)


# In[5]:


D=[] #DataSet
for row in data.itertuples():
   print(row)


# In[6]:


D=[] #DataSet
for row in data.itertuples():
   # print(row)
     y,sr=lr.load('E:/AI/Gender Recognition of Speaker/training samples/voices/' + row.Filename, duration=2.97)
     ps=librosa.feature.melspectrogram(y=y, sr=sr)
     if ps.shape !=(128,128):
        #print(file)
        continue
     D.append((ps,row.Class))
print(D)


# In[92]:


'''D=[] #DataSet
for file in range (0,len(audio_files), 1):
    y,sr=lr.load(audio_files[file], duration=2.97)
    ps=librosa.feature.melspectrogram(y=y, sr=sr)
       if ps.shape !=(128,128):
        #print(file)
        continue
    D.append((ps,audio_files[file]))
print(D)'''


# In[12]:


print("Number of samples:",len(D))


# In[7]:


dataset = D
random.shuffle(dataset)
train=dataset[:300]

print(train)


# In[8]:


test=dataset[300:]
print(test)


# In[9]:


X_train, Y_train = zip(*train)
print(X_train)


# In[10]:


print(Y_train)


# In[11]:


X_test, Y_test = zip(*test)


# In[12]:


#Reshape for CNN input
X_train = np.array([x.reshape((128,128,1)) for x in X_train])
X_test = np.array([x.reshape((128,128,1)) for x in X_test])
print(X_train)


# In[13]:


# One-Hot encoding for classes
Y_train = np.array(keras.utils.to_categorical(Y_train,2))
Y_test = np.array(keras.utils.to_categorical(Y_test,2))
print(Y_train)


# In[14]:


print(Y_test)


# In[15]:


model = Sequential()
input_shape=(128,128,1)
model.add(Conv2D(24,(5,5), strides=(1,1), input_shape=input_shape))
model.add(MaxPooling2D ((4,2), strides= (4,2)))
model.add(Activation('relu'))
model.add(Conv2D (48, (5,5), padding = 'valid'))
model.add(MaxPooling2D ((4,2), strides = (4,2)))
model.add(Activation('relu'))

model.add(Conv2D (48, (5,5), padding = 'valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout( rate = 0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate= 0.5))


model.add(Dense(2))
model.add(Activation('softmax'))


# In[17]:


model.compile(
    optimizer="Adam",
    loss = "categorical_crossentropy",
    metrics=['accuracy'])


# In[18]:


history=model.fit(
    x = X_train,
    y = Y_train,
    epochs = 50,
    batch_size = 40,
    validation_data = (X_test, Y_test))


# In[19]:



score=model.evaluate(
    x=X_test,
    y=Y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[27]:


model.summary()


# In[20]:


plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:




