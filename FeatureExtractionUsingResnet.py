'''

This program reads videos, extracts features using ResNet50 and trains LSTM model
for emotion recognition.

'''
import numpy as np
import cv2
from keras.applications import ResNet50
from keras.models import Model, model_from_json
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#loading label csv files
def loadDataLabels(filenames):
    df2 = pd.DataFrame([])
    for filename in filenames:
        df = pd.read_csv('Data/labels/' + filename + '.csv', sep=';', header=None)
        df1 = df.ix[:, 2:5]
        df2 = df2.append(df1)

    labels = np.array(df2)
    #print(labels.shape)
    return labels

#reading videos
def loadVideos(filenames):
    print("Loading videos")
    images = []
    for filename in filenames:

        cap = cv2.VideoCapture('Data/video/'+filename+'.avi')

        while True:
            boolen, frame = cap.read() # get the frame

            if boolen:
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                # taking only those frames for which labels are recorded
                if (pos_frame-1)%5 == 0:
                    frame = cv2.resize(frame, (224, 224))
                    images.append(frame)


            if cv2.waitKey(10) == 27:
                break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break

    all_frames = np.array(images, dtype = np.float64)
    all_frames = preprocess_input(all_frames)
    features = model.predict(all_frames)
    features = features.reshape(all_frames.shape[0], 2048, 1)
    print(features.shape)
    return features

# Building LSTM
def buildLSTMModel(x_train, y_train, x_val, y_val, x_test, y_test):
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(20, x_train.shape[1], x_train.shape[2]), return_sequences=False, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='linear'))
    model.compile(loss='mse', optimizer='RMSprop', metrics=['mse'])
    history = model.fit(x_train, y_train, batch_size=20, epochs=5, validation_data=(x_val, y_val), shuffle=False)
    loss, acc = model.evaluate(x_test, y_test, batch_size=20, verbose=0)
    results = model.predict(x_test, batch_size = 20)
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    #saving trained model on local machine
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
       json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def main():
    training_files = ['Train_01', 'Train_02', 'Train_03', 'Train_04', 'Train_05',
                      'Train_06', 'Train_07', 'Train_08', 'Train_09', 'Train_10',
                      'Train_11', 'Train_12', 'Train_13', 'Train_14', 'Train_15',
                      'Train_16', 'Train_17', 'Train_18', 'Train_19', 'Train_20']
    development_files = ['Devel_01', 'Devel_02', 'Devel_03', 'Devel_04', 'Devel_05',
                         'Devel_06', 'Devel_07', 'Devel_08', 'Devel_09', 'Devel_10',
                         'Devel_11', 'Devel_12', 'Devel_13', 'Devel_14']

    #loading training videos
    x_train = loadVideos(training_files)

    #loading training labels
    y_train = loadDataLabels(training_files)

    #making number of frames divisible by batch size
    x_train, y_train = x_train[:-2, :], y_train[:-2,:]

    #loading validation videos
    x_dev = loadVideos(development_files)
    x_dev = x_dev[:-8,:]

    #loading validation labels
    y_dev = loadDataLabels(development_files)
    y_dev = y_dev[:-8, :]

    #dividing validation dataset into vaildation and testing dataset
    x_test, y_test =  x_dev[17720:, :], y_dev[17720:, :]
    x_val, y_val = x_dev[:17720, :], y_dev[:17720, :]

    buildLSTMModel(x_train, y_train, x_val, y_val, x_test, y_test)

main()

