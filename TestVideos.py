'''

This program loads the trained model and predicts arousal, valence, and
likability values for testing dataset.

'''

from keras.models import model_from_json
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
from keras.applications import ResNet50
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#reads the saved model and predicts values for testing dataset
def predict( x ):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    results = loaded_model.predict(x, batch_size = 20)
    return results

#loads testing videos
def loadTestVideos(files):
    images = []
    for filename in files:

        cap = cv2.VideoCapture('Data/video/' + filename + '.avi')

        while True:
            boolen, frame = cap.read()  # get the frame

            if boolen:
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                # print(pos_frame)
                if (pos_frame - 1) % 5 == 0:
                    frame = cv2.resize(frame, (224, 224))
                    images.append(frame)

            if cv2.waitKey(10) == 27:
                break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break

    all_frames = np.array(images, dtype=np.float64)
    all_frames = preprocess_input(all_frames)
    features = model.predict(all_frames)
    features = features.reshape(all_frames.shape[0], 2048, 1)
    print(features.shape)
    return features

def loadTestLabels(files):
    df2 = pd.DataFrame([])
    for filename in files:
        df = pd.read_csv('Data/labels/' + filename + '.csv', sep=';', header=None)
        df1 = df.ix[:, 2:5]
        df2 = df2.append(df1)

    labels = np.array(df2)
    #print(labels.shape)
    return labels

def test_main():
    testing_files = ['Test_02']
    print("testing")
    x = loadTestVideos(testing_files)
    rem = x.shape[0] % 20
    x = x[:-rem, :]
    print(x.shape)
    print("Loaded video")
    # y = loadTestLabels(testing_files)
    # y = y[:-rem, :]
    results = predict(x)
    print(results)
    np.savetxt('results.csv', results, delimiter = ' ')
    #arousal_rmse = math.sqrt(mean_squared_error(y[:,0], results[:,0]))
    #print("Arousal: ",arousal_rmse)
    #valence_rmse = math.sqrt(mean_squared_error(y[:,1], results[:,1]))
    #print("Valence: ",valence_rmse)
    #likability_rmse = math.sqrt(mean_squared_error(y[:,2], results[:,2]))
    #print("Likability: ",likability_rmse)

test_main()
