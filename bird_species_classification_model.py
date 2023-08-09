import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.layers import MaxPool2D, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
import pickle

DATA_PATH ="/content/drive/MyDrive/data.json"

def load_data(data_path):
  #Loads training dataset from json file

  with open(data_path,"r") as fp:
    data = json.load(fp)

  X=np.array(data["mfcc"])
  y=np.array(data["labels"])
  return X, y


def prepare_datasets(test_size, validation_size):
  #Load data
  X, y =load_data(DATA_PATH)

  #Create train/test split
  X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=test_size)

  #create train/valid split
  X_train,X_validation,y_train,y_validation=train_test_split(X_train,y_train, test_size=validation_size)

  #3d array ->(130,3,1)
  X_train = X_train[...,np.newaxis] # will be a 4d array with num_samples inclusive
  X_test = X_test[...,np.newaxis]
  X_validation = X_validation[...,np.newaxis]

  return X_train,X_validation,X_test,y_train,y_validation,y_test


def build_model(input_shape):
  #create a model
  model=keras.Sequential()

  #1st Conv Layer
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  #2nd Conv Layer
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())

  #3rd Conv Layer
  model.add(Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())

  #flatten the output and feed it into dense layer
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(64,activation='relu'))
  model.add(keras.layers.Dropout(0.3))# prevent over fitting

  #output layer
  model.add(keras.layers.Dense(15,activation='softmax'))

  return model

def predict(model, X, y):
  X =X[np.newaxis, ...]

  #prediction =[[]]=> 2D array
  prediction = model.predict(X)#4D array ->

  #extract index with max value
  predicted_index = np.argmax(prediction, axis=1)
  print("Expected index: {}, Predicted index: {}".format(y,predicted_index))




if __name__ =="__main__":
  #create train, validation and test sets
  X_train,X_validation,X_test,y_train,y_validation,y_test=prepare_datasets(0.25,0.2)

  #buid the CNN network
  input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
  model = build_model(input_shape)

  #compile the CNN network
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

  model.compile(optimizer = optimizer,
                loss = "sparse_categorical_crossentropy",
                metrics = ['accuracy'])

  #train the CNN
  model.fit(X_train,y_train,validation_data=(X_validation, y_validation), batch_size=32, epochs=70)

  #evaluate the CNN on the test set
  test_error, test_accuracy = model.evaluate(X_test,y_test, verbose=1)
  print("Accuracy on the set is: {} ".format(test_accuracy))

   # Make predictions on the test set
  X= X_test[100]
  y= y_test[100]
  predict(model, X, y)

import librosa
def extract_mfcc(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T  # Transpose to match the expected input shape

    return mfcc[np.newaxis, ...]  # Add a new axis for batching

new_audio_path = "/content/drive/MyDrive/All1/major/CRYVAR08.mp3"
new_mfcc = extract_mfcc(new_audio_path)
# Assuming you've already extracted new_mfcc from the new audio file
# new_mfcc.shape is (1, 6422, 13)

# Reshape the new_mfcc array to have the expected time steps (65)
new_mfcc_reshaped = new_mfcc[:, :65, :]

# Add the channel dimension (1) to match the model's input shape
new_mfcc_reshaped = new_mfcc_reshaped[..., np.newaxis]

# Now new_mfcc_reshaped.shape is (1, 65, 13, 1)

from tensorflow.keras.models import load_model

model_path = "/content/bird_specie.h5"
loaded_model = load_model(model_path)

def predict_species(model, mfcc):
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)
    return predicted_index

data_path="/content/drive/MyDrive/dat1.json"
with open(data_path,"r") as fp:
    data = json.load(fp)


predicted_species_index = predict_species(loaded_model, new_mfcc_reshaped)
predicted_species = data["mapping"][predicted_species_index[0]]
print("Predicted species:", predicted_species)


