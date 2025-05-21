import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
from datetime import datetime

dataset_path = "My_Dataset"
labels = ['A', 'Am', 'C', 'D', 'E', 'F', 'G']
extracted_features = []

def extract_features(file):
    audio_data, sample_rate = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

for label in labels:
    folder_path = os.path.join(dataset_path, label)
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            data = extract_features(file_path)
            extracted_features.append([data, label]) 

extracted_features_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
#print(extracted_features_features_df)

X=np.array(extracted_features_features_df['feature'].tolist())
y=np.array(extracted_features_features_df['class'].tolist())
print(y)
y=np.array(pd.get_dummies(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_labels = y.shape[1]

model = Sequential()

#Layer 1
model.add(Dense(100, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Layer 2
model.add(Dense(200, input_shape=(7,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Layer 3
model.add(Dense(100, input_shape=(7,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Final Layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

#print(model.summary())

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
num_epochs = 100
n_batch_size=32
checkpointer = ModelCheckpoint(filepath='saved_models/firsttry.keras', verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=n_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer])
duration=datetime.now()-start
print("Training Completed in :",duration)
test_accuracy = model.evaluate(X_test, y_test, verbose = 0)
print(test_accuracy[1])

file = "Testing Samples/E.wav"
pred_feature = extract_features(file)
pred_feature = pred_feature.reshape(1,-1)

predicted_class_index = np.argmax(model.predict(pred_feature), axis=1)
predicted_class_label = labels[predicted_class_index[0]]
print(predicted_class_label)