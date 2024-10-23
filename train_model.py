import cv2
import numpy as np
import os
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense


def load_data(data_directory):
    sequences = []
    labels = []
    actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                        'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])


    for action in actions:
        action_path = os.path.join(data_directory, f"{action}.csv")
        if os.path.exists(action_path):
            data = pd.read_csv(action_path)  # Read the CSV file into a DataFrame
            sequences.append(data.values)  # Append the keypoints as numpy array
            labels.append(actions.tolist().index(action))  # Get the label index

    return np.array(sequences), np.array(labels)


DATA_DIRECTORY = 'processeddata_csv'  # Update this to your actual CSV directory


X, y = load_data(DATA_DIRECTORY)


print("Original shape of X:", X.shape)


total_frames = sum([seq.shape[0] for seq in X])
print("Total frames across all sequences:", total_frames)


num_samples = len(X)

new_sequence_length = 30
new_feature_length = 63


reshaped_X = []

for seq in X:

    if seq.shape[0] > new_sequence_length:
        seq = seq[:new_sequence_length]

    elif seq.shape[0] < new_sequence_length:
        padding = np.zeros((new_sequence_length - seq.shape[0], seq.shape[1]))
        seq = np.vstack((seq, padding))

    reshaped_X.append(seq)


reshaped_X = np.array(reshaped_X)

print("Reshaped shape of X:", reshaped_X.shape)

y = to_categorical(y).astype(int)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))  # Updated input shape
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.fit(reshaped_X, y, epochs=100, batch_size=64, verbose=1)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

print("Model trained and saved successfully.")
