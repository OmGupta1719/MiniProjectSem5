import cv2
import numpy as np
import os
import pandas as pd
import mediapipe as mp


mp_hands = mp.solutions.hands


DATA_PATH = 'dataset'
OUTPUT_PATH = 'processeddata_csv'
ACTIONS = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z'])


hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):

    keypoints = np.zeros(21 * 3)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()

    return keypoints


def create_dataset():

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        frames = os.listdir(action_path)


        sequences = []

        for frame in frames:
            frame_path = os.path.join(action_path, frame)
            if frame_path.endswith('.jpg') or frame_path.endswith('.png'):  # Check if the file is an image
                image = cv2.imread(frame_path)
                if image is not None:
                    image, results = mediapipe_detection(image, hands)
                    keypoints = extract_keypoints(results)
                    sequences.append(keypoints)


        if len(sequences) > 30:
            sequences = sequences[-30:]
        elif len(sequences) < 30:

            padding = np.zeros((30 - len(sequences), 63))
            sequences = np.vstack((sequences, padding))


        if sequences:
            sequences = np.array(sequences)
            csv_path = os.path.join(OUTPUT_PATH, f'{action}.csv')
            pd.DataFrame(sequences).to_csv(csv_path, index=False)


if __name__ == "__main__":
    create_dataset()
    print("Data preprocessing complete! Keypoints saved as CSV files.")
