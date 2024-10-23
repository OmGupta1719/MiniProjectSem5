import cv2
import numpy as np
from keras.models import model_from_json
import mediapipe as mp
from function import mediapipe_detection, extract_keypoints


mp_hands = mp.solutions.hands


try:
    with open("model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()


actions = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [str(i) for i in range(10)]  # 'A' to 'Z' and '0' to '9'


sequence = []
sentence = []
predictions = []
threshold = 0.8


cap = cv2.VideoCapture(0)


with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break


        cropframe = frame[40:400, 0:300]
        image, results = mediapipe_detection(cropframe, hands)


        if results.multi_hand_landmarks:
            print("Hands detected!")
        else:
            print("No hands detected.")


        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames

        if len(sequence) == 30:

            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))


            if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    sentence.append(actions[np.argmax(res)])


        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        if sentence:
            cv2.putText(frame, f"Prediction: {' '.join(sentence[-1:])}", (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Prediction: ", (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('Webcam Feed', frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
