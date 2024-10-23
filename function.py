import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


def extract_keypoints(results, num_keypoints=63):
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            keypoints.append(rh)


    if len(keypoints) == 0:
        return np.zeros(num_keypoints)


    return keypoints[0]


sequence_buffer = []


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)


    num_keypoints = 63

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image, results = mediapipe_detection(image, hands)
            draw_styled_landmarks(image, results)


            keypoints = extract_keypoints(results, num_keypoints)
            sequence_buffer.append(keypoints)


            if len(sequence_buffer) > 30:
                sequence_buffer.pop(0)


            if len(sequence_buffer) == 30:

                sequence_data = np.array(sequence_buffer)
                print("Sequence Data Shape:", sequence_data.shape)



            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
