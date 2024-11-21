import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model
model = load_model('hand_gesture_model.h5')

# Define gesture classes
gesture_classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
le = LabelEncoder()
le.classes_ = np.array(gesture_classes)

def extract_hand_connections(image):
    """
    Extract hand connections (landmark vectors) from an input image using MediaPipe.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        connections = []
        for connection in mp_hands.HAND_CONNECTIONS:
            start = results.multi_hand_landmarks[0].landmark[connection[0]]
            end = results.multi_hand_landmarks[0].landmark[connection[1]]
            vector = [end.x - start.x, end.y - start.y, end.z - start.z]
            connections.extend(vector)
        return np.array(connections), results
    return None, None

def predict_gesture(frame, model, le):
    """
    Predict the hand gesture from the given frame using the trained model.
    """
    connections, results = extract_hand_connections(frame)
    if connections is not None:
        connections = connections.reshape(1, -1)  # Reshape for model input
        prediction = model.predict(connections)  # Predict probabilities
        gesture_index = np.argmax(prediction)  # Get index of highest probability
        gesture_label = le.inverse_transform([gesture_index])[0]  # Convert to label
        return gesture_label, results
    return None, None

# Real-time prediction using webcam
cap = cv2.VideoCapture(0)

# Initialize variables for gesture tracking
current_gesture = None
gesture_start_time = 0
gesture_duration = 1.5  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Predict the gesture
    gesture, results = predict_gesture(frame, model, le)

    # Check for gesture change
    if gesture is not None:
        if current_gesture != gesture:
            if time.time() - gesture_start_time >= gesture_duration:
                current_gesture = gesture
                gesture_start_time = time.time()
        else:
            gesture_start_time = time.time()

    # Display the gesture in a box at the bottom
    if current_gesture:
        # Define box properties
        box_color = (0, 0, 0)  # Black background for the box
        text_color = (255, 255, 255)  # White text color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 2
        
        # Get the size of the text
        text = f"Gesture: {current_gesture}"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Define box dimensions
        box_height = text_height + baseline + 20
        box_width = frame.shape[1]
        box_x1 = 0
        box_y1 = frame.shape[0] - box_height
        box_x2 = frame.shape[1]
        box_y2 = frame.shape[0]
        
        # Draw the box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, thickness=cv2.FILLED)
        
        # Calculate centered text position
        text_x = (box_width - text_width) // 2
        text_y = box_y1 + (box_height - text_height) // 2 + text_height
        
        # Overlay the text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Draw the hand landmarks
    if results and results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
