import cv2
import mediapipe as mp
from typing import Any

def show_landmarks(hand_landmarks: mp.solutions.hands.HandLandmark, frame: Any) -> None: 
    """
    Draws the hand landmarks on the image.
    """
    for i, landmark in enumerate(hand_landmarks.landmark):
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
def calc_between_landmarks(landmark1: mp.solutions.hands.HandLandmark, landmark2: mp.solutions.hands.HandLandmark) -> float:
    """
    Calculates the distance between two landmarks.
    """
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def draw_landmark(landmark: mp.solutions.hands.HandLandmark, frame: Any) -> None:
    """
    Draws the hand landmarks on the image.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        frame,
        landmark,
        mp.solutions.hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
    
def draw_between_landmarks(landmark1: mp.solutions.hands.HandLandmark, landmark2: mp.solutions.hands.HandLandmark, frame: Any) -> None:
    """
    Draws a line between two landmarks on the image.
    """
    x1, y1 = int(landmark1.x * frame.shape[1]), int(landmark1.y * frame.shape[0])
    x2, y2 = int(landmark2.x * frame.shape[1]), int(landmark2.y * frame.shape[0])
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)