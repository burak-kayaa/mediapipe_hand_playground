from time import sleep
from typing import Any
from unittest import result
import cv2
import mediapipe as mp
from util import show_landmarks, calc_between_landmarks, draw_landmark, draw_between_landmarks

class hand_detection:
    HAND_WRIST: int = 0
    THUMB_BOT: int = 2
    THUMB_MID: int = 3
    THUMB_TOP: int = 4
    INDEX_BOT: int = 6
    INDEX_MID: int = 7
    INDEX_TOP: int = 8
    MIDDLE_BOT: int = 10
    MIDDLE_MID: int = 11
    MIDDLE_TOP: int = 12
    RING_BOT: int = 14
    RING_MID: int = 15
    RING_TOP: int = 16
    PINKY_BOT: int = 18
    PINKY_MID: int = 19
    PINKY_TOP: int = 20
    THUMB_JOINT: int = 1
    INDEX_JOINT: int = 5
    MIDDLE_JOINT: int = 9
    RING_JOINT: int = 13
    PINKY_JOINT: int = 17
    WINDOW_NAME: str = "Hand Detection"
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.mode = "default"
        self.brightness = 0
        self.contrast = 0
        
    def _determine_mode(self, frame) -> str:
        results = self._get_results(frame)
        if results is None:
            return "default"
        if len(results.multi_hand_landmarks) == 1:
            index_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[self.INDEX_BOT],
                results.multi_hand_landmarks[0].landmark[self.INDEX_TOP]
            )
            thumb_index_joint_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[self.INDEX_JOINT],
                results.multi_hand_landmarks[0].landmark[self.THUMB_TOP]
            )
            if index_length > 0.12 and thumb_index_joint_length > 0.12:
                print("Default mode")
                return "default"
            elif index_length > thumb_index_joint_length:
                print("Brightness mode")
                return "brightness"
            print("Contrast mode")
            return "default"
        return "default"

    def _mode_hub(self, index_length, thumb_index_joint_length, frame):
        start_time = cv2.getTickCount()
        if index_length > 0.12 and thumb_index_joint_length > 0.12:
            print("Default mode")
            self.mode = "default"
            return
        if index_length > thumb_index_joint_length:
            print("Brightness mode needs two hands. Selecting in 3 seconds")
            while True:
                results = self._get_results(frame)
                if results is None:
                    continue
                current_time = cv2.getTickCount()
                while current_time - start_time < 3 * cv2.getTickFrequency():
                    frame = self._read_capture()
                    current_time = cv2.getTickCount()
                    print("Executing brightness mode")
                    cv2.putText(frame, f"Time: {int((current_time - start_time) / cv2.getTickFrequency())}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(self.WINDOW_NAME, frame)
                    cv2.waitKey(1)
                    if self._determine_mode(frame) != "brightness":
                        self.mode = "default"
                        return
                while True:
                    print("Waiting for two hands")
                    frame = self._read_capture()
                    cv2.putText(frame, f"Waiting for two hands", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(self.WINDOW_NAME, frame)
                    cv2.waitKey(1)
                    results = self._get_results(frame)
                    if results is None:
                        continue
                    if len(results.multi_hand_landmarks) == 2:
                        print("Brightness mode")
                        self.mode = "brightness"
                        return
                
    def _exec_mode(self, frame):
        results = self._get_results(frame)
        if results is None:
            return frame
        if self.mode == "brightness":
            distance = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[self.INDEX_TOP],
                results.multi_hand_landmarks[1].landmark[self.INDEX_TOP]
            )
            frame = self._change_brightness(frame, distance)
        elif self.mode == "contrast":
            index_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[self.INDEX_BOT],
                results.multi_hand_landmarks[0].landmark[self.INDEX_TOP]
            )
            thumb_index_joint_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[self.INDEX_JOINT],
                results.multi_hand_landmarks[0].landmark[self.THUMB_TOP]
            )
            distance = index_length / thumb_index_joint_length
            frame = self._change_contrast(frame, distance)
        return frame

    def _change_contrast(self, image, distance):
        # Assuming distance is between 0 and 1
        contrast = int(distance * 255)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        return image

    def _change_brightness(self, image, distance):
        # Assuming distance is between 0 and 1
        self.brightness = int(distance * 255)
        image = cv2.convertScaleAbs(image, alpha=1, beta=self.brightness)
        return image

    def _get_results(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            print("No hands detected")
            return None
        return results
    
    def _read_capture(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            exit()
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)
        return frame

    def __call__(self) -> Any:
        while self.cap.isOpened():
            frame = self._read_capture()
            results = self._get_results(frame)
            if results is None:
                self.mode = "default"
                print("No hands detected")
            elif len(results.multi_hand_landmarks) == 1:
                print("Mode selecting")
                index_length = calc_between_landmarks(
                    results.multi_hand_landmarks[0].landmark[self.INDEX_BOT],
                    results.multi_hand_landmarks[0].landmark[self.INDEX_TOP]
                )
                thumb_index_joint_length = calc_between_landmarks(
                    results.multi_hand_landmarks[0].landmark[self.INDEX_JOINT],
                    results.multi_hand_landmarks[0].landmark[self.THUMB_TOP]
                )
                self._mode_hub(index_length, thumb_index_joint_length, frame)
                # draw landmarks
                draw_landmark(results.multi_hand_landmarks[0], frame)
                show_landmarks(results.multi_hand_landmarks[0], frame)
            else:
                frame = self._exec_mode(frame)
            cv2.imshow(self.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()