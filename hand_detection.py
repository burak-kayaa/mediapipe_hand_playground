from enum import Enum
from time import sleep
from typing import Any
from unittest import result
from enums.hand_landmark import HandLandmark
from enums.control_mode import ControlMode
import cv2
import mediapipe as mp
from util import show_landmarks, calc_between_landmarks, draw_landmark, draw_between_landmarks


class HandGestureController:
    """
    Hand Gesture Controller using MediaPipe for hand tracking.
    This class detects hand gestures and performs actions based on the detected gestures.
    """
    WINDOW_NAME: str = "Hand Gesture Controller"
    
    def __init__(
        self,
        camera_idx: int = 0,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5
    ) -> None:
        """
        Initializes the HandGestureController with the specified camera index and maximum number of hands.
        
        Args:
            camera_idx (int): The index of the camera to use (default is 0).
            max_num_hands (int): The maximum number of hands to detect (default is 2).
            min_detection_confidence (float): The minimum confidence for hand detection (default is 0.5).
        """
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )
        self._video_capture = cv2.VideoCapture(camera_idx)
        if not self._video_capture.isOpened():
            raise IOError(f"Cannot open webcam {camera_idx}")
        self._current_mode = ControlMode.DEFAULT
        self.brightness = 0
        self.contrast = 0
        
    def _determine_mode(
        self,
        frame: cv2.typing.MatLike
    ) -> str:
        """
        Determines the mode based on the hand landmarks detected in the frame.
        
        Args:
            frame: The current video frame.
        """
        results = self._get_results(frame)
        if results is None:
            return "default"
        if len(results.multi_hand_landmarks) == 1:
            index_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_BOT.value],
                results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_TOP.value]
            )
            thumb_index_joint_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_JOINT.value],
                results.multi_hand_landmarks[0].landmark[HandLandmark.THUMB_TOP.value]
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

    def _mode_hub(
        self,
        index_length: float,
        thumb_index_joint_length: float,
        frame: cv2.typing.MatLike
    ) -> None:
        """
        Hub for determining the mode based on hand landmarks.
        Counts the time for mode selection and switches modes accordingly.
        
        Args:
            index_length: Length between index finger bottom and top landmarks.
            thumb_index_joint_length: Length between thumb joint and index top landmarks.
            frame: The current video frame.
        """
        start_time = cv2.getTickCount()
        if index_length > 0.12 and thumb_index_joint_length > 0.12:
            print("Default mode")
            self._current_mode = ControlMode.DEFAULT
            return
        if index_length > thumb_index_joint_length:
            print("Brightness mode needs two hands. Selecting in 3 seconds")
            while True:
                results = self._get_results(frame)
                if results is None:
                    continue
                current_time = cv2.getTickCount()
                while current_time - start_time < 3 * cv2.getTickFrequency():
                    frame = self._read_frame()
                    current_time = cv2.getTickCount()
                    print("Executing brightness mode")
                    self._display_frame(frame, f"Brightness mode in {int((current_time - start_time) / cv2.getTickFrequency())} seconds")
                    if self._determine_mode(frame) != "brightness":
                        self._current_mode = ControlMode.DEFAULT
                        return
                while True:
                    frame = self._read_frame()
                    self._display_frame(frame, "Waiting for two hands")
                    results = self._get_results(frame)
                    if results is None:
                        continue
                    if len(results.multi_hand_landmarks) == 2:
                        print("Brightness mode")
                        self._current_mode = ControlMode.BRIGHTNESS
                        return
                
    def _exec_mode(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Executes the current mode based on the detected hand landmarks.
        
        Args:
            frame: The current video frame.
        """
        results = self._get_results(frame)
        if results is None:
            return frame
        if self._current_mode == ControlMode.BRIGHTNESS:
            distance = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_TOP.value],
                results.multi_hand_landmarks[1].landmark[HandLandmark.INDEX_TOP.value]
            )
            frame = self._change_brightness(frame, distance)
        elif self._current_mode == ControlMode.CONTRAST:
            index_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_BOT.value],
                results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_TOP.value]
            )
            thumb_index_joint_length = calc_between_landmarks(
                results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_JOINT.value],
                results.multi_hand_landmarks[0].landmark[HandLandmark.THUMB_TOP.value]
            )
            distance = index_length / thumb_index_joint_length
            frame = self._change_contrast(frame, distance)
        return frame

    def _display_frame(
        self,
        frame: cv2.typing.MatLike,
        str: str,
        show_image: bool = True
    ) -> None:
        """
        Displays the current frame with the specified string.
        
        Args:
            frame: The current video frame.
            str: The string to display on the frame.
            show_image: Whether to show the image in a window (default is True).
        """
        cv2.putText(frame, f"Mode: {str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if show_image:
            cv2.imshow(self.WINDOW_NAME, frame)
            cv2.waitKey(1)

    def _change_contrast(
        self,
        image: cv2.typing.MatLike,
        distance: float
    ) -> cv2.typing.MatLike:
        """
        Changes the contrast of the image based on the distance.
        """
        contrast = int(distance * 255)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        return image

    def _change_brightness(
        self,
        image: cv2.typing.MatLike,
        distance: float
    ) -> cv2.typing.MatLike:
        """
        Changes the brightness of the image based on the distance.
        """
        self.brightness = int(distance * 255)
        image = cv2.convertScaleAbs(image, alpha=1, beta=self.brightness)
        return image

    def _get_results(self, frame: cv2.typing.MatLike) -> Any:
        """
        Processes the current frame to detect hand landmarks.
        
        Args:
            frame: The current video frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        if not results.multi_hand_landmarks:
            print("No hands detected")
            return None
        return results
    
    def _read_frame(self) -> cv2.typing.MatLike:
        """
        Reads a frame from the video capture device.
        
        Returns:
            frame: The current video frame.
        """
        ret, frame = self._video_capture.read()
        if not ret:
            print("Failed to grab frame")
            exit()
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)
        return frame

    def __call__(self) -> Any:
        """
        Main loop for the hand gesture controller.
        Continuously captures frames from the camera, detects hand landmarks,
        and performs actions based on the detected gestures.
        """
        while self._video_capture.isOpened():
            frame = self._read_frame()
            results = self._get_results(frame)
            if results is None:
                self._current_mode = ControlMode.DEFAULT
                self._display_frame(frame, "No hands detected", show_image=False)
            elif len(results.multi_hand_landmarks) == 1:
                self._display_frame(frame, "One hand detected", show_image=False)
                index_length = calc_between_landmarks(
                    results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_BOT.value],
                    results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_TOP.value]
                )
                thumb_index_joint_length = calc_between_landmarks(
                    results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_JOINT.value],
                    results.multi_hand_landmarks[0].landmark[HandLandmark.THUMB_TOP.value]
                )
                self._mode_hub(index_length, thumb_index_joint_length, frame)
            else:
                frame = self._exec_mode(frame)
            cv2.imshow(self.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._video_capture.release()
        cv2.destroyAllWindows()