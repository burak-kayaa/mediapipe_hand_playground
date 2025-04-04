import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from enum import Enum, IntEnum
from typing import Optional, Tuple, Any, Dict, List

# Assuming util.py contains these functions adjusted for MediaPipe landmarks
# If not, you'll need to implement them or adapt existing ones.
# from util import show_landmarks, calc_between_landmarks, draw_landmark, draw_between_landmarks
# --- Placeholder implementations for util functions ---
def calc_landmark_distance(p1, p2) -> float:
    """Calculates the Euclidean distance between two landmarks."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws landmarks and connections (Placeholder). Uses mp_drawing internally."""
    if not detection_result or not detection_result.multi_hand_landmarks:
        return rgb_image

    annotated_image = np.copy(rgb_image)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    for hand_landmarks in detection_result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return annotated_image
# --- End Placeholder implementations ---


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

class OperatingMode(Enum):
    DEFAULT = "Default"
    BRIGHTNESS_CONTROL = "Brightness"
    CONTRAST_CONTROL = "Contrast"

class HandDetection:
    """
    Detects hands using MediaPipe, determines operating mode based on gestures,
    and applies brightness/contrast adjustments to the video feed accordingly.
    """
    WINDOW_NAME: str = "Hand Detection Control"
    # Thresholds for gesture detection (adjust based on testing)
    INDEX_EXTENSION_THRESHOLD: float = 0.12 # Relative length for index finger to be considered extended
    THUMB_TO_INDEX_THRESHOLD: float = 0.08 # Relative distance for thumb tip to index MCP to be considered close

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.6, tracking_confidence: float = 0.5):
        """
        Initializes MediaPipe Hands, OpenCV VideoCapture, and state variables.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Cannot open camera")
            raise IOError("Cannot open camera")

        self.mode: OperatingMode = OperatingMode.DEFAULT
        self.brightness_level: int = 0      # Beta value for cv2.convertScaleAbs [-127, 127] ideally
        self.contrast_level: float = 1.0    # Alpha value for cv2.convertScaleAbs [0.0, 3.0] ideally

        logging.info("HandDetection initialized successfully.")

    def _get_results(self, frame: np.ndarray) -> Optional[Any]:
        """
        Processes the frame with MediaPipe Hands to detect landmarks.

        Args:
            frame: The input video frame (BGR).

        Returns:
            MediaPipe hands processing results, or None if detection fails.
        """
        # Flip the frame horizontally for a later selfie-view display,
        # and convert the BGR image to RGB.
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True # Allow writing again if needed elsewhere
        return results

    def _calculate_distances(self, results: Any) -> Dict[str, Optional[float]]:
        """
        Calculates relevant distances between landmarks based on detected hands.

        Args:
            results: MediaPipe hands processing results.

        Returns:
            A dictionary containing calculated distances:
            - 'num_hands': Number of hands detected.
            - 'index_tip_distance': Distance between index fingertips (if 2 hands).
            - 'hand1_index_extension': Index finger extension length (if >= 1 hand).
            - 'hand1_thumb_index_mcp_distance': Distance between thumb tip and index MCP (if >= 1 hand).
        """
        distances = {
            'num_hands': 0,
            'index_tip_distance': None,
            'hand1_index_extension': None,
            'hand1_thumb_index_mcp_distance': None,
        }

        if not results or not results.multi_hand_landmarks:
            return distances

        landmarks_list = results.multi_hand_landmarks
        num_hands = len(landmarks_list)
        distances['num_hands'] = num_hands

        if num_hands >= 1:
            hand1_lm = landmarks_list[0].landmark
            # Calculate index finger extension (Tip to PIP - better represents bending)
            distances['hand1_index_extension'] = calc_landmark_distance(
                hand1_lm[HandLandmark.INDEX_FINGER_TIP],
                hand1_lm[HandLandmark.INDEX_FINGER_PIP]
            )
            # Calculate distance between thumb tip and index base (MCP)
            distances['hand1_thumb_index_mcp_distance'] = calc_landmark_distance(
                hand1_lm[HandLandmark.THUMB_TIP],
                hand1_lm[HandLandmark.INDEX_FINGER_MCP]
            )

        if num_hands == 2:
            hand1_lm = landmarks_list[0].landmark
            hand2_lm = landmarks_list[1].landmark
            # Calculate distance between index finger tips of both hands
            distances['index_tip_distance'] = calc_landmark_distance(
                hand1_lm[HandLandmark.INDEX_FINGER_TIP],
                hand2_lm[HandLandmark.INDEX_FINGER_TIP]
            )

        return distances

    def _update_mode(self, distances: Dict[str, Optional[float]]) -> None:
        """
        Updates the operating mode based on the number of hands and gestures.

        Args:
            distances: Dictionary of calculated landmark distances.
        """
        num_hands = distances['num_hands']

        if num_hands == 2:
            # Always switch to brightness control if two hands are detected
            if self.mode != OperatingMode.BRIGHTNESS_CONTROL:
                logging.info("Switching to Brightness Control Mode (2 hands detected)")
                self.mode = OperatingMode.BRIGHTNESS_CONTROL
        elif num_hands == 1:
            index_ext = distances['hand1_index_extension']
            thumb_idx_dist = distances['hand1_thumb_index_mcp_distance']

            # Gesture for Contrast: Index finger extended, thumb close to index base
            is_index_extended = index_ext is not None and index_ext > self.INDEX_EXTENSION_THRESHOLD
            is_thumb_close = thumb_idx_dist is not None and thumb_idx_dist < self.THUMB_TO_INDEX_THRESHOLD

            if is_index_extended and is_thumb_close:
                if self.mode != OperatingMode.CONTRAST_CONTROL:
                    logging.info("Switching to Contrast Control Mode (Gesture detected)")
                    self.mode = OperatingMode.CONTRAST_CONTROL
            # If only one hand detected and not making contrast gesture, revert to default
            elif self.mode != OperatingMode.DEFAULT:
                 logging.info("Switching to Default Mode (1 hand, no specific gesture)")
                 self.mode = OperatingMode.DEFAULT

        else: # num_hands == 0
            if self.mode != OperatingMode.DEFAULT:
                logging.info("Switching to Default Mode (No hands detected)")
                self.mode = OperatingMode.DEFAULT

    def _apply_effects(self, frame: np.ndarray, distances: Dict[str, Optional[float]]) -> np.ndarray:
        """
        Applies brightness or contrast adjustments to the frame based on the current mode.

        Args:
            frame: The input video frame (BGR, flipped horizontally).
            distances: Dictionary of calculated landmark distances.

        Returns:
            The modified frame with effects applied.
        """
        if self.mode == OperatingMode.BRIGHTNESS_CONTROL:
            dist = distances.get('index_tip_distance')
            if dist is not None:
                # Map distance (e.g., 0.0 to 0.5) to brightness beta (-100 to 100)
                # Adjust the scaling factor (400) and offset (-100) as needed
                self.brightness_level = int(np.clip(dist * 400 - 100, -127, 127))
                # Apply only brightness change
                return cv2.convertScaleAbs(frame, alpha=self.contrast_level, beta=self.brightness_level)
            else:
                 # Reset brightness if distance couldn't be calculated (shouldn't happen often in 2-hand mode)
                 self.brightness_level = 0

        elif self.mode == OperatingMode.CONTRAST_CONTROL:
            dist = distances.get('hand1_index_extension')
            if dist is not None:
                 # Map index extension (e.g., 0.05 to 0.2) to contrast alpha (0.5 to 2.5)
                 # Adjust scaling (e.g., 10-15) and offset (e.g., 0.3-0.5)
                 self.contrast_level = float(np.clip(dist * 12.0 + 0.3, 0.3, 3.0))
                 # Apply only contrast change
                 return cv2.convertScaleAbs(frame, alpha=self.contrast_level, beta=self.brightness_level)
            else:
                # Reset contrast if distance couldn't be calculated
                self.contrast_level = 1.0

        # In default mode, or if distances were None, apply potentially existing levels
        # (or reset them if you prefer default mode to always be neutral)
        # Optional: Reset levels in default mode
        # self.brightness_level = 0
        # self.contrast_level = 1.0
        return cv2.convertScaleAbs(frame, alpha=self.contrast_level, beta=self.brightness_level)


    def _draw_overlays(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draws hand landmarks, connections, and status information on the frame.

        Args:
            frame: The video frame (BGR, flipped horizontally, effects applied).
            results: MediaPipe hands processing results.

        Returns:
            The frame with overlays drawn.
        """
        annotated_frame = frame.copy()

        # Draw standard landmarks and connections
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

            # Optional: Draw line between index tips in brightness mode
            if self.mode == OperatingMode.BRIGHTNESS_CONTROL and len(results.multi_hand_landmarks) == 2:
                 lm1 = results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_FINGER_TIP]
                 lm2 = results.multi_hand_landmarks[1].landmark[HandLandmark.INDEX_FINGER_TIP]
                 h, w, _ = annotated_frame.shape
                 pt1 = (int(lm1.x * w), int(lm1.y * h))
                 pt2 = (int(lm2.x * w), int(lm2.y * h))
                 cv2.line(annotated_frame, pt1, pt2, (255, 255, 0), 2) # Cyan line

        # Display current mode and levels
        mode_text = f"Mode: {self.mode.value}"
        brightness_text = f"Brightness: {self.brightness_level}"
        contrast_text = f"Contrast: {self.contrast_level:.2f}"

        cv2.putText(annotated_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, brightness_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, contrast_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        return annotated_frame

    def run(self) -> None:
        """
        Starts the main video processing loop.
        """
        logging.info("Starting video stream processing...")
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                logging.warning("Ignoring empty camera frame.")
                continue

            # 1. Get MediaPipe results (includes flipping)
            results = self._get_results(frame)

            # Need the flipped frame for consistency in display and drawing coordinates
            processed_frame = cv2.flip(frame, 1)

            # 2. Calculate relevant distances
            distances = self._calculate_distances(results)

            # 3. Update operating mode based on distances/gestures
            self._update_mode(distances)

            # 4. Apply visual effects based on mode
            processed_frame = self._apply_effects(processed_frame, distances)

            # 5. Draw overlays (landmarks, status text)
            annotated_frame = self._draw_overlays(processed_frame, results)

            # 6. Display the frame
            cv2.imshow(self.WINDOW_NAME, annotated_frame)

            # 7. Exit condition
            if cv2.waitKey(5) & 0xFF == ord('q'):
                logging.info("Quit signal received.")
                break

        # Cleanup
        self.release()
        logging.info("Video stream processing finished.")

    def release(self) -> None:
        """Releases resources."""
        logging.info("Releasing resources...")
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
        logging.info("Resources released.")
