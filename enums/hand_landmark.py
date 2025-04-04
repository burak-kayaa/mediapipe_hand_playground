from enum import Enum


class HandLandmark(Enum):
    HAND_WRIST = 0
    THUMB_BOT = 2
    THUMB_MID = 3
    THUMB_TOP = 4
    INDEX_BOT = 6
    INDEX_MID = 7
    INDEX_TOP = 8
    MIDDLE_BOT = 10
    MIDDLE_MID = 11
    MIDDLE_TOP = 12
    RING_BOT = 14
    RING_MID = 15
    RING_TOP = 16
    PINKY_BOT = 18
    PINKY_MID = 19
    PINKY_TOP = 20
    THUMB_JOINT = 1
    INDEX_JOINT = 5
    MIDDLE_JOINT = 9
    RING_JOINT = 13
    PINKY_JOINT = 17