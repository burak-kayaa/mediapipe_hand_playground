from enum import Enum


class ControlMode(Enum):
    """Enum representing the current control mode."""
    DEFAULT = "Default"
    BRIGHTNESS = "Brightness Control"
    CONTRAST = "Contrast Control"