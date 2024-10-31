import numpy as np


class ChangeOfBasis:
    LANDSCAPE_TO_PORTRAIT = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    PORTRAIT_TO_LANDSCAPE = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    ARKIT_TO_VISION = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    NED_TO_VISION = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    MATRIX_TO_VISION = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]        
    )

    @classmethod
    def convert_matrix_to_vision_convention(cls, pose: np.ndarray) -> np.ndarray:
        """Converts pose from the arkit coordinate system to openCV."""
        change_of_basis = cls.MATRIX_TO_VISION
        return change_of_basis.dot(pose).dot(change_of_basis.T)

    @classmethod
    def convert_arkit_to_vision_convention(cls, pose: np.ndarray) -> np.ndarray:
        """Converts pose from the arkit coordinate system to openCV."""
        change_of_basis = cls.ARKIT_TO_VISION
        return change_of_basis.dot(pose).dot(change_of_basis.T)

    @classmethod
    def convert_landscape_to_portrait(cls, pose: np.ndarray) -> np.ndarray:
        """Converts pose from the landscape orientation to the portrait orientation."""
        change_of_basis = cls.LANDSCAPE_TO_PORTRAIT
        return change_of_basis.dot(pose).dot(change_of_basis.T)

    @classmethod
    def convert_portrait_to_landscape(cls, pose: np.ndarray) -> np.ndarray:
        """Converts pose from the portrait orientation to the landscape orientation."""
        change_of_basis = cls.PORTRAIT_TO_LANDSCAPE
        return change_of_basis.dot(pose).dot(change_of_basis.T)

    @classmethod
    def convert_ned_to_vision_convention(cls, pose: np.ndarray) -> np.ndarray:
        """Converts pose from the NED (north-east-down) coordinate system to openCV."""
        change_of_basis = cls.NED_TO_VISION
        return change_of_basis.dot(pose).dot(change_of_basis.T)
