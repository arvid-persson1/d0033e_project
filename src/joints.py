from enum import Enum
from typing import Iterator


class Joint(Enum):
    CENTER_HIP = 1
    SPINE = 2
    CENTER_SHOULDER = 3
    HEAD = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    LEFT_HAND = 8
    RIGHT_SHOULDER = 9
    RIGHT_ELBOW = 10
    RIGHT_WRIST = 11
    RIGHT_HAND = 12
    LEFT_HIP = 13
    LEFT_KNEE = 14
    LEFT_ANKLE = 15
    LEFT_FOOT = 16
    RIGHT_HIP = 17
    RIGHT_KNEE = 18
    RIGHT_ANKLE = 19
    RIGHT_FOOT = 20

    @staticmethod
    def headers(
            positions: bool = True,
            angles: bool = True,
            means: bool = True,
            deviations: bool = True
    ) -> Iterator[str]:

        if positions and means:
            for joint in Joint:
                yield f"{joint.name}_POS_X_MEAN"
                yield f"{joint.name}_POS_Y_MEAN"
                yield f"{joint.name}_POS_Z_MEAN"
        if positions and deviations:
            for joint in Joint:
                yield f"{joint.name}_POS_X_STD"
                yield f"{joint.name}_POS_Y_STD"
                yield f"{joint.name}_POS_Z_STD"
        if angles and means:
            for joint in Joint:
                yield f"{joint.name}_ANGLE_1_MEAN"
                yield f"{joint.name}_ANGLE_2_MEAN"
                yield f"{joint.name}_ANGLE_3_MEAN"
        if angles and deviations:
            for joint in Joint:
                yield f"{joint.name}_ANGLE_1_STD"
                yield f"{joint.name}_ANGLE_2_STD"
                yield f"{joint.name}_ANGLE_3_STD"
