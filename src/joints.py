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
    def headers() -> Iterator[str]:
        """
        Yields appropriate headers for the data.
        The order will be: POS_MEAN, POS_STD, ANGLE_MEAN, ANGLE_STD.
        :return: an iterator over the headers.
        """

        # repeated iteration like this is necessary because of the order.
        for joint in Joint:
            yield f"{joint.name}_POS_X_MEAN"
            yield f"{joint.name}_POS_Y_MEAN"
            yield f"{joint.name}_POS_Z_MEAN"
        for joint in Joint:
            yield f"{joint.name}_POS_X_STD"
            yield f"{joint.name}_POS_Y_STD"
            yield f"{joint.name}_POS_Z_STD"
        for joint in Joint:
            yield f"{joint.name}_ANGLE_1_MEAN"
            yield f"{joint.name}_ANGLE_2_MEAN"
            yield f"{joint.name}_ANGLE_3_MEAN"
        for joint in Joint:
            yield f"{joint.name}_ANGLE_1_STD"
            yield f"{joint.name}_ANGLE_2_STD"
            yield f"{joint.name}_ANGLE_3_STD"

        yield "GESTURE_LABEL"
        yield "GESTURE_ID"
