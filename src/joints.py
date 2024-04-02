from enum import Enum
from typing import Iterator


class Joint(Enum):
    # ordered by number in image, numbered by number in data
    CENTER_HIP = 12
    SPINE = 11
    CENTER_SHOULDER = 2
    HEAD = 1
    LEFT_SHOULDER = 3
    LEFT_ELBOW = 5
    LEFT_WRIST = 7
    LEFT_HAND = 9
    RIGHT_SHOULDER = 4
    RIGHT_ELBOW = 6
    RIGHT_WRIST = 8
    RIGHT_HAND = 10
    LEFT_HIP = 13
    LEFT_KNEE = 15
    LEFT_ANKLE = 17
    LEFT_FOOT = 19
    RIGHT_HIP = 14
    RIGHT_KNEE = 16
    RIGHT_ANKLE = 18
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

    @staticmethod
    def connections() -> list[tuple[int, int]]:
        # not the same as in the image since the joint numbers aren't the same
        return [(12, 11), (12, 13), (12, 14), (11, 2), (2, 1), (2, 3), (2, 4), (3, 5), (5, 7), (7, 9),
                (4, 6), (6, 8), (8, 10), (13, 15), (15, 17), (17, 19), (14, 16), (16, 18), (18, 20)]
