from enum import Enum
from typing import Iterator


class Joint(Enum):
    HEAD = 1
    CENTER_SHOULDER = 2
    LEFT_SHOULDER = 3
    RIGHT_SHOULDER = 4
    LEFT_ELBOW = 5
    RIGHT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_WRIST = 8
    LEFT_HAND = 9
    RIGHT_HAND = 10
    SPINE = 11
    CENTER_HIP = 12
    LEFT_HIP = 13
    RIGHT_HIP = 14
    LEFT_KNEE = 15
    RIGHT_KNEE = 16
    LEFT_ANKLE = 17
    RIGHT_ANKLE = 18
    LEFT_FOOT = 19
    RIGHT_FOOT = 20

    def number(self):
        """
        Gets the number associated with this joint in the data.
        This is an alias for `self.value`
        :return: the number associated with the joint.
        """

        return self.value

    def image_number(self):
        """
        Gets the number associated with this joint in the image.
        :return: the number associated with the joint.
        """

        # Python somehow thinks this is calling get on Joint directly.
        # The line below disables this warning in PyCharm.
        # noinspection PyUnresolvedReferences
        return Joint.__image_numbers.get(self.value)

    __image_numbers = {
        CENTER_HIP: 1,
        SPINE: 2,
        CENTER_SHOULDER: 3,
        HEAD: 4,
        LEFT_SHOULDER: 5,
        LEFT_ELBOW: 6,
        LEFT_WRIST: 7,
        LEFT_HAND: 8,
        RIGHT_SHOULDER: 9,
        RIGHT_ELBOW: 10,
        RIGHT_WRIST: 11,
        RIGHT_HAND: 12,
        LEFT_HIP: 13,
        LEFT_KNEE: 14,
        LEFT_ANKLE: 15,
        LEFT_FOOT: 16,
        RIGHT_HIP: 17,
        RIGHT_KNEE: 18,
        RIGHT_ANKLE: 19,
        RIGHT_FOOT: 20
    }

    @staticmethod
    def headers(include_candidate: bool = False) -> Iterator[str]:
        """
        Yields appropriate headers for the data.
        The order will be: POS_MEAN, POS_STD, ANGLE_MEAN, ANGLE_STD.
        :param include_candidate: whether or not the candidate column should be included.
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
        if include_candidate:
            yield "CANDIDATE"

    @staticmethod
    def connections() -> list[tuple[int, int]]:
        # this is not the same as in the image since the joint numbers aren't the same
        return [(12, 11), (12, 13), (12, 14), (11, 2), (2, 1), (2, 3), (2, 4), (3, 5), (5, 7), (7, 9),
                (4, 6), (6, 8), (8, 10), (13, 15), (15, 17), (17, 19), (14, 16), (16, 18), (18, 20)]
