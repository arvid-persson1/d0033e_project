from enum import Enum
from typing import Iterator, Optional


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

    @staticmethod
    def headers(include_candidate: bool = False) -> Iterator[str]:
        """
        Yields appropriate headers for the data.
        The order will be: POS_MEAN, POS_STD, ANGLE_MEAN, ANGLE_STD.
        :param include_candidate: whether or not the candidate column should be included.
        :return: an iterator over the headers.
        """

        # Repeated iteration like this are necessary because of the order.
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
        # This is not the same as in the image since the joint numbers aren't the same.
        return [(12, 11), (12, 13), (12, 14), (11, 2), (2, 1), (2, 3), (2, 4), (3, 5), (5, 7), (7, 9),
                (4, 6), (6, 8), (8, 10), (13, 15), (15, 17), (17, 19), (14, 16), (16, 18), (18, 20)]


__gesture_names = {
    1: "Afternoon",
    2: "Baby",
    3: "Big",
    4: "Born",
    5: "Bye",
    6: "Calendar",
    7: "Child",
    8: "Cloud",
    9: "Come",
    10: "Daily",
    11: "Dance",
    12: "Dark",
    13: "Day",
    14: "Enjoy",
    15: "Go",
    16: "Hello",
    17: "Home",
    18: "Love",
    19: "My",
    20: "Name",
    21: "No",
    22: "Rain",
    23: "Sorry",
    24: "Strong",
    25: "Study",
    26: "Thank you",
    27: "Welcome",
    28: "Wind",
    29: "Yes",
    30: "You"
}


def gesture_name(gesture_id: int) -> Optional[str]:
    return __gesture_names.get(gesture_id)
