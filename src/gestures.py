from enum import Enum


class Gesture(Enum):
    AFTERNOON = 1
    BABY = 2
    BIG = 3
    BORN = 4
    BYE = 5
    CALENDAR = 6
    CHILD = 7
    CLOUD = 8
    COME = 9
    DAILY = 10
    DANCE = 11
    DARK = 12
    DAY = 13
    ENJOY = 14
    GO = 15
    HELLO = 16
    HOME = 17
    LOVE = 18
    MY = 19
    NAME = 20
    NO = 21
    RAIN = 22
    SORRY = 23
    STRONG = 24
    STUDY = 25
    THANK_YOU = 26
    WELCOME = 27
    WIND = 28
    YES = 29
    YOU = 30

    def data_name(self) -> str:
        """
        Gets the name as it is formatted in the data.
        That is, lower case with no spaces.
        :return: the appropriately formatted name.
        """
        return self.name.replace('_', '').lower()
