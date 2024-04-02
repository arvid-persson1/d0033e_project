import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joints import Joint

names = [
    *Joint.headers(),
    "GESTURE_LABEL",
    "GESTURE_ID"
]

training = pd.read_csv("../data/training.csv", names=names)

# print(training.head(5))
print(training.describe())
