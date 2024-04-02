<<<<<<< HEAD
# Hej Arvid
=======
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
>>>>>>> 3466a350a586500a1941305a423867ff156b3510
