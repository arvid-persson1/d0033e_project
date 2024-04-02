import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

from joints import Joint

names = tuple(Joint.headers(labels=True))

training = pd.read_csv(r"C:\Users\ammar\OneDrive\Desktop\MLPR\d0033e_project\data\training.csv", names=names)

#print(training.describe())
