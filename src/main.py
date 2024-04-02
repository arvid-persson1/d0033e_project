import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

training = pd.read_csv("../data/training.csv")

print(training.head(5))

print(training.describe())