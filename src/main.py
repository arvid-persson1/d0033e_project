#<<<<<<< HEAD
# Hej Arvid
#=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joints import Joint

names = tuple(Joint.headers(labels=True))

training = pd.read_csv(r"C:\Users\ammar\OneDrive\Desktop\MLPR\d0033e_project\data\training.csv", names=names)
hello_gesture = training.head(1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

"""
For loop is going through the first 20 data enteries
which should represents the "mean positions" of the 20 diffrent joints
according to the overleaf document.
"""

for i in range(0,60,3):     
    xs = hello_gesture.iloc[0,i]
    ys = hello_gesture.iloc[0,i+1]
    zs = hello_gesture.iloc[0,i+2]
    ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#print(training.describe())
#>>>>>>> 3466a350a586500a1941305a423867ff156b3510
