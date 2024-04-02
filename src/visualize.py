#from main import *
import pandas as pd
import matplotlib.pyplot as plt
from joints import Joint

names = tuple(Joint.headers())
#Change path 
training = pd.read_csv(r"C:\Users\ammar\OneDrive\Desktop\MLPR\d0033e_project\data\training.csv", names=names)
hello_gesture = training.head(1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

"""
Given a gesture from a dataset, getJoint() returns the specified "mean" Joint from the dataset.
"""
def getJoint(dataset, joint):
    return [dataset.iloc[0,(joint-1)*3],dataset.iloc[0,(joint-1)*3+1],dataset.iloc[0,(joint-1)*3+2]]
"""
Given a gesture from a dataset, drawHuman() should draw the human figure,
the issue is the data point seems to be in random positions. 
"""
def drawHuman(dataset, ax): #Does not wrok
    A = [19,17,15,13,12,11,2,3,5,7,9]
    B = [20,18,16,14,12]
    C = [10,8,6,4,2]

    for i in range(0,len(A)-1):
        X = getJoint(dataset, A[i])
        Y = getJoint(dataset, A[i+1])
        ax.plot([X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]])
    for i in range(0,len(B)-1):
        X = getJoint(dataset, B[i])
        Y = getJoint(dataset, B[i+1])
        ax.plot([X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]])
    for i in range(0,len(C)-1):
        X = getJoint(dataset, C[i])
        Y = getJoint(dataset, C[i+1])
        ax.plot([X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]])
    head = getJoint(dataset, 2)
    neck = getJoint(dataset, 1)
    ax.plot([head[0], neck[0]], [head[1], neck[1]], [head[2], neck[2]])

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

drawHuman(hello_gesture,ax)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#>>>>>>> 3466a350a586500a1941305a423867ff156b3510
