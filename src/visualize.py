# override default environment setting; display plot in new window
# if this raises errors, try `pip install PyQt5`
import matplotlib
matplotlib.use("Qt5Agg")

from main import *
from helpers import *

hello_gesture = training.head(1)
hello_positions = filter_df(hello_gesture, positions=True, means=True)
xs = filter_df(hello_positions, positions=(True, False, False), means=True)
ys = filter_df(hello_positions, positions=(False, True, False), means=True)
zs = filter_df(hello_positions, positions=(False, False, True), means=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs)

for i, j in Joint.connections():
    # adjust offset
    i = (i - 1) * 3
    j = (j - 1) * 3

    x_i, y_i, z_i = hello_positions.iloc[0, i:i + 3].values
    x_j, y_j, z_j = hello_positions.iloc[0, j:j + 3].values
    ax.plot([x_i, x_j], [y_i, y_j], [z_i, z_j])

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

plt.show()
