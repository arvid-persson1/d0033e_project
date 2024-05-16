from numpy import array, float64, cross, sin, cos, arccos, dot, append
from numpy.linalg import norm as mag
from pandas import read_csv, Series, DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from joints import Joint

# Least number of components giving 95% accuracy.
N_COMPONENTS = 88


def main():
    training = read_csv("../data/training_filled.csv", names=tuple(Joint.headers()))
    testing = read_csv("../data/testing_filled.csv", names=tuple(Joint.headers()))

    training = training.apply(__process, axis=1)
    testing = testing.apply(__process, axis=1)

    training.to_csv("../data/training_processed.csv", index=False)
    testing.to_csv("../data/testing_processed.csv", index=False)

    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training.iloc[:, 0:240])
    training.iloc[:, 0:240] = training_scaled
    testing_scaled = scaler.transform(testing.iloc[:, 0:240])
    testing.iloc[:, 0:240] = testing_scaled

    training.to_csv("../data/training_scaled.csv", index=False)
    testing.to_csv("../data/testing_scaled.csv", index=False)

    pca = PCA(N_COMPONENTS)
    columns = tuple(f"PC{i}" for i in range(N_COMPONENTS))
    training_pc = pca.fit_transform(training_scaled)
    training_pc = DataFrame(training_pc, columns=columns)
    testing_pc = pca.transform(testing_scaled)
    testing_pc = DataFrame(testing_pc, columns=columns)

    training_pc.to_csv("../data/training_pc.csv", index=False)
    testing_pc.to_csv("../data/testing_pc.csv", index=False)


def __process(row: Series) -> Series:
    cs = array(row.iloc[30:30 + 3], dtype=float64)
    ls = array(row.iloc[6:6 + 3], dtype=float64)
    rs = array(row.iloc[9:9 + 3], dtype=float64)

    # Normalize rotations

    cl = ls - cs
    cr = rs - cs
    # Python thinks this code is unreachable.
    # noinspection PyUnreachableCode
    n = 1 / (mag(cl) * mag(cr)) * cross(cl, cr)
    k = (0, 0, 1)

    # mag(k) is known to be 1
    theta = arccos(dot(n, k) / mag(n))

    # Numpy likely has built-in optimized methods to these transformations that should be used instead.
    st = sin(theta)
    ct = cos(theta)
    rotation = array([
        [ct, 0, st, 0],
        [0, 1, 0, 0],
        [-st, 0, ct, 0],
        [0, 0, 0, 1]
    ])

    for i in range(len(Joint)):
        row.iloc[i * 3: i * 3 + 3] = dot(rotation, append(row.iloc[i * 3: i * 3 + 3], 1))[:3]

    # Normalize positions

    row[0:60:3] -= cs[0]
    row[1:60:3] -= cs[1]
    row[2:60:3] -= cs[2]

    return row


if __name__ == "__main__":
    main()
