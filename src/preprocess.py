from sys import argv, exit

from numpy import array, float64, cross, sin, cos, arccos, dot, append
from numpy.linalg import norm as mag
from pandas import read_csv, Series

from joints import Joint


def main():
    try:
        _, src, dest = argv
    except ValueError:
        print("Usage: py preprocess.py [src path] [dest path]")
        exit(1)

    df = read_csv(src, names=tuple(Joint.headers()))
    df = df.apply(__process, axis=1)
    df.to_csv(dest, index=False)


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

    # Numpy likely has built-in optimized methods to these transformations.
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
