from sklearn.decomposition import PCA
from data import training_features_scaled


def main():
    df = training_features_scaled()

    print("n\tTotal variance\t\tVariances")

    for i in range(1, 241):
        pca = PCA(n_components=i)
        pca.fit_transform(df)
        var = pca.explained_variance_ratio_
        print(f"{str(i).ljust(3)}\t{str(sum(var)).ljust(18)}\t{", ".join(str(f) for f in var)}")


if __name__ == "__main__":
    main()
