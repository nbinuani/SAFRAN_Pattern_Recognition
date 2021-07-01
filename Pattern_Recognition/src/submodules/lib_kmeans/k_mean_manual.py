from sklearn.cluster import KMeans
import numpy as np

def _color_generation(labels, colorsbar):
    color_label = []
    for i in range(len(labels)):
        color_label.append(colorsbar[labels[i]])
    return color_label


def kmean_clustering_manual(df, colorsbar, nb_cluster):
    X = np.array(df)
    # Generate kmean model with k clusters
    model = KMeans(n_clusters=nb_cluster)

    # Fit the model
    model.fit(X)

    # Retrieve the labels
    labels = model.predict(X)

    # Compute the colors
    labels = _color_generation(labels, colorsbar)

    # Add the label to the dataframe
    df['label'] = labels

    return df
