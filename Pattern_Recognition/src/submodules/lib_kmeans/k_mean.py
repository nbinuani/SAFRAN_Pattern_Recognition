from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def _color_generation(labels, colorsbar):
    color_label = []
    for i in range(len(labels)):
        color_label.append(colorsbar[labels[i]])
    return color_label


def kmean_clustering(df, colorsbar):
    X = np.array(df)
    result = []
    for k in range(2, 10):
        # Generate kmean model with k clusters
        model = KMeans(n_clusters=k)

        # Fit the model
        model.fit(X)

        # Retrieve the labels
        labels = model.predict(X)

        # Compute the score
        score = silhouette_score(X, labels)

        # Append the result
        result.append(dict(nb_class=k, score=score))

    # Get the cluster number of the maximum score
    nb_class = 0
    score = 0
    for i in range(len(result)):
        if result[i]['score'] > score:
            nb_class = result[i]['nb_class']
            score = result[i]['score']
        else:
            continue

    # Re compute the KMean for the optimum k number of clusters
    model = KMeans(n_clusters=nb_class)
    model.fit(X)
    labels = model.predict(X)

    # Compute the colors
    labels = _color_generation(labels, colorsbar)

    # Add the label to the dataframe
    df['label'] = labels

    return df




