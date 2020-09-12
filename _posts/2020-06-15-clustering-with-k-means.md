---
layout: post
---

Clustering analysis is one of several approaches to unsupervised learning. It groups data in such a way that objects in the same group/cluster are similar to each other and objects in different clusters diverge.

There are many algorithms implementing cluster analysis with different ideas behind them, k-Means is one of the most used.

At first I'll try the k-Means algorithm from the `sklearn` Python library. Then I'll write a simple clustering algorithm. After that I will show how high dimensional data may be visualized.

```
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import classification_report
```

## K-Means Clustering
K-Means clusters data by grouping the samples in groups of equal variance by minimizing within-cluter sum-of-squares:

$$
\sum_{i=0}^{n} \min_{\mu_i \in C} \{||x_i - \mu_i||^2\}
$$

The steps of the algorithm are the following:

1. Set initial centroids (starting points)
2. Assign samples to the nearest centroids
3. Take means of all samples assigned to centroids and create new centroids with these values
4. Repeat 2-3 until centroids converge

K-Means always converges[^1], but sometimes to a local minimum. That's why the algorithm is usually ran several times with different initial centroids.

The number of clusters must be specified, so if we don't know how many clusters exist in the data, then the algorithm should be ran with various number of clusters to find the best match.

But the serious advantage is that algorithm is simple and can be easily optimised, which allows to run it even on big datasets.

In this notebook I use dataset with information about wheat seeds. The variables in the dataset include:

* *area* - area of seed
* *perimeter* - perimeter of seed
* *compactness* - compactness of a given seed
* *length* - length of a kernel
* *width* - width of a kernel
* *asymmetry* - the asymmetry coefficient
* *length* - length of kernel groove
* *type* - unique type of seed

```
header = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'length_g', 'type']
seeds = pd.read_csv('seeds_dataset.txt', delimiter='\t+', names=header, engine='python')
```

Using `seeds.info()`, we see that there are no missing values and all columns are numerical (nice!)

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 210 entries, 0 to 209
Data columns (total 8 columns):
area           210 non-null float64
perimeter      210 non-null float64
compactness    210 non-null float64
length         210 non-null float64
width          210 non-null float64
asymmetry      210 non-null float64
length_g       210 non-null float64
type           210 non-null int64
dtypes: float64(7), int64(1)
memory usage: 13.2 KB
```


I know that there are 3 types of unique seeds (210 seeds in total), so I can run `KMeans()` to find 3 clusters of data:

```
km = KMeans(n_clusters=3, n_jobs=-1)
kmeans_pred = km.fit_predict(seeds.drop(['type'], axis=1))
```


### K-Means Clustering with Two Variables
Now let's try clustering for only two variables so that we can visualize it.

I'll cluster `area` and `length`.
<img style="float: center;" src="/assets/images/2cluster.png" width="500" height="250">

Clustering with two variables gives us a **clustering accuracy of 84.29%**. A scatterplot is a good representation of the fact that clustering gives the results very similar to what true classification gives.

In this case it seems that two vaiables are enough to cluster data with a good accuracy.

## Implementation of k-Means

Now I'll implement an algorithm similar to k-Means manually. It is based on Andrew NG's course on Coursera.

```
# I'll use only two variables at first for visualization.
X = np.array(seeds[['area', 'asymmetry']])

# There are 3 clusters and two variables. Set initial centroids with some values.
first_centroids = np.array([[12, 4], [18,5], [19,3]])

# Visualizing the data
def clus_col(X, centroids, preds):
    """
    Function to assign colors to clusters.
    """
    for x in range(centroids[0].shape[0]):
        yield (np.array([X[i] for i in range(X.shape[0]) if preds[i] == x]))

def draw_hist(h, centroids):
    """
    Data for plotting history
    """
    for centroid in centroids:
        yield (centroid[:,h])


def plot_clust(X, centroids, preds=None):
    # Number of colors shoud be equal to the number of clusters,
    # so add more if necessary.
    colors = ['green', 'fuchsia', 'tan']

    # If clusters are defined (preds != None), colors are assigned to clusters.
    clust = [X] if preds is None else list(clus_col(X, centroids, preds))

    # Plot clusters
    fig = plt.figure(figsize=(7, 5))
    for i in range(len(clust)):
        plt.plot(clust[i][:,0], clust[i][:,1], 'o', color=colors[i],
        alpha=0.75, label='Cluster %d'%i)
    plt.xlabel('area')
    plt.ylabel('asymmetry')

    # Plot history of centroids.
    tempx = list(draw_hist(0, centroids))
    tempy = list(draw_hist(1, centroids))

    for x in range(len(tempx[0])):
        plt.plot(tempx, tempy, 'ro--', markersize=6)

    leg = plt.legend(loc=4)
```

```
# Scatterplot with initial centroids.
plot_clust(X,[first_centroids])
```

<img style="float: left;" src="/assets/images/initial.png" width="550" height="320">

Now the algorithm itself. At first, the closest centroids are found for each point in data. Then means of samples assigned to each centroid are calculated and new centroids are created with these values (these steps are repeated). I could define a threshold so that iterations stop when centroids move for lesser distance than threshold level, but even current implementation is good enough.

```
def find_centroids(X, centroids):
    preds = np.zeros((X.shape[0], 1))
    for j in range(preds.shape[0]):

        dist, label = 9999999, 0
        for i in range(centroids.shape[0]):
            distsquared = np.sum(np.square(X[j] - centroids[i]))
            if distsquared < dist:
                dist = distsquared
                label = i

        preds[j] = label

    return preds
```

```
def calc_centroids(X, preds):
    # Calculate new centroids
    for x in range(len(np.unique(preds))):
        yield np.mean((np.array([X[i] for i in range(X.shape[0]) if preds[i] == x])), axis=0)
```

```
def iters(X, first_centroids, K, n_iter):
    centroid_history = []
    current_centroids = first_centroids
    for iter in range(n_iter):
        centroid_history.append(current_centroids)
        preds = find_centroids(X, current_centroids)
        current_centroids = np.array(list(calc_centroids(X, preds)))
    return preds, centroid_history
```

Now to plot:

```
preds, centroid_history = iters(X, first_centroids, 3, 20)
plot_clust(X,centroid_history,preds)
```

<img style="float: left;" src="/assets/images/centroid.png" width="550" height="320">

This is how the process of finding optimal centroids looks like. From initial points centroids move to optimize the loss function. As a result there are three distinguishable clusters.

Now let's try to predict labels using all our variables.

```
first_centroids = np.array([[12, 13, 0.85, 6, 2, 4, 4], [18, 15, 0.9, 6, 3, 5, 5], [19, 14, 0.9, 5.8, 2, 3, 6]])
X = np.array(seeds.drop(['type'], axis=1))

preds, centroid_history = iters(X,first_centroids,K=3,n_iter=20)

# Reshaping into 1-D array.
r = np.reshape(preds, 210, 1).astype(int)

# Labels created by KMeans don't correspond to the original (obviously), so they need to be changed.
for i in range(len(r)):
    if r[i] == 0:
        r[i] = 3

sum(r == seeds.type) / len(seeds.type)
```

The ouput is `0.89047619047619042`, almost the same as when using k-Means from `sklearn`!

### Voronoi Diagram

A Voronoi diagram is a partitioning of a plane into regions based on distances to points in a specific subset of the plane. It can be used as a visualization of clusters in high-dimensional data if combined with principle component analysis (PCA). In machine learning, it is commonly used to project data to a lower dimensional space.

```
reduced_data = PCA(n_components=2).fit_transform(seeds.drop(['type'], axis=1))
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
kmeans.fit(reduced_data)

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Obtain labels for each point in mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Put the result into a color plot
plt.figure(1)
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering with PCA-reduced data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
```

We then have the following Voronoi diagram:

<img style="float: left;" src="/assets/images/voronoi.png" width="550" height="320">

We can see that data in reduced state is visually separable into three clusters. But the graph is 2-D and gives little information about the real state of data.

### High-dimensional Data Visualization

It is also possible to visualize data having more than two dimensions. 3D plot has three dimensions, size, shape and color may represent three more variables. Let's try.

```
# I'll only take 30 samples for better visualization
seeds_little = pd.concat([seeds[50:60],seeds[70:80],seeds[140:150]])

def scatter6d(x,y,z, color, colorsMap='summer'):
    cNorm = matplotlib.colors.Normalize(vmin=min(color), vmax=max(color))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(colorsMap))
    fig = plt.figure()
    ax = Axes3D(fig)
    markers = ['s', 's','o','^']
    for i in seeds.type.unique():
        ax.scatter(x, y, z, c=scalarMap.to_rgba(color), marker=markers[i], s = seeds_little.asymmetry*50 )
    scalarMap.set_array(color)
    fig.colorbar(scalarMap,label='{}'.format('length'))
    plt.show()

scatter6d(seeds_little.area, seeds_little.perimeter, seeds_little.compactness, seeds_little.length)
```

<img style="float: left;" src="/assets/images/3-d.png" width="550" height="320">

Sadly this isn't very representative due to fact that all variables (except `type`) are numerical. If variables used for `size`, `shape` and `color` were categorical with several values, then the graph could be really used for analysis.

So, if you want to get an overview of high-dimensional data, you could take 2 to 3 variables and plot them in 2-D or 3-D. If some variables are categorical, they can be also be used.

This was an overview of K-Means algorithm for data clusterization. It is a general and simple approach which nonetheless works quite well.

You can find all the datasets I used and the code to create this project on my [Github](https://github.com/lukaszamora/K-means-Clustering/) page.

---
{: data-content="footnotes"}

[^1]: k-means always converges by definition. see [proof](https://homepages.warwick.ac.uk/~masfk/Gamma_Convergence_of_k-Means.pdf).