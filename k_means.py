import matplotlib.pyplot as plt 
import numpy as np

from sklearn.cluster import KMeans

import imageio
import os
 
class Point: 
    def __init__(self, x, y, cluster = -1): 
        self.x = x 
        self.y = y 
        self.cluster = cluster

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
 
def dist(a, b): 
    return np.sqrt((a.x-b.x)**2+(a.y-b.y)**2) 
 
def rand_points(n): 
    points = [] 
    for i in range(n): 
        point = Point(np.random.randint(0, 100), np.random.randint(0, 100)) 
        points.append(point) 
    return points

def three_cluster_points():
    points = []

    for i in range(100):
        points.append(Point(np.random.randint(0, 100), np.random.randint(0, 100)))
        points.append(Point(np.random.randint(300, 400), np.random.randint(0, 100)))
        points.append(Point(np.random.randint(300, 400), np.random.randint(300, 400)))
    return points
 
def centroids(points, k): 
    x_center = np.mean(list(map(lambda p: p.x, points)))
    y_center = np.mean(list(map(lambda p: p.y, points)))
    center = Point(x_center, y_center)
    R = max(map(lambda r: dist(r, center), points))
    centers = []
    for i in range(k):
        x_c = x_center + R * np.cos(2 * np.pi * i / k)
        y_c = y_center + R * np.sin(2 * np.pi * i / k)
        centers.append(Point(x_c, y_c))
    return centers

def new_center(points):
    x_center = np.mean(list(map(lambda p: p.x, points)))
    y_center = np.mean(list(map(lambda p: p.y, points)))
    center = Point(x_center, y_center)
    return center
 
 
def nearest_centroids(points, centroids): 
    for point in points:
        min_dist = dist(point, centroids[0])
        point.cluster = 0
        for i in range(len(centroids)):
            temp = dist(point, centroids[i])
            if temp < min_dist:
                min_dist = temp
                point.cluster = i

# https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/
 
if __name__ == "__main__": 
    n = 500 # кол-во тчк 
    k = 3 # кол-во кластеров 
    points = rand_points(n)
    points = three_cluster_points()
    centers = centroids(points, k)

    # centers = []
    # for i in range(k):
    #     centers.append(points[np.random.randint(0, len(points))])
    # plt.scatter(list(map(lambda p: p.x, points)), list(map(lambda p: p.y, points))) 
    # plt.scatter(list(map(lambda p: p.x, centers)), list(map(lambda p: p.y, centers)), color = 'r') 

    nearest_centroids(points, centers)

    colors = ["r", "g", "black", "blue"]

    iteration = 0

    past_centers = [Point(-1, -1)] * k

    while True:

        all_eq = True

        for i in range(k):
            if centers[i] != past_centers[i]:
                all_eq = False

        if all_eq:
            break

        clusters_points = []

        for i in range(k):
            clusters_points.append([])

        for point in points:
            clusters_points[point.cluster].append(point)

        for i, center in enumerate(centers):
            plt.scatter(center.x, center.y, color=colors[i], linewidths=6, marker='v')
        for cluster in clusters_points:
            plt.scatter(list(map(lambda l: l.x, cluster)), list(map(lambda l: l.y, cluster)), color=colors[cluster[0].cluster], linewidths=3)

        past_centers = centers
        centers = []

        for cluster in clusters_points:
            centers.append(new_center(cluster))
            # centers.append(centroids(cluster, 1)[0])
        
        nearest_centroids(points, centers)
        print(f"{centers[0].x}, {centers[0].y}")
        plt.savefig(f"{iteration}.png")
        iteration += 1
        plt.close()
        # plt.show()

    plt.close()
    for cluster in clusters_points:
        plt.scatter(list(map(lambda l: l.x, cluster)), list(map(lambda l: l.y, cluster)), color=colors[cluster[0].cluster], linewidths=3)
    plt.savefig("res.png")

    print(list(map(lambda x: x.cluster, points)))

    with imageio.get_writer('res.gif', mode='I') as writer:
        for filename in [f'{i}.png' for i in range(iteration)]:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set([f'{i+1}.png' for i in range(iteration-1)]):
        os.remove(filename)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array([list(map(lambda p: p.x, points)), list(map(lambda p: p.y, points))]))

    print(kmeans.labels_)