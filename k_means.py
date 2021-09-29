import matplotlib.pyplot as plt 
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

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

def get_r_points():
    points = []
    X, y_true = make_blobs(n_samples=300, centers=5,
                       cluster_std=0.50, random_state=0)

    for xy in X:
        points.append(Point(xy[0], xy[1]))

    return points, y_true
 
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

def clasters_calculate(points, k):

    centers = centroids(points, k)
    nearest_centroids(points, centers)

    iteration = 0
    past_centers = [Point(-1, -1)] * k

    while True:

        colors = []

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

        for cluster in clusters_points:
            p = plt.scatter(list(map(lambda l: l.x, cluster)), list(map(lambda l: l.y, cluster)), linewidths=3)
            colors.append(p.get_facecolor())
        for i, center in enumerate(centers):
            plt.scatter(center.x, center.y, linewidths=6, marker='v', color=colors[i])
        past_centers = centers
        centers = []

        for cluster in clusters_points:
            centers.append(new_center(cluster))
        
        nearest_centroids(points, centers)
        # print(f"{centers[0].x}, {centers[0].y}")
        plt.savefig(f"{iteration}_{k}.png")
        iteration += 1
        plt.close()

    plt.close()
    for cluster in clusters_points:
        plt.scatter(list(map(lambda l: l.x, cluster)), list(map(lambda l: l.y, cluster)), linewidths=3)
    plt.savefig(f"res_{k}.png")

    # print(list(map(lambda x: x.cluster, points)))
    # print(y_true)

    s = 0
    for i, cluster in enumerate(clusters_points):
        cluster_center = new_center(cluster)
        for j, point in enumerate(cluster):
            s += dist(cluster_center, point)

    with imageio.get_writer(f'res_{k}.gif', mode='I') as writer:
        for filename in [f'{i}_{k}.png' for i in range(iteration)]:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set([f'{i+1}_{k}.png' for i in range(iteration-1)]):
        os.remove(filename)

    return s

if __name__ == "__main__": 
    n = 500 # кол-во точек
    k = 4 # кол-во кластеров 
    points = rand_points(n)
    points, y_true = get_r_points()

    # for i in range(1, 7):
    #     print(clasters_calculate(points, i))

    past_r = clasters_calculate(points, 1)
    rs = [past_r]
    diffs = []
    i = 2
    while True:
        r = clasters_calculate(points, i)
        print(past_r)
        print(r)
        print(past_r - r)
        print("--------------")
        diffs.append(past_r - r)
        rs.append(r)
        if past_r - r < 20:
            break
        past_r = r
        i += 1

    print(f"Оптимальное количество кластеров = {i - 1}")

    # print(diffs)

    plt.close()
    plt.plot(range(len(rs)), rs)
    plt.scatter(range(len(rs)), rs)
    plt.xlabel("Кол-во кластеров")
    plt.ylabel("Изменение расстояния")
    plt.show()