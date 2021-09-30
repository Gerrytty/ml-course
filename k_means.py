import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs

import imageio
import os


class Point:
    def __init__(self, x, y, cluster=-1):
        self.x = x
        self.y = y
        self.cluster = cluster
        self.bi = 0
        self.ai = 0

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def rand_points(n):
    points = []
    for i in range(n):
        point = Point(np.random.randint(0, 100), np.random.randint(0, 100))
        points.append(point)
    return points


def get_r_points(n, k):
    points = []
    X, y_true = make_blobs(n_samples=n, centers=k,
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
    if points:
        x_center = np.mean(list(map(lambda p: p.x, points)))
        y_center = np.mean(list(map(lambda p: p.y, points)))
        center = Point(x_center, y_center)
    else:
        center = Point(0, 0)

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


def clusters_calculate(points, k, plot=False):
    centers = centroids(points, k)
    nearest_centroids(points, centers)

    iteration = 0
    past_centers = [Point(-1, -1)] * k

    while True:

        colors = []

        if all(centers[i] == past_centers[i] for i in range(k)):
            break

        clusters_points = [[] for _ in range(k)]

        for point in points:
            clusters_points[point.cluster].append(point)

        if plot:
            for cluster in clusters_points:
                p = plt.scatter(list(map(lambda l: l.x, cluster)), list(map(lambda l: l.y, cluster)), linewidths=3)
                colors.append(p.get_facecolor())
            for i, center in enumerate(centers):
                plt.scatter(center.x, center.y, linewidths=6, marker='v', color=colors[i])

        past_centers = centers
        centers = list(map(new_center, clusters_points))

        nearest_centroids(points, centers)
        if plot:
            print(f"{centers[0].x}, {centers[0].y}")
            plt.savefig(f"{iteration}_{k}.png")
            plt.close()
        iteration += 1

    if plot:
        plt.close()
        for cluster in clusters_points:
            plt.scatter(list(map(lambda l: l.x, cluster)), list(map(lambda l: l.y, cluster)), linewidths=3)
        plt.savefig(f"res_{k}.png")
        plt.close()

    s = 0  # intra-cluster distance
    for i, cluster in enumerate(clusters_points):
        cluster_center = new_center(cluster)
        for j, point in enumerate(cluster):
            s += dist(cluster_center, point)

    for c1, cluster1 in enumerate(clusters_points):
        for p1, point1 in enumerate(cluster1):
            sums = 0
            for p2, point2 in enumerate(cluster1):
                if p1 != p2:
                    sums += dist(point1, point2)
            sums *= (1 / (len(cluster1) - 1))

            point1.ai = sums

    for c1, cluster1 in enumerate(clusters_points):
        if cluster1:
            for p1, point1 in enumerate(cluster1):
                dists_point_to_another_clusters = []
                for c2, cluster2 in enumerate(clusters_points):
                    if c1 != c2 and cluster2:
                        dists_for_one_cluster = []
                        for point2 in cluster2:
                            dists_for_one_cluster.append(dist(point1, point2))
                        mean_dist_to_one_cluster = np.mean(dists_for_one_cluster)
                        dists_point_to_another_clusters.append(mean_dist_to_one_cluster)
                if dists_point_to_another_clusters:
                    bi = min(dists_point_to_another_clusters)
                    point1.bi = bi

    if plot:
        with imageio.get_writer(f'res_{k}.gif', mode='I') as writer:
            for filename in [f'{i}_{k}.png' for i in range(iteration)]:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in set([f'{i + 1}_{k}.png' for i in range(iteration - 1)]):
            os.remove(filename)

    return s, np.mean(list(map(lambda point: (point.bi - point.ai) / max(point.ai, point.bi), points)))


def plot_k(arr, title):
    plt.figure()
    plt.title(title)
    plt.plot(arr)
    plt.scatter(range(len(arr)), arr)


def get_optimal_clusters_number(points):
    distances = []
    sis = []
    i = 1
    while i <= 10:
        print(i)
        distance, si = clusters_calculate(points, i)
        sis.append(si)
        distances.append(distance)
        i += 1

    # Silhouette method
    print(f"Оптимальное число кластеров = {np.argmax(sis) + 1}")

    plot_k(distances, "Расстояния от точек до центроидов")
    plot_k(np.diff(distances), "Производная расстояний")
    plot_k(sis, "Silhouette")
    plt.show()

    return np.argmax(sis) + 2


if __name__ == "__main__":
    n = 300  # кол-во точек
    k = 4  # кол-во кластеров

    # Беру рандомные точки, распределенные так чтобы было k кластеров
    # из sklearn чтобы потом визуально было видно что алгоритм работает
    points, y_true = get_r_points(n, k)

    get_optimal_clusters_number(points)

    # чтобы посмотреть визуализацию алгоритма нужно запустить следующую функцию
    # clusters_calculate(points, k, plot=True)