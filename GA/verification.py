# TSP-GA 算法结果验证

import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt

# 随机生成城市坐标


class cityInit(object):
    def __init__(self, number=4):
        self.site = np.zeros([number, 2])
        s = np.arange(number)
        np.random.shuffle(s)
        self.site[:, 0] = s
        np.random.shuffle(s)
        self.site[:, 1] = s


# 给定一组城市坐标
city0 = np.array([[10,  2],
                  [0,  3],
                  [7,  0],
                  [0, 10],
                  [9,  8],
                  [13,  0]])  # 6
city1 = np.array([[10,  2],
                  [0,  3],
                  [7,  0],
                  [0, 10],
                  [9,  8],
                  [15,  2],
                  [4,  8],
                  [13,  0]])  # 8
city2 = np.array([[6., 5.],
                  [5., 7.],
                  [0., 1.],
                  [2., 0.],
                  [4., 9.],
                  [9., 2.],
                  [1., 8.],
                  [7., 6.],
                  [3., 4.],
                  [8., 3.]])  # 10


def journey(numOfCity):
    # 所有可能路线
    #journeyAll = np.zeros([numOfCity,np.math.factorial(numOfCity)])
    allRank = list(permutations(np.arange(numOfCity).tolist()))
    journeyAll = np.array(allRank).transpose()
    return journeyAll


def distanceMat(city):
    numOfCity = len(city)
    # 初始化任意城市间距离矩阵
    intervalMat = np.zeros([numOfCity, numOfCity])
    for ii in range(numOfCity):
        intervalMat[:, ii] = np.sqrt(
            np.sum(np.square(city[ii, :] - city), 1))
    return intervalMat


def findMaxAndMin(journeyAll, intervalMat):
    # 初始化路程向量
    distance = np.zeros(len(journeyAll[0]))
    for ii in range(len(journeyAll[0])):
        wayRow = journeyAll[:, ii]
        wayCol = np.roll(wayRow, 1)
        distance[ii] = np.sum(intervalMat[wayRow, wayCol])
    # 最长与最短路径
    journeyMax = np.where(distance == np.max(distance))[0]
    journeyMin = np.where(distance == np.min(distance))[0]
    print(distance[journeyMin[0]])
    return journeyMax, journeyMin


def plot(city, journeyAll, journeyMax, journeyMin):
    plt.plot(city[:, 0], city[:, 1], 'ro')  # 散点图
    # for item in journeyMin:
    #     plt.plot(city[journeyAll[:, item], 0],
    #              city[journeyAll[:, item], 1])
    plt.plot(city[journeyAll[:, journeyMin[0]], 0],
             city[journeyAll[:, journeyMin[0]], 1])
    # plt.plot(city[journeyAll[:, journeyMax[0]], 0],
    #               city[journeyAll[:, journeyMax[0]], 1])


def main():
    city = city1
    # distancMat(city)
    journeyAll = journey(len(city))
    journeyMax, journeyMin = findMaxAndMin(
        journeyAll, distanceMat(city))
    plot(city, journeyAll, journeyMax, journeyMin)


if __name__ == "__main__":
    main()
