# NP完全问题
# 全排列问题
# 生成排序 ->计算路程 ->评价 ->选择 ->遗传 ->


import numpy as np
from matplotlib import pyplot as plt
from data import city3, city4


# 随机生成城市坐标
class cityInit(object):
    def __init__(self, number=4):
        self.site = np.random.randint(1, 10, [number, 2])


# TSP求解类
class tspGA(object):

    def __init__(self, city=city3):
        self._city = city
        self._size = len(city)
        self.intervalMat = np.zeros([self._size, self._size])
        # 初始化距离矩阵
        for ii in range(self._size):
            self.intervalMat[:, ii] = np.sqrt(
                np.sum(np.square(city[ii, :] - city), 1))

    # 参数控制
    def parameter(self):
        self._pPopu = 0.0001     # 初始化种群个体数占总可能的比例
        self._pAban = 0.20      # 样本总体中舍弃个体比例
        self._pVari = 0.60      # 群体变异比例
        self._pResrve = 0.001    # 样本总体中保留个体比例
        self._nMax = 10000      # 最大个体数

        # 内部跟随变化参数
        self._nPopu = round(np.math.factorial(self._size)*self._pPopu)
        if(self._nMax < self._nPopu):
            self._nPopu = self._nMax
        self._nAban = np.floor(self._nPopu*self._pAban).astype(np.int)
        self._nResr = round(np.floor(self._nPopu*self._pResrve)).astype(np.int)
        self._nVari = round(np.floor(self._nPopu*self._pVari)).astype(np.int)
        print(self._nPopu, self._nAban, self._nResr, self._nVari)

    # 初始化种群
    def populationInit(self):
        # 以乱序向量代表路线
        self.popul = np.zeros([self._size, self._nPopu]).astype(np.int)
        for ii in range(self._nPopu):
            self.popul[:, ii] = np.random.permutation(self._size)

    # 计算路线路程
    def distanceInit(self):
        self._dist = np.zeros(self._nPopu)
        for ii in range(self._nPopu):
            self._planX = self.popul[:, ii]
            self._planY = np.roll(self._planX, 1)
            # print(type(self._planX))
            self._dist[ii] = np.sum(self.intervalMat[self._planX, self._planY])

    # 路程更新(只更新变化的部分)
    def distanceUpdate(self):
        for ii in range(len(self._next)):
            self._planX = self.popul[:, self._next[ii]]
            self._planY = np.roll(self._planX, 1)
            self._dist[self._next[ii]] = np.sum(
                self.intervalMat[self._planX, self._planY])

    # 适应度评价
    def fitness(self):
        # 按路程排序的路线编号
        self._rank = self._dist.argsort()
        # 保留个体
        self._resrve = self._rank[0:self._nResr]
        # 舍弃个体
        self._aban = self._rank[-self._nAban-1:]  # 舍弃个体编号
        # 变异个体
        self._vari = self._rank[self._nResr:self._nResr+self._nVari]  # 变异个体编号
        # 交叉个体
        self._cross = self._rank[0:self._nAban]  # 交叉个体数应与舍弃个体数相同
        # 下一轮所有变化个体
        self._next = self._rank[~np.isin(self._rank, self._resrve)]

    # 变异、交叉、遗传
    def heredity(self):
        # 交叉
        # print(len(self._next), len(self._aban))
        for ii in range(self._nAban//2):
            # 生成子代基因
            self._fGene = np.zeros(self._size*2).astype(np.int)
            self._cGene0 = list()
            self._cGene1 = list()
            # 待交叉父代基因
            for ij in range(self._size):
                self._fGene[[ij*2, ij*2+1]] = [self.popul[:,
                                                          self._cross[ii*2]][ij], self.popul[:, self._cross[ii*2+1]][ij]]
            # 检测点
            # print(self.popul[:, self._cross[ii*2]],
            #       self.popul[:, self._cross[ii*2+1]])
            # print(type(self._fGene), self._fGene)
            # 交替位置交叉法
            for item in self._fGene:
                if not item in self._cGene0:
                    self._cGene0.append(item)
                else:
                    self._cGene1.append(item)
            # 子代替代已舍弃个体
            # print(type(self._cGene0), self._cGene0,
            #       type(self._cGene1), self._cGene1)
            self.popul[:, self._aban[ii*2]] = self._cGene0
            self.popul[:, self._aban[ii*2+1]] = self._cGene1

        # 变异
        elect = np.random.randint(0, self._size, len(self._vari)*2)
        # print(len(elect), elect)
        for ii in range(len(self._vari)):
            self.popul[:, self._vari[ii]][elect[[ii*2, ii*2+1]]
                                          ] = self.popul[:, self._vari[ii]][elect[[ii*2+1, ii*2]]]

    def plot(self, index):
        plt.plot(self._city[:, 0], self._city[:, 1], 'ro')  # 散点图
        index = self.popul[:, index]
        plt.plot(self._city[np.append(index, index[0]), 0],
                 self._city[np.append(index, index[0]), 1])


def main():
    tsp = tspGA(city4)
    tsp.parameter()
    tsp.populationInit()
    tsp.distanceInit()
    flage = True
    while(flage):
        tsp.fitness()
        tsp.heredity()
        tsp.distanceUpdate()
        good10 = tsp._dist[tsp._dist.argsort()]
        # gAvg = np.average(good10[0:tsp._nResr])
        gMax = np.max(good10[0:round(tsp._nResr*10)])
        dmin = np.min(tsp._dist)
        avg = np.average(tsp._dist)
        print(dmin, gMax, avg)
        if(abs(dmin-gMax) < 0.001):
            flage = False
    index = tsp._dist.argmin()
    print(tsp.popul[:, index])
    tsp.plot(index)


if __name__ == "__main__":
    main()
