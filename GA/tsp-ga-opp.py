# NP完全问题
# 全排列问题
# 生成排序 ->计算路程 ->评价 ->选择 ->遗传 ->


import numpy as np
import random
from matplotlib import pyplot as plt

# 导入数据集
from data import city3, city4

# 参数设置
city = city4     # 默认城市数据集
pPopu = 0.0001   # 初始化种群个体数占总可能的比例
pAban = 0.25     # 样本总体中舍弃个体比例
pVari = 0.60     # 群体变异比例
pVariSec = 0.10  # 二次变异在变异总体比例比例
pVariUn = 0.20  # 不确定性变异点位数占变异总体比例
pVariSup = 0.40  # 优质个体直接变异产生新个体比例
pResrve = 0.005  # 样本总体中保留个体比例
nMax = 5000     # 最大个体数

# 因变参数
size = len(city)    # 城市数量
nPopu = round(np.math.factorial(size)*pPopu)
if(nMax < nPopu):
    nPopu = nMax        # 样本总数
nAban = np.floor(nPopu*pAban).astype(np.int)    # 舍弃个体总数
nResr = np.floor(nPopu*pResrve).astype(np.int)   # 保留个体总数
nVari = np.floor(nPopu*pVari).astype(np.int)     # 变异个体总数
nVariSec = np.floor(nVari*pVariSec).astype(np.int)  # 二次变异个体总数
nVariSup = (nAban*pVariSup).astype(np.int)  # 优质个体直接变异产生新个体数
# nResr = np.floor(nPopu*pVari).astype(np.int)
nCross = nAban-nVariSup  # 交叉个体总数
nNext = nAban+nVari  # 下一轮调整个数

# 规模数据输出
# print(nPopu, nAban, nResr, nVari)

# 内部数据矩阵、向量初始化
mInterval = np.zeros([size, size])  # 城市间距矩阵
mRoute = np.zeros([size, nPopu]).astype(np.int)  # 路线矩阵
aDistance = np.zeros(nPopu)  # 距离向量

# 内部编号向量初始化
aResr = np.zeros(nResr).astype(np.int)  # 选择保留个体
aAban = np.zeros(nAban).astype(np.int)  # abandon 舍弃个体编号
aVari = np.zeros(nVari).astype(np.int)  # variation 变异个体编号
aVariSup = np.zeros(nVariSup).astype(np.int)  # 直接变成新个体编号
aCross = np.zeros(nCross).astype(np.int)  # Cross 交叉个体编号
aNext = np.zeros(nAban+nVari).astype(np.int)  # 下一轮需要更新距离个体（本轮已调整个体）

# aNext初始化
aNext = np.arange(nPopu)  # 对于初始轮所有个体均需要更新


# 初始化城市间隔矩阵
def intervalInitialization(dataImport=0, input=False):
    global mInterval
    if input:
        mInterval = dataImport
    else:
        for ii in range(size):
            mInterval[:, ii] = np.sqrt(np.sum(np.square(city[ii, :]-city), 1))


# 初始化种群
def populationInitialization():
    global mRoute
    for ii in range(nPopu):
        mRoute[:, ii] = np.random.permutation(size)


# 计算路线路程
def distanceUpdate():
    global aDistance
    for ii in range(len(aNext)):
        way = mRoute[:, aNext[ii]]
        wayRoll1 = np.roll(way, 1)
        aDistance[aNext[ii]] = np.sum(mInterval[way, wayRoll1])


# def roulette():
#     roul = (1/aDistance)/np.sum(1/aDistance)


# 适应度评价
def fitness():
    global aAban, aVari, aCross, aResr, aNext
    rank = aDistance.argsort()
    aAban[:] = np.array(random.sample(
        list(rank[-(1.2*nAban).astype(np.int):]), nAban))
    # print(round(1.5*nResr), nResr, type(nResr))
    aResr[:] = rank[0:nResr]
    rank = rank[~np.isin(rank, aAban)]
    aCross[:] = np.append(rank[0:nCross//2], np.array(
        random.sample(list(rank), nCross//2)))
    rank = rank[~np.isin(rank, aResr)]
    aVari[:] = np.array(random.sample(list(rank), nVari))
    aVariSup[:] = rank[0:nVariSup]
    # aVari[:] = rank[nResr*2:nResr*2+nVari]
    aNext = np.append(aAban, aVari)


# 变异、交叉、遗传
def heredity():
    global mRoute
    # Cross
    for ii in range(nCross//2):
        fGene = np.zeros(size*2).astype(np.int)
        cGene0 = list()
        cGene1 = list()
        for ij in range(size):
            fGene[[ij*2, ij*2+1]] = [mRoute[:, aCross[ii]][ij],
                                     mRoute[:, aCross[ii+nCross//2]][ij]]
        for item in fGene:
            if not item in cGene0:
                cGene0.append(item)
            else:
                cGene1.append(item)
        mRoute[:, aAban[ii*2]] = cGene0
        mRoute[:, aAban[ii*2+1]] = cGene1
    # Variation
    # 变异位置
    elect = np.random.randint(0, size, round(len(aVari)*2))
    # 优质个体直接变异产生新个体
    for ii in range(nVariSup):
        mRoute[:, aAban[ii+nCross]] = mRoute[:, aVariSup[ii]]
        (mRoute[:, aAban[ii+nCross]])[elect[[ii*2, ii*2+1]]
                                      ] = mRoute[:, aAban[ii+nCross]][elect[[ii*2+1, ii*2]]]
    # 一次变异
    for ii in range(len(aVari)):
        (mRoute[:, aVari[ii]])[elect[[ii*2, ii*2+1]]
                               ] = (mRoute[:, aVari[ii]])[elect[[ii*2+1, ii*2]]]
    # 二次变异
    for ii in range(nVariSec):
        (mRoute[:, aVari[ii*2]])[elect[[ii, ii*3]]
                                 ] = (mRoute[:, aVari[ii*2]])[elect[[ii*3, ii]]]


# plot
def plot():
    plt.plot(city[:, 0], city[:, 1], 'ro')
    indexMin = aDistance.argmin()
    index = mRoute[:, indexMin]
    plt.plot(city[np.append(index, index[0]), 0],
             city[np.append(index, index[0]), 1])


def main():
    populationInitialization()
    intervalInitialization()
    distanceUpdate()
    flage = True
    count = 0
    print("{g:<10}{cp:<26}{jc:<26}{avg:<25}".format(g='gene',
                                                    cp='Current optimal', jc='Judgment condition', avg='Population mean'))
    while(flage):
        fitness()
        heredity()
        distanceUpdate()
        good = aDistance[aDistance.argsort()]
        jc = np.max(good[0:min(15, nResr)])
        avg = np.average(aDistance)
        cp = np.min(aDistance)
        count += 1
        print("{0:<9}".format(count), "{0:<25}".format(cp),
              "{0:<25}".format(jc), "{0:<25}".format(avg))
        if(abs(cp-jc) < 0.01):
            flage = False
    plot()
    print(mRoute[:, aDistance.argmin()])


if __name__ == "__main__":
    main()
