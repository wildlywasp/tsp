# NP完全问题
# 全排列问题
# 生成排序 ->计算路程 ->评价 ->选择 ->遗传 ->


import numpy as np
from numpy import sum as npsum
from numpy import append as npappend
from numpy.random import randint as nprandint
import random
from matplotlib import pyplot as plt
from numba import jit
import time
import json
# 导入数据集
from data import city2, city3, city4, city5, city6


# 参数设置
city = city4     # 默认城市数据集
pPopu = 0.0001   # 初始化种群个体数占总可能的比例
pAban = 0.45     # 样本总体中舍弃个体比例
pVari = 0.4     # 群体变异比例
pCloneWithVari = 0.05  # 优质个体直接变异产生新个体比例
pResrve = 0.01  # 样本总体中保留个体比例
nMax = 3000     # 最大个体数


# 样本种群数量
size = len(city)    # 城市数量
nPopu = round(np.math.factorial(size)*pPopu)
if(nMax < nPopu):
    nPopu = nMax        # 样本总数


# 内部数据矩阵、向量初始化 (预分配空间不足时修改此处)
mInterval = np.zeros([size, size])  # 城市间距矩阵
mRoute = np.zeros([size, nPopu]).astype(np.int)  # 路线矩阵
# mCross = np.zeros([size, round(nPopu*0.5)]).astype(np.int)  # 待交叉备份矩阵
mVari = np.zeros([size, round(nPopu*0.5)]).astype(np.int)  # 待变异备份矩阵
mConbine = np.zeros([size*2, round(nPopu*0.5)]).astype(np.int)  # 交叉准备矩阵
aDistance = np.zeros(nPopu)  # 距离向量
aResr = np.zeros(round(nPopu*0.3)).astype(np.int)  # 选择保留个体
aAban = np.zeros(round(nPopu*0.5)).astype(np.int)  # abandon 舍弃个体编号
aVari = np.zeros(round(nPopu*0.7)).astype(np.int)  # variation 变异个体编号
aCloneWithVari = np.zeros(round(nPopu*0.2)).astype(np.int)  # 直接变成新个体编号
aCross = np.zeros(round(nPopu*0.5)).astype(np.int)  # Cross 交叉个体编号
aNext = np.zeros(nPopu).astype(np.int)  # 下一轮需要更新距离个体（本轮已调整个体）
# aNext初始化
aNext = np.arange(nPopu)  # 对于初始轮所有个体均需要更新


# 规模初始化
def scaleInitialization():
    global nAban, nResr, nCross, nVari, nCloneWithVari  # , nNext
    nAban = np.floor(nPopu*pAban).astype(np.int)    # 舍弃个体总数
    nResr = np.floor(nPopu*pResrve).astype(np.int)   # 保留个体总数
    nVari = np.floor(nPopu*pVari).astype(np.int)     # 变异个体总数
    # 优质个体直接变异产生新个体数
    nCloneWithVari = (np.floor(nPopu*pCloneWithVari).astype(np.int))//2*2
    # print(nCloneWithVari)
    nCross = nAban-nCloneWithVari  # 交叉个体总数
    # nNext = nAban + nVari  # 下一轮调整个数


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
def distanceUpdate(nNext):
    global aDistance
    wayRoll = np.zeros(size).astype(np.int)
    for ii in range(nNext):
        way = mRoute[:, aNext[ii]]
        wayRoll[0] = way[-1]
        wayRoll[1:] = way[0:size-1]
        aDistance[aNext[ii]] = npsum(mInterval[way, wayRoll])


# 规模更新
def scaleUpdate(prev, step, direction):
    global nAban, nResr, nCross, nVari, nCloneWithVari  # , nNext
    # 方案一（迟滞反馈）
    # 主要作用在于后期精细化搜索
    # if (step < 0.00001):
    #     direction -= 1
    #     if (direction == 0):
    #         scaleInitialization()
    #     if (-2 < direction) and (direction <= 0):
    #     # if (-2 < direction):
    #         nAban = np.round(nAban*0.49).astype(np.int)*2
    #         nCross = nAban - nCloneWithVari
    #         nVari = np.floor(nVari*1.02).astype(np.int)
    #         nCloneWithVari = ((nPopu*pCloneWithVari).astype(np.int))//2*2
    #         # nNext = nAban + nVari
    # # 主要作用在于前期加速搜索
    # elif (step > 0.005*prev):
    #     direction += 1
    #     if (direction == 0):
    #         scaleInitialization()
    #     if (0 <= direction) and (direction < 2):
    #     # if (direction < 2):
    #         nAban = np.round(nAban*0.51).astype(np.int)*2
    #         nCross = nAban - nCloneWithVari
    #         nVari = np.floor(nVari*0.98).astype(np.int)
    #         nCloneWithVari = ((nPopu*pCloneWithVari).astype(np.int))//2*2
    #         # nNext = nAban + nVari

    # 方案二（实时反馈）
    change = False
    if (-2 < direction) and (step < 0.00001):
        direction -= 1
        change = True
    elif (direction < 2) and (step > 0.003*prev):
        direction = 2
        change = True

    if change:
        # scaleInitialization()
        if (direction == -1):
            nAban = np.round(nPopu*pAban*0.49).astype(np.int)*2
            nCross = nAban - nCloneWithVari
            nVari = np.floor(nPopu*pVari*1.02).astype(np.int)
            # nCloneWithVari = ((nPopu*pCloneWithVari).astype(np.int))//2*2
        elif (direction == -2):
            nAban = np.round(nPopu*pAban*0.48).astype(np.int)*2
            nCross = nAban - nCloneWithVari
            nVari = np.floor(nPopu*pVari*1.04).astype(np.int)
            # nCloneWithVari = ((nPopu*pCloneWithVari).astype(np.int))//2*2
        elif (direction == 1):
            nAban = np.round(nPopu*pAban*0.51).astype(np.int)*2
            nCross = nAban - nCloneWithVari
            nVari = np.floor(nPopu*pVari*0.98).astype(np.int)
            # nCloneWithVari = ((nPopu*pCloneWithVari).astype(np.int))//2*2
        elif (direction == 2):
            nAban = np.round(nPopu*pAban*0.52).astype(np.int)*2
            nCross = nAban - nCloneWithVari
            nVari = np.floor(nPopu*pVari*0.96).astype(np.int)
            # nCloneWithVari = ((nPopu*pCloneWithVari).astype(np.int))//2*2

    return direction


# def roulette():
#     roul = (1/aDistance)/np.sum(1/aDistance)


# 适应度评价
def fitness():
    global aAban, aVari, aCross, aCloneWithVari, aResr, aNext
    rank = aDistance.argsort()
    aAban[0:nAban] = np.array(random.sample(
        list(rank[-(1.5*nAban).astype(np.int):]), nAban))
    aResr[0:nResr] = rank[0:nResr]
    # rank = rank[np.isin(rank, aAban, invert=True)]
    # aCross[0:nCross] = npappend(rank[0:7*nResr], np.array(
    #     random.sample(list(rank[7*nResr:]), nCross-7*nResr)))
    aCross[0:15*nResr] = rank[0:30*nResr:2]
    aCross[15*nResr:nCross
           ] = random.sample(list(rank[30*nResr:]), nCross-15*nResr)
    np.random.shuffle(aCross[0:nCross])
    # aCross[0:nCross] = npappend(np.array(
    #     random.sample(list(rank[0:nCross//6*4]), nCross//2)), np.array(
    #     random.sample(list(rank), nCross//2)))
    aCloneWithVari[0:nCloneWithVari] = rank[0:nCloneWithVari]
    # rank = rank[np.isin(rank, aResr[0:nResr], invert=True)]
    # rank = rank[np.isin(rank, aAban[0:nAban], invert=True)]
    rank = np.array(list(set(rank)-set(aResr[0:nResr])-set(aAban[0:nAban])))
    # aVari[0:nVari] = npappend(np.array(random.sample(list(
    #     rank[nResr:6*nResr]), 3*nResr)), np.array(random.sample(list(rank[6*nResr:]), nVari-3*nResr)))
    # aVari[0:nVari]=npappend(rank[nResr+np.arange(0, 2*nResr)*3],
    #                            np.array(random.sample(list(rank[7*nResr:]), nVari-2*nResr)))
    aVari[0:5*nResr] = rank[np.arange(0, 5*nResr)*2+nResr]
    aVari[5*nResr:nVari
          ] = random.sample(list(rank[11*nResr:]), nVari-5*nResr)
    np.random.shuffle(aVari[0:nVari])
    # aNext[0:nNext] = npappend(aAban[0:nAban], aVari[0:nVari])
    aNext[0:nAban] = aAban[0:nAban]
    aNext[nAban:nAban+nVari] = aVari[0:nVari]


# 变异、交叉、遗传
# @profile
def heredity():
    global mRoute, mVari, mConbine
    # Cross 方案一（交叉位置交叉法）
    # for ii in range(nCross//2):
    #     fGene = np.zeros(size*2).astype(np.int)
    #     cGene0 = list()
    #     cGene1 = list()
    #     for ij in range(size):
    #         fGene[[ij*2, ij*2+1]] = [mRoute[:, aCross[ii]][ij],
    #                                  mRoute[:, aCross[ii+nCross//2]][ij]]
    #     for item in fGene:
    #         if (not item in cGene0):
    #             cGene0.append(item)
    #         else:
    #             cGene1.append(item)
    #     mRoute[:, aAban[ii*2]] = cGene0
    #     mRoute[:, aAban[ii*2+1]] = cGene1

    # Cross 方案二（基于运行效率考虑，交叉原理同方案一）
    # 备份待交叉个体，防止被下一步操作影响（部分个体同时存在于待交叉与待舍弃行列）
    # mCross[:, 0:nCross] = mRoute[:, aCross[0:nCross]]
    # # 对所有即将被交叉替代路线操作，路线值为负，方便后续查找
    # mRoute[:, aAban[0:nCross]] -= size
    # for ii in range(nCross//2):
    #     # 数组指针
    #     ptr0 = ptr1 = 0
    #     for ij in range(size):
    #         # 第一个父代个体
    #         if (mCross[:, ii][ij] in mRoute[:, aAban[ii*2]]):
    #             mRoute[:, aAban[ii*2+1]][ptr1] = mCross[:, ii][ij]
    #             ptr1 += 1
    #         else:
    #             mRoute[:, aAban[ii*2]][ptr0] = mCross[:, ii][ij]
    #             ptr0 += 1
    #         # 第二个父代个体
    #         if (mCross[:, ii+nCross//2][ij] in mRoute[:, aAban[ii*2]]):
    #             # 存在边界问题
    #             mRoute[:, aAban[ii*2+1]][ptr1] = mCross[:, ii+nCross//2][ij]
    #             ptr1 += 1
    #         else:
    #             mRoute[:, aAban[ii*2]][ptr0] = mCross[:, ii+nCross//2][ij]
    #             ptr0 += 1

    # 方案三（高效 不同于方案一、方案二 直接交叉 去重补缺）
    # 点位选取
    position = np.random.randint(0, size, nCross//2)
    # 程度（交叉位置数）
    level = np.random.randint(1, size//3, nCross//2)
    # 交叉位置索引
    crossArea = np.array([np.arange(x, x+y) %
                          size for x, y in zip(position, level)])
    # 带交叉父代基因合并
    mConbine[0:size, 0:nCross//2] = mRoute[:, aCross[0:nCross//2]]
    mConbine[size:size*2, 0:nCross//2] = mRoute[:, aCross[nCross//2:nCross]]
    entirety = np.arange(size*2)
    for ii in range(nCross//2):
        mConbine[[npappend(crossArea[ii], crossArea[ii]+size)],
                 ii] = mConbine[[npappend(crossArea[ii]+size, crossArea[ii])], ii]
        # 找出未重复的一组路线
        _, index = np.unique(mConbine[:, ii], return_index=True)
        mRoute[:, aAban[ii*2]] = mConbine[:, ii][np.sort(index)]
        # 剩下的一组路线
        # indexLeft = entirety[np.isin(entirety, index, invert=True)]
        indexLeft = np.array(list(set(entirety)-set(index)))
        # print(indexLeft)
        mRoute[:, aAban[ii*2+1]] = mConbine[:, ii][np.sort(indexLeft)]

    # # Variation
    # # 优质个体直接变异替代被舍弃个体
    # # 变异
    # # 点位选取
    # position0 = np.random.randint(0, size, nCloneWithVari)
    # # 变异程度（移动位置数）
    # level = np.random.randint(1, size//3, nCloneWithVari)
    # position1 = np.array([nprandint(x+y, x+size-y) %
    #                       size for x, y in zip(position0, level)])
    # posSwap = np.array([(np.arange(x, x+y) % size, np.arange(y, y+z) % size)
    #                     for x, y, z in zip(position0, position1, level)])
    # mRoute[:, aAban[nCross:nCross+nCloneWithVari]
    #        ] = mRoute[:, aCloneWithVari[0:nCloneWithVari]]
    # for ii in range(nCloneWithVari):
    #     mRoute[:, aAban[ii+nCross]][(npappend(posSwap[ii][0], posSwap[ii][1]))
    #                                 ] = mRoute[:, aAban[ii+nCross]][(npappend(posSwap[ii][0], posSwap[ii][1]))]

    # # 方法二（所有新个体与原个体均不同，相比方法一收敛减慢）
    # 点位选取
    position = np.random.randint(0, size, nCloneWithVari)
    # 变异程度（移动位置数）
    level = np.random.randint(1, size, nCloneWithVari)
    posInsert = np.array([x % (size-y) for x, y in zip(position, level)])
    for ii in range(nCloneWithVari):
        # 选中位移的位置
        posSelect = np.arange(position[ii], position[ii]+level[ii]) % size
        # 剩余位置向量
        posLeft = np.arange(position[ii]+level[ii], position[ii]+size) % size
        #
        mRoute[:, aAban[ii+nCross]][0:posInsert[ii]
                                    ] = mRoute[:, aCloneWithVari[ii]][posLeft[0:posInsert[ii]]]

        mRoute[:, aAban[ii+nCross]][posInsert[ii]:posInsert[ii] + level[ii]
                                    ] = mRoute[:, aCloneWithVari[ii]][posSelect]

        mRoute[:, aAban[ii+nCross]][posInsert[ii]+level[ii]:
                                    ] = mRoute[:, aCloneWithVari[ii]][posLeft[posInsert[ii]:]]

    # 变异
    # 点位选取
    position0 = np.random.randint(0, size, nVari)
    # 变异程度（移动位置数）
    level = np.random.randint(1, 5, nVari)
    position1 = np.array([nprandint(x+y, x+size-y) %
                          size for x, y in zip(position0, level)])
    posSwap = np.array([(np.arange(x, x+y) % size, np.arange(y, y+z) % size)
                        for x, y, z in zip(position0, position1, level)])
    # mVari[:, 0:nVari] = mRoute[:, aVari[0:nVari]]
    for ii in range(nVari):
        mRoute[:, aVari[ii]][(npappend(posSwap[ii][0], posSwap[ii][1]))
                             ] = mRoute[:, aVari[ii]][(npappend(posSwap[ii][0], posSwap[ii][1]))]


# plot
def plot():
    plt.plot(city[:, 0], city[:, 1], 'ro')
    index = mRoute[:, aDistance.argmin()]
    plt.plot(city[npappend(index, index[0]), 0],
             city[npappend(index, index[0]), 1])


# data output
def write(usedTime, count):
    data = [size, count, np.min(aDistance), usedTime,
            mRoute[:, aDistance.argmin()].tolist()]
    with open("route.json", 'a', encoding='utf-8') as output:
        json.dump(dict(zip(('number of city:', 'generation:', 'optimal distance:',
                            'used time(s)', 'route:'), data)), output, indent=4, ensure_ascii=False)
    print("All done!")


def main():
    t0 = time.process_time()
    # 首次全体均需重新计算distance
    populationInitialization()
    intervalInitialization()
    scaleInitialization()
    distanceUpdate(nPopu)
    # 当前最优(current optimal)
    cp = np.min(aDistance)
    # run = True
    gene = 0
    thresholdValue = 0
    direction = 0
    print("{g:<10}{cp:<26}{jc:<26}".format(
        g='gene', cp='Current optimal', jc='Judgment condition'))

    while(3*size >= thresholdValue):
        gene += 1
        prev = cp
        fitness()
        heredity()
        distanceUpdate(nAban+nVari)
        jc = np.average(np.sort(aDistance)[4*nResr:7*nResr])
        cp = np.min(aDistance)
        # 最优结果步进
        step = prev - cp
        # 判决门限
        if (not step):
            thresholdValue += 1
        else:
            thresholdValue = 0
        direction = scaleUpdate(prev, step, direction)
        print("{0:<9}".format(gene), "{0:<25}".format(
            cp), "{0:<25}".format(jc))
    print(mRoute[:, aDistance.argmin()])

    usedTime = time.process_time()-t0

    # 结果输出
    write(usedTime, gene)


if __name__ == "__main__":
    t0 = time.process_time()
    main()
    print(time.process_time()-t0)
    plot()
    plt.show()
