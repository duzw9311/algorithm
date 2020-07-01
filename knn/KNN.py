#coding:utf-8
from numpy import *
import operator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#数据提取预处理
def file2matrix(filename):
    fr = open(filename)
    arrayolines = fr.readlines() #一次性读取整个文件内容到一个迭代器以供我们遍历（读取到一个list中，以供使用，比较方便）
    numberoflines = len(arrayolines)
    returnmat = zeros((numberoflines,3))
    classlabelvector = []
    index = 0
    for line in arrayolines:
        line = line.strip() #字符串中删除开头和结尾空白符，包括转义字符
        listfromline = line.split('\t')#转换为列表以\t为分隔（遇见\t就把前后两部分切片）
        returnmat[index,:] = listfromline[0:3]#列表每行前三个读取为array格式（numpy）
        classlabelvector.append(int(listfromline[-1])) #每一行最后一个数据，也就是标签
        index+=1
    return returnmat,classlabelvector

#归一化特征值(x-min)/(max-min)归为（0，1）
def autonorm(dataset):
    #获取每一特征维度最小值和最大值
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals-minvals
    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normdataset = (dataset -tile(minvals,(m,1)))/tile(ranges,(m,1))
    print(tile(minvals,(m,1)))
    return normdataset,ranges,minvals

#分类算法
def classfy0(inx,dataset,labels,k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inx,(datasetsize,1))-dataset #扩充为行数为数据集行数的array(求两点x，y差值)
    sqdiffmat = diffmat**2                      #差值加平方
    sqdistance = sqdiffmat.sum(axis = 1)**0.5   #(x2+y2)^1/2
    sorteddistance = sqdistance.argsort()       #从小到大排序返回列表索引
    classcount = {}
    for i in range(k):
        voteilabel = labels[sorteddistance[i]]
        classcount[voteilabel] = classcount.get(voteilabel,0)+1
    sortedclasscount = sorted(classcount.items(),key = operator.itemgetter(1),reverse = True)
    #true降序，false升序 key指定比较第二个元素也就是值value itemgetter获取第1个域的值和lambda相似
    #.items返回成列表元组类型可遍历【（），（），（）】
    return sortedclasscount[0][0]

#进行预测代码计算错误率
def datingclasstest():
    horatio = 0.10
    datingdatamat,datinglabels = file2matrix('datingTestSet2.txt')
    normmat,ranges,minvals = autonorm(datingdatamat)
    m = normmat.shape[0]
    numtestvecs = int(m*horatio)
    print(numtestvecs)
    errorcount = 0.0
    for i in range(numtestvecs):#前100个数据进行测试，后面900个数据是原数据集，k取3
        classifierresult = classfy0(normmat[i,:],normmat[numtestvecs:m,:],datinglabels[numtestvecs:m],4)
        print("the classifier came back with %d the real answer is %d" %(classifierresult,datinglabels[i]))
        if classifierresult!=datinglabels[i]:errorcount+=1.0
    print("the total error rate is %f"%(errorcount/float(numtestvecs)))

#人性化手动输入约会网站推荐标签测试自己喜欢不
def classifyperson():
    resultlist = ["not at all","in small doses","in large doses"]
    percenttacts = float(input("percentage of time spend playing video games"))
    ffmiles = float(input("FREQUENT FLITER MILES EARNED PER YEAR"))
    icecream = float(input("liters of ice cream consumed per year"))
    datingdatamat,datinglabels = file2matrix('datingTestSet2.txt')
    normmat,ranges,minvals = autonorm(datingdatamat)
    inarr = array([ffmiles,percenttacts,icecream])
    classifierresult = classfy0(((inarr-minvals)/ranges),normmat,datinglabels,4)
    print("you will like this person",resultlist[classifierresult-1])#分类结果为1，2，3所以减1


#huitu
def showdatas(datingDataMat, datingLabels):
     fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13,8))
     plt.rcParams['font.sans-serif'] = ['SimHei']
     numberOfLabels = len(datingLabels)
     LabelsColors = []
     for i in datingLabels:
         if i == 1:
             LabelsColors.append('black')
         if i == 2:
             LabelsColors.append('orange')
         if i == 3:
             LabelsColors.append('red')
     #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
     axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
     #设置标题,x轴label,y轴label
     axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
     axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
     axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占')
     plt.setp(axs0_title_text, size=9, weight='bold', color='red')
     plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
     plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
     #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
     axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
     #设置标题,x轴label,y轴label
     axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数')
     axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
     axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数')
     plt.setp(axs1_title_text, size=9, weight='bold', color='red')
     plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
     plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

     #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
     axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
     #设置标题,x轴label,y轴label
     axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
     axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
     axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
     plt.setp(axs2_title_text, size=9, weight='bold', color='red')
     plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
     plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
     #设置图例
     didntLike = mlines.Line2D([], [], color='black', marker='.',
                       markersize=6, label='didntLike')
     smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                       markersize=6, label='smallDoses')
     largeDoses = mlines.Line2D([], [], color='red', marker='.',
                       markersize=6, label='largeDoses')
     #添加图例
     axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
     axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
     axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
     #显示图片
     plt.show()

datingdatamat,datinglabels = file2matrix('datingTestSet2.txt')
showdatas(datingdatamat,datinglabels)
classifyperson()
#绘原数据点图
# datingdatamat,datinglabels = file2matrix('datingTestSet2.txt')
# #第一维度特征和第二维度特征绘制散点图（比23区分度明显）
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingdatamat[:,0],datingdatamat[:,1],15.0*array(datinglabels),15.0*array(datinglabels))#x,y,s,c  marker = '*'修改marker有很多超好看的图hhhh
# plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
# ax.set_xlabel("玩视频游戏所耗时间百分比")
# ax.set_ylabel("每周消费的冰淇淋公斤数")
# plt.show()
