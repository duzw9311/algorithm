from numpy import *
import operator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os import listdir
#数据读取
def img2vector(filename):
    returnvect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j] = int(linestr[j])
    print(returnvect)
    return returnvect

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

# filename = 'testDigits/0_13.txt'
# img2vector(filename)

def handwritingclasstest():
    hwlabels = []
    trainingfilelist = listdir('trainingDigits')
    m = len(trainingfilelist)
    trainingmat = zeros((m,1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]#以.为分隔符，将标签和.txt 分开，取前面的
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i] = img2vector('trainingDigits/{}'.format(filenamestr))

    testfilelist = listdir('testDigits')
    n = len(testfilelist)
    error = 0.0
    for i in range(n):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        if classfy0(img2vector('testDigits/{}'.format(filenamestr)),trainingmat,hwlabels,3) !=classnumstr:
            error +=1
    print("the total numbers of error is：{}".format(error))
    print('错误率',error/n)

handwritingclasstest()