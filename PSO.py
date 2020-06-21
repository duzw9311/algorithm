import numpy as np
import matplotlib.pyplot as plt
# y=[]
# x=[]
# z=[]
#绘制粒子位置3d图所使用的坐标系
pltx=[]
plty=[]
pltz=[]
plt.ion()                                   #开启连续绘制图像（动态显示图）
fig = plt.figure(figsize=(30,5))            #创建画布（比较长因为直接横着放三个图像了）
ax = plt.subplot(131, projection='3d')      #左图，显示粒子位置，，3维图像
bx = plt.subplot(132)                       #中间的图，最后绘出迭代后函数值f（x）曲线，，2维图像
cx =plt.subplot(133,projection='3d')        #右图，画出（-5，5）函数曲面图，3维图像
#定义粒子类
class particle(object):
    def __init__(self):
        self.position = np.array([0,0])   #粒子当前所处位置
        self.speed = np.array([0,0])      #粒子当前速度
        self.PBest = np.array([0,0])      #粒子历史最优位置

#定义Pso算法类
class PSO(object):
        #设置参数
        def __init__(self,w,c1,c2,number,epoches):
            self.w = w              #惯性权重
            self.c1 = c1            #自我认知学习因子
            self.c2 = c2            #社会认知学习因子
            self.GBest = np.array([0,0])      #种群最优位置
            self.number = number    #种群粒子数目
            self.epoches = epoches  #迭代次数
            self.population = []    #粒子种群
            self.points_average_unfitness = []#存放每一轮迭代中所有粒子的平均适应度

        #计算粒子适应度
        def get_unfitness(self,position):
            x = position[0]                                               #得到粒子的x坐标
            y = position[1]                                               #得到粒子y坐标
            #unfitness = np.square(x+2*y-7)+np.square(2*x+y-5)        #通过booth函数计算粒子当前位置对应函数值f（x）
            unfitness = np.square(np.square(x)+y-11)+np.square(x+np.square(y)-7)#Himmelblau's函数公式
            return unfitness

        #初始化粒子群
        def init(self):
            for i in range(self.number):                            #迭代生成粒子群
                point = particle()                                  #实例化粒子点
                point.position = np.random.uniform(-5,5,(2))      #生成粒子点的（-5，5）的随机二维坐标
                point.PBest = point.position                        #将粒子的历史最优位置设定为当前位置
                self.population.append(point)                       #将粒子对象放入粒子种群列表中

        #获得群体最优位置
        def get_gbest(self):
                for point in self.population:                        #通过迭代整个群体所有粒子的历史最优位置获得群体最优位置
                    if self.get_unfitness(point.PBest)<self.get_unfitness(self.GBest):#如果粒子个体最优位置不适应度小于当前群体最优位置不适应度
                        self.GBest = point.position                                 #把该粒子最优位置作为群体最优位置

        #状态更新函数
        def update(self,j):
            sum = 0
            for i,point in enumerate(self.population):                                #迭代整个粒子种群的粒子对象
                #速度更新   r1,r2每次迭代都是在（0，1）之间的随机数，所以直接使用numpy生成
                speed = self.w * point.speed + self.c1 * np.random.random(1) * (
                point.PBest - point.position) + self.c2 * np.random.random(1) * (
                self.GBest - point.position)
                print("现在是粒子%d"%i)
                print("speed",speed)
                #位置更新
                position = point.position + speed
                print("position",position)
                #更新粒子参数
                if -5<position[0]<5:          #x限定在（-5，5）
                    if -5<position[1]<5:      #y限定在（-5，5）
                        point.position = position
                        point.speed = speed
                        unfitness = self.get_unfitness(point.position)
                        if unfitness <self.get_unfitness(point.PBest):#如果当前位置适应度大于粒子个体粒子适应度则更新粒子个体最优位置
                            point.PBest = point.position
                #计算所有粒子最优适应度和
                sum = self.get_unfitness(point.PBest)+sum#得到粒子函数值f（x）和作为总体适应度
            self.points_average_unfitness.append(sum/len(self.population))#得到这一轮后所有粒子最优适应度平均值

            #动态更新三维坐标系中的点的位置
            ax.set_xticks(np.linspace(-20,20,10))
            ax.set_yticks(np.linspace(-20,20,10))
            ax.set_zticks(np.linspace(0,200,20))
            ax.set_title("particle_image")
            plt.pause(0.01)
            if j <epoches-1:
                 ax.cla()           #清除ax轴所有粒子
            for i in self.population:
                pltx.append(i.position[0])
                plty.append(i.position[1])
                pltz.append(self.get_unfitness(i.position))
            ax.scatter(pltx,plty,pltz)                  #绘制粒子散点图

            #修改xyz轴刻度，否则最后一轮迭代完成会改变刻度值导致区分度太小
            ax.set_xticks(np.linspace(-20,20,10))
            ax.set_yticks(np.linspace(-20,20,10))
            ax.set_zticks(np.linspace(0,200,20))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        #总体运行函数
        def for_all(self):
            #调用初始化粒子群函数
            self.init()

            #绘制函数图像曲面
            x1 = np.linspace(-5,5,1000)
            x2 = np.linspace(-5,5,1000)
            X1,X2 = np.meshgrid(x1,x2)
            #Z=np.square(X1+2*X2-7)+np.square(2*X1+X2-5)
            Z=np.square(np.square(X1)+X2-11)+np.square(X1+np.square(X2)-7)
            cx.set_title("Himmelblau's")
            cx.plot_surface(X1,X2,Z,cmap = 'winter')
            cx.set_xlabel('X')
            cx.set_ylabel('Y')
            cx.set_zlabel('Z')

            #进行规定迭代次数的迭代
            for i in range(self.epoches):
                print("############################################################epoches",i)
                #更新状态
                self.update(i)
                #3d绘图列表元素清0，否则会导致粒子越来越多看不出来有没有收敛
                pltx.clear()
                plty.clear()
                pltz.clear()
                #获取粒子群全局最优位置
                self.get_gbest()
##################################################################################################################类定义完成
#各参数设置
epoches = 100
number = 20
w=0.5
c1=1.
c2=1.
pso = PSO(w,c1,c2,number,epoches)
pso.for_all()

# for point in pso.population:
#     x.append(point.position[0])
#     y.append(point.position[1])
#     z.append((pso.get_unfitness(point.position)))
#     print("x=", point.position[0],"y=",point.position[1],"f(x)=", pso.get_unfitness(point.position))

print("unfitness_average",pso.points_average_unfitness)
#绘制函数图形对应fx值
epoches = np.arange(0,epoches,1)
bx.plot(epoches,pso.points_average_unfitness,'r')
bx.set_xlabel('epoches')
bx.set_ylabel('unfitness_average')
bx.set_title("unfitness_average~epoches")
plt.ioff()  #关闭动态显示图，防止直接关闭
plt.show()

