# -*- coding: utf-8 -*-
#http://pybrain.org/docs/tutorial/fnn.html
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)   #创建分类数据组
for n in xrange(400):                                 #建立了1200个数据组
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])
tstdata, trndata = alldata.splitWithProportion( 0.25 )   
#用于训练的75%  测试的25%  tstdata 包含三项 tstdata['input']  tstdata['target'] tstdata['class']
#目前target是每个样本的类[0]or[1]or[2] class为空
trndata._convertToOneOfMany( )     #通过这个函数target变为[[1,0,0][0,1,0][0,0,1]...]                    
tstdata._convertToOneOfMany( )     #class 变为[0]or[1]or[2]
print "Number of training patterns: ", len(trndata)  #训练数据的样本数
print "Input and output dimensions: ", trndata.indim, trndata.outdim    
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
ticks = arange(-3.,6.,0.2)   #建立一个从-3到6，梯度0.2的坐标
X, Y = meshgrid(ticks, ticks) #建立X和Y矩阵，给每个点附一个值 
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)  
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])    #通过X.ravel()将一个多维数组变成一维数组
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy
for i in range(20):
    trainer.trainEpochs( 1 )     #将数据训练一次
    trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
    out = fnn.activateOnDataset(griddata)   
    #out输出来的是griddata样本个数行的，类型个数列的一个数组，[[0.24,0.84,0.66][0.21,0.36,0.93][0.11,0.556,0.21]...]
    #这表示每一个样本分别在1，2，3类的概率是多少
    out = out.argmax(axis=1)  # the highest output activation gives the class，每个样本取最大概率的类 out=[[1],[2],[3],[2]...]
    out = out.reshape(X.shape) #把这些变得和X一个形状，方便画图
    
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in [0,1,2]:
        here, _ = where(tstdata['class']==c)
        #where算出来的是类型等于c的样本在tstdata的序数
        plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
        contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on
    draw()  # update the plot
ioff()
show()
