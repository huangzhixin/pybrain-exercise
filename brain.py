from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import LinearLayer
net = buildNetwork(2, 3, 1, bias=True)
ds = SupervisedDataSet(2, 1)
for i in range(0,10):
   for j in range (0,10):
      if(i==j):
        ds.addSample((i, j), (i+j))
      else:
        ds.addSample((i, j), (i+j))


trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(epochs=1000)
net.activate([1,1])


hiddenclass=TanhLayer



