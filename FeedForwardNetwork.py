from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from numpy import ndarray
net = FeedForwardNetwork()
inLayer = LinearLayer(2)
hiddenLayer1 = SigmoidLayer(3) 
hiddenLayer2 = SigmoidLayer(3) 
outLayer = LinearLayer(1)
net.addInputModule(inLayer)
net.addModule(hiddenLayer1)
net.addModule(hiddenLayer2)
net.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer, hiddenLayer1)
hidden_to_out = FullConnection(hiddenLayer2, outLayer)
hidden_to_hidden = FullConnection(hiddenLayer1, hiddenLayer2)
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_hidden)
net.addConnection(hidden_to_out)

net.sortModules()
net.params
ds = SupervisedDataSet(2, 1)
for i in range(-50,50):
   for j in range (-50,50):
      if(i<=j):
        ds.addSample((i, j), (0))
      else:
        ds.addSample((i, j), (1))


trainer = BackpropTrainer(net, ds)
trainer.train()
net.params
net.activate([1,1])
