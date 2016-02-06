from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
net = RecurrentNetwork()
inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3) 
outLayer = LinearLayer(1)
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
hidden_to_hidden = FullConnection(hiddenLayer, hiddenLayer)
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)
net.addRecurrentConnection(hidden_to_hidden)
net.sortModules()
