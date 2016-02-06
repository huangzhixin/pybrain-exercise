
from pybrain.tools.customxml.networkwriter import NetworkWriter 
from pybrain.tools.customxml.networkreader import NetworkReader
NetworkWriter.writeToFile(net, 'mynetwork.xml')
net = NetworkReader.readFrom('mynetwork.xml')

net.params
net.activate([1,1])
