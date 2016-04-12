from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from numpy import ndarray
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

class UnmannedNet:
  def __init__(self,n_in,n_hidden,n_out):
      self.net = FeedForwardNetwork()
      inLayer = LinearLayer(n_in)
      hiddenLayer1 = SigmoidLayer(n_hidden)
      hiddenLayer2 = SigmoidLayer(n_hidden)
      outLayer = LinearLayer(n_out)
      self.net.addInputModule(inLayer)
      self.net.addModule(hiddenLayer1)
      self.net.addModule(hiddenLayer2)
      self.net.addOutputModule(outLayer)
      in_to_hidden = FullConnection(inLayer, hiddenLayer1)
      hidden_to_out = FullConnection(hiddenLayer2, outLayer)
      hidden_to_hidden = FullConnection(hiddenLayer1, hiddenLayer2)
      self.net.addConnection(in_to_hidden)
      self.net.addConnection(hidden_to_hidden)
      self.net.addConnection(hidden_to_out)
      self.net.sortModules()
      #self.net.params
      self.ds = SupervisedDataSet(n_in, n_out)

  def load_network(self,fName='./data/mynetwork.xml'):
      self.net = NetworkReader.readFrom(fName)

  def save_network(self,fName='./data/mynetwork.xml'):
      NetworkWriter.writeToFile(self.net, fName)
  def read_data(self,fName="./data/mydata"):
      self.ds = SupervisedDataSet.loadFromFile(fName)

  def prediction(self,image):
      return self.net.activate(image)

  def evaluate(self,valueFaultTolerant):
      target = self.ds.data.get('target')
      inputvalue = self.ds.data.get('input')
      numberOfSample = target.shape[0]
      numberOfCorrect = 0
      print "the number of sample is "+str(numberOfSample)
      for i in range(0,numberOfSample):
         #print target[i] , self.prediction(inputvalue[i])
         diff1=abs(target[i][0]-self.prediction(inputvalue[i])[0])
         diff2=abs(target[i][1]-self.prediction(inputvalue[i])[1])
         if (diff1<=valueFaultTolerant and diff2<=valueFaultTolerant):
            numberOfCorrect+=1
      print "Correct rate is"+str(float(numberOfCorrect)/float(numberOfSample))

if __name__ == "__main__":
  huang = UnmannedNet(2,5,1)
   for i in range(-50,50):
    for j in range (-50,50):
      if(i<=j):
        huang.add_data((i, j), (0))
      else:
        huang.add_data((i, j), (1))
   #huang.save_data()
   #huang.read_data()
   print "start training"
   huang.train(5)
   print "start evaluate"
   huang.evaluate(0.2)
   for i in range(-50,50):
    for j in range (-50,50):
      print str(i)+" "+str(j)+"   "+str(huang.output([i,j]))


  
