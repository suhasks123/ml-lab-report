import numpy as np
import random
import math
import sys

def loadFile(df):
  resultList = []
  f = open(df, 'r')
  for line in f:
    line = line.rstrip('\n')  # "1.0,2.0,3.0"
    sVals = line.split(',')   # ["1.0", "2.0, "3.0"]
    fVals = list(map(np.float32, sVals))  # [1.0, 2.0, 3.0]
    resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
  f.close()
  return np.asarray(resultList, dtype=np.float32)  # not necessary
	
def showMatrixPartial(m, numRows, dec, indices):
  fmt = "%." + str(dec) + "f" # like %.4f
  lastRow = len(m) - 1
  width = len(str(lastRow))
  for i in range(numRows):
    if indices == True:
      print("[", end='')
      print(str(i).rjust(width), end='')
      print("] ", end='')	  
  
    for j in range(len(m[i])):
      x = m[i,j]
      if x >= 0.0: print(' ', end='')
      print(fmt % x + '  ', end='')
    print('')
  print(" . . . ")

  if indices == True:
    print("[", end='')
    print(str(lastRow).rjust(width), end='')
    print("] ", end='')	  
  for j in range(len(m[lastRow])):
    x = m[lastRow,j]
    if x >= 0.0: print(' ', end='')
    print(fmt % x + '  ', end='')
  print('')	  
  
class NeuralNetwork:

  def __init__(self, numInput, numHidden, numOutput, seed):
    self.numInputs = numInput
    self.numHidden = numHidden
    self.numOutputs = numOutput
	
    self.InputNodes = np.zeros(shape=[self.numInputs], dtype=np.float32)
    self.HiddenNodes = np.zeros(shape=[self.numHidden], dtype=np.float32)
    self.OutputNodes = np.zeros(shape=[self.numOutputs], dtype=np.float32)
	
    self.inputHiddenWeights = np.zeros(shape=[self.numInputs,self.numHidden], dtype=np.float32)
    self.hiddenOutputWeights = np.zeros(shape=[self.numHidden,self.numOutputs], dtype=np.float32)
	
    self.HiddenLayerBias = np.zeros(shape=[self.numHidden], dtype=np.float32)
    self.OutputLayerBias = np.zeros(shape=[self.numOutputs], dtype=np.float32)
	
    self.randomInitializer = random.Random(seed) # allows multiple instances
    self.initializeWeights()
 	
  def setWeights(self, weights):
    if len(weights) != self.totalWeights(self.numInputs, self.numHidden, self.numOutputs):
      print("Warning: len(weights) error in setWeights()")	

    idx = 0
    for i in range(self.numInputs):
      for j in range(self.numHidden):
        self.inputHiddenWeights[i,j] = weights[idx]
        idx += 1
		
    for j in range(self.numHidden):
      self.HiddenLayerBias[j] = weights[idx]
      idx += 1

    for j in range(self.numHidden):
      for k in range(self.numOutputs):
        self.hiddenOutputWeights[j,k] = weights[idx]
        idx += 1
	  
    for k in range(self.numOutputs):
      self.OutputLayerBias[k] = weights[idx]
      idx += 1
	  
  def getWeights(self):
    tw = self.totalWeights(self.numInputs, self.numHidden, self.numOutputs)
    result = np.zeros(shape=[tw], dtype=np.float32)
    idx = 0  # points into result
    
    for i in range(self.numInputs):
      for j in range(self.numHidden):
        result[idx] = self.inputHiddenWeights[i,j]
        idx += 1
		
    for j in range(self.numHidden):
      result[idx] = self.HiddenLayerBias[j]
      idx += 1

    for j in range(self.numHidden):
      for k in range(self.numOutputs):
        result[idx] = self.hiddenOutputWeights[j,k]
        idx += 1
	  
    for k in range(self.numOutputs):
      result[idx] = self.OutputLayerBias[k]
      idx += 1
	  
    return result
 	
  def initializeWeights(self):
    numWts = self.totalWeights(self.numInputs, self.numHidden, self.numOutputs)
    wts = np.zeros(shape=[numWts], dtype=np.float32)
    lo = -0.01; hi = 0.01
    for idx in range(len(wts)):
      wts[idx] = (hi - lo) * self.randomInitializer.random() + lo
    self.setWeights(wts)

  def computeOutputs(self, xValues): # A function to do forward propagation over the 3 layers of the neural network.E
    hiddenLayerSums = np.zeros(shape=[self.numHidden], dtype=np.float32)
    outputLayerSums = np.zeros(shape=[self.numOutputs], dtype=np.float32)

    for i in range(self.numInputs):        # Input layer
      self.InputNodes[i] = xValues[i]   

    for j in range(self.numHidden):        # Linear combination of the weights between input and hidden layers and the input neurons
      for i in range(self.numInputs):
        hiddenLayerSums[j] += self.InputNodes[i] * self.inputHiddenWeights[i,j]

    for j in range(self.numHidden):        # Adding the hidden layer bias
      hiddenLayerSums[j] += self.HiddenLayerBias[j]
	  
    for j in range(self.numHidden):        # Activation function for hidden layer(TanH)
      self.HiddenNodes[j] = self.hypertan(hiddenLayerSums[j])

    for k in range(self.numOutputs):        # Linear combination of the weights between hidden and output layers and the hidden layer neurons
      for j in range(self.numHidden):
        outputLayerSums[k] += self.HiddenNodes[j] * self.hiddenOutputWeights[j,k]

    for k in range(self.numOutputs):        # Adding output layer Bias
      outputLayerSums[k] += self.OutputLayerBias[k]
 
    softmaxOutput = self.softmax(outputLayerSums)
    for k in range(self.numOutputs):        # Calculating Softmax output between the 3 classes 
      self.OutputNodes[k] = softmaxOutput[k]
	  
    result = np.zeros(shape=self.numOutputs, dtype=np.float32)
    for k in range(self.numOutputs):
      result[k] = self.OutputNodes[k]
	  
    return result
	
  def train(self, trainData, maxEpochs, learnRate):
    hiddenOutputGradients = np.zeros(shape=[self.numHidden, self.numOutputs], dtype=np.float32)  # hidden-to-output weights gradients
    OutputBiasGradients = np.zeros(shape=[self.numOutputs], dtype=np.float32)  # output node biases gradients
    InputHiddenGradients = np.zeros(shape=[self.numInputs, self.numHidden], dtype=np.float32)  # input-to-hidden weights gradients
    HiddenBiasGradients = np.zeros(shape=[self.numHidden], dtype=np.float32)  # hidden biases gradients
	
    outputSignals = np.zeros(shape=[self.numOutputs], dtype=np.float32)  # output signals: gradients w/o assoc. input terms
    hiddenSignals = np.zeros(shape=[self.numHidden], dtype=np.float32)  # hidden signals: gradients w/o assoc. input terms

    epoch = 0
    x_values = np.zeros(shape=[self.numInputs], dtype=np.float32)
    t_values = np.zeros(shape=[self.numOutputs], dtype=np.float32)
    numTrainItems = len(trainData)
    indices = np.arange(numTrainItems)          # [0, 1, 2, . . n-1]  # randomInitializer.shuffle(v)

    while epoch < maxEpochs:
      self.randomInitializer.shuffle(indices)   # scramble order of training items
      for ii in range(numTrainItems):
        idx = indices[ii]
        for j in range(self.numInputs):
          x_values[j] = trainData[idx, j]             # get the input values	
        for j in range(self.numOutputs):
          t_values[j] = trainData[idx, j + self.numInputs]   # get the corresponding class values
        self.computeOutputs(x_values)                 # results stored internally
		
    ### BACKPROPAGATION PHASE BEGINS 

        # Compute output node signals
        for k in range(self.numOutputs):
          derivative = (1 - self.OutputNodes[k]) * self.OutputNodes[k]
          outputSignals[k] = derivative * (self.OutputNodes[k] - t_values[k])
        
        # Compute hidden-to-output weight gradients using output signals
        for j in range(self.numHidden):
          for k in range(self.numOutputs):
            hiddenOutputGradients[j, k] = outputSignals[k] * self.HiddenNodes[j]
			
        # Compute output node bias gradients using output signals
        for k in range(self.numOutputs):
          OutputBiasGradients[k] = outputSignals[k] * 1.0  # 1.0 dummy input can be dropped
		  
        # Compute hidden node signals
        for j in range(self.numHidden):
          sum = 0.0
          for k in range(self.numOutputs):
            sum += outputSignals[k] * self.hiddenOutputWeights[j,k]
          derivative = (1 - self.HiddenNodes[j]) * (1 + self.HiddenNodes[j])  # tanh activation's derivative
          hiddenSignals[j] = derivative * sum
		 
        # Compute input-to-hidden weight gradients using hidden signals
        for i in range(self.numInputs):
          for j in range(self.numHidden):
            InputHiddenGradients[i, j] = hiddenSignals[j] * self.InputNodes[i]

        # Compute hidden node bias gradients using hidden signals
        for j in range(self.numHidden):
          HiddenBiasGradients[j] = hiddenSignals[j] * 1.0  # 1.0 dummy input can be dropped

        # Update weights and biases using the gradients
		
        # Update input-to-hidden weights
        for i in range(self.numInputs):
          for j in range(self.numHidden):
            delta = -1.0 * learnRate * InputHiddenGradients[i,j]
            self.inputHiddenWeights[i, j] += delta
			
        # Update hidden node biases
        for j in range(self.numHidden):
          delta = -1.0 * learnRate * HiddenBiasGradients[j]
          self.HiddenLayerBias[j] += delta      
		  
        # Update hidden-to-output weights
        for j in range(self.numHidden):
          for k in range(self.numOutputs):
            delta = -1.0 * learnRate * hiddenOutputGradients[j,k]
            self.hiddenOutputWeights[j, k] += delta
			
        # Update output node biases
        for k in range(self.numOutputs):
          delta = -1.0 * learnRate * OutputBiasGradients[k]
          self.OutputLayerBias[k] += delta

    ### BACKPROPAGATION PHASE ENDS
 		  
      epoch += 1
	  
      if epoch % 10 == 0:
        mse = self.meanSquaredError(trainData)
        print("Epoch = " + str(epoch) + " Mean Squared Error = %0.4f " % mse)
    # end while
    
    result = self.getWeights()
    return result
  # end train
  
  def accuracy(self, tdata):  # train or test data matrix
    num_correct = 0; num_wrong = 0
    x_values = np.zeros(shape=[self.numInputs], dtype=np.float32)
    t_values = np.zeros(shape=[self.numOutputs], dtype=np.float32)

    for i in range(len(tdata)): 
      for j in range(self.numInputs):  
        x_values[j] = tdata[i,j]
      for j in range(self.numOutputs): 
        t_values[j] = tdata[i, j+self.numInputs]

      y_values = self.computeOutputs(x_values)  
      max_index = np.argmax(y_values)  

      if abs(t_values[max_index] - 1.0) < 1.0e-5:
        num_correct += 1
      else:
        num_wrong += 1

    return (num_correct * 1.0) / (num_correct + num_wrong)

  def meanSquaredError(self, tdata):  # on train or test data matrix
    sumSquaredError = 0.0
    x_values = np.zeros(shape=[self.numInputs], dtype=np.float32)
    t_values = np.zeros(shape=[self.numOutputs], dtype=np.float32)

    for ii in range(len(tdata)):  # walk thru each data item
      for jj in range(self.numInputs):  # peel off input values from curr data row 
        x_values[jj] = tdata[ii, jj]
      for jj in range(self.numOutputs):  # peel off target values from curr data row
        t_values[jj] = tdata[ii, jj+self.numInputs]

      y_values = self.computeOutputs(x_values)  # computed output values
	  
      for j in range(self.numOutputs):
        err = t_values[j] - y_values[j]
        sumSquaredError += err * err  # (t-o)^2
		
    return sumSquaredError / len(tdata)
          
  @staticmethod
  def hypertan(x):
    if x < -20.0:
      return -1.0
    elif x > 20.0:
      return 1.0
    else:
      return math.tanh(x)

  @staticmethod	  
  def softmax(outputLayerSums):
    result = np.zeros(shape=[len(outputLayerSums)], dtype=np.float32)
    m = max(outputLayerSums)
    divisor = 0.0
    for k in range(len(outputLayerSums)):
       divisor += math.exp(outputLayerSums[k] - m)
    for k in range(len(result)):
      result[k] =  math.exp(outputLayerSums[k] - m) / divisor
    return result
	
  @staticmethod
  def totalWeights(nInput, nHidden, nOutput):
   tw = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
   return tw

# end class NeuralNetwork

def main():
    
  # Neural Network with 4 inputs, 5 neuron hidden layer and 3 neuron output softmax layer.
  numInput = 4
  numHidden = 5
  numOutput = 3
  print("Number of Input layer neurons: " + str(numInput))
  print("Number of Hidden layer neurons: " + str(numHidden))
  print("Number of Output layer neurons: " + str(numOutput))
  nn = NeuralNetwork(numInput, numHidden, numOutput, seed=3)
  
  print("\nLoading Iris training and test data ")
  trainDataPath = "../Dataset/irisTrainData.txt"
  trainDataMatrix = loadFile(trainDataPath)
  print("\nTrain data: ")
  showMatrixPartial(trainDataMatrix, 4, 1, True)
  testDataPath = "../Dataset/irisTestData.txt"
  testDataMatrix = loadFile(testDataPath)
  
  maxEpochs = 50
  learnRate = 0.05
  print("\nSetting maxEpochs = " + str(maxEpochs))
  print("Setting learning rate = %0.3f " % learnRate)
  print("\nStarting training")
  nn.train(trainDataMatrix, maxEpochs, learnRate)
  print("Training complete")
  
  accTrain = nn.accuracy(trainDataMatrix)
  accTest = nn.accuracy(testDataMatrix)
  
  print("\nAccuracy on 120-item train data = %0.4f " % accTrain)
  print("Accuracy on 30-item test data   = %0.4f " % accTest)
     
if __name__ == "__main__":
  main()

# end script

