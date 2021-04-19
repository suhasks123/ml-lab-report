import numpy as np
import random
import math
import sys

def loadFile(data):
  resultList = []
  f = open(data, 'r')
  for line in f:
    line = line.rstrip('\n')  
    sVals = line.split(',')   
    fVals = list(map(np.float32, sVals))  
    resultList.append(fVals) 
  f.close()
  npresults = np.asarray(resultList, dtype=np.float32)
  return npresults

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
  
class NeuralNetwork:

  def __init__(self, numInput, numHidden, numOutput, seed):
    self.numInputs = numInput
    self.numHidden = numHidden
    self.numOutputs = numOutput
	
    self.InputNodes = np.zeros(shape=[self.numInputs], dtype=np.float32)
    self.inputHiddenWeights = np.zeros(shape=[self.numInputs,self.numHidden], dtype=np.float32)
    self.HiddenLayerBias = np.zeros(shape=[self.numHidden], dtype=np.float32)

    self.HiddenNodes = np.zeros(shape=[self.numHidden], dtype=np.float32)
    self.hiddenOutputWeights = np.zeros(shape=[self.numHidden,self.numOutputs], dtype=np.float32)
    self.OutputLayerBias = np.zeros(shape=[self.numOutputs], dtype=np.float32)
    self.OutputNodes = np.zeros(shape=[self.numOutputs], dtype=np.float32)
		
    self.randomInitializer = random.Random(seed) # allows multiple instances
    self.totalWeights = (self.numInputs * self.numHidden) + (self.numHidden * self.numOutputs) + self.numHidden + self.numOutputs
    self.initializeWeights()
 	
  def setWeights(self, weights):
    if len(weights) != self.totalWeights:
      print("Weights number mismatch! Cannot set!")	

    idx = 0
    for i in range(self.numInputs):
      for j in range(self.numHidden):
        self.inputHiddenWeights[i,j] = weights[idx]
        idx += 1
		
    for i in range(self.numHidden):
      self.HiddenLayerBias[i] = weights[idx]
      idx += 1

    for i in range(self.numHidden):
      for j in range(self.numOutputs):
        self.hiddenOutputWeights[i,j] = weights[idx]
        idx += 1
	  
    for i in range(self.numOutputs):
      self.OutputLayerBias[i] = weights[idx]
      idx += 1
 	
  def initializeWeights(self):
    wts = [0] * self.totalWeights
    wts = np.asarray(wts, dtype=np.float32)
    low = -0.01 
    high = 0.01
    for idx in range(0, len(wts)):
      wts[idx] = (high - low) * self.randomInitializer.random() + low
    self.setWeights(wts)

  def computeOutputs(self, xValues, epochnum): # A function to do forward propagation over the 3 layers of the neural network.
    hiddenLayerSums = np.zeros(shape=[self.numHidden], dtype=np.float32)
    outputLayerSums = np.zeros(shape=[self.numOutputs], dtype=np.float32)

    for i in range(self.numInputs):        # Input layer
      self.InputNodes[i] = xValues[i]   

    if epochnum == 0: 
      print("Input nodes: ")
      print(self.InputNodes)

    for j in range(self.numHidden):        # Linear combination of the weights between input and hidden layers and the input neurons
      for i in range(self.numInputs):
        hiddenLayerSums[j] += self.InputNodes[i] * self.inputHiddenWeights[i,j]

    if epochnum == 0:
      print("Hidden layer sums before bias: ")
      print(hiddenLayerSums)

    for j in range(self.numHidden):        # Adding the hidden layer bias
      hiddenLayerSums[j] += self.HiddenLayerBias[j]
    if epochnum == 0:
      print("Hidden layer sums after bias: ")
      print(hiddenLayerSums)
	  
    for j in range(self.numHidden):        # Activation function for hidden layer(TanH)
      interSum = hiddenLayerSums[j]
      if interSum < -18.0:
        self.HiddenNodes[j] = -1.0
      elif interSum > 18.0:
        self.HiddenNodes[j] = 1.0
      else:
        self.HiddenNodes[j] = math.tanh(interSum)

    if epochnum == 0:
      print("Hidden layer sums after activation: ")
      print(self.HiddenNodes)

    for k in range(self.numOutputs):        # Linear combination of the weights between hidden and output layers and the hidden layer neurons
      for j in range(self.numHidden):
        outputLayerSums[k] += ( self.HiddenNodes[j] * self.hiddenOutputWeights[j,k])

    if epochnum == 0:
      print("Output layer sums before bias: ")
      print(outputLayerSums)

    for k in range(self.numOutputs):        # Adding output layer Bias
      outputLayerSums[k] += self.OutputLayerBias[k]

    if epochnum == 0:
      print("Output layer sums after bias: ")
      print(outputLayerSums)

    softmaxOutput = self.softmax(outputLayerSums)
    for k in range(self.numOutputs):        # Calculating Softmax output between the 3 classes 
      self.OutputNodes[k] = softmaxOutput[k]
    
    if epochnum == 0:
      print("Output nodes after softmax: ")
      print(self.OutputNodes)

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
      if epoch == 0:
        print("Input - Hidden Weights: ")
        print(self.inputHiddenWeights)
        print("Hidden - Output Weights: ")
        print(self.hiddenOutputWeights)
        print("Hidden bias: ")
        print(self.HiddenLayerBias)
        print("Output bias: ")
        print(self.OutputLayerBias)
      epochnum = epoch
      self.randomInitializer.shuffle(indices)   # scramble order of training items
      for ii in range(numTrainItems):
        idx = indices[ii]
        for j in range(self.numInputs):
          x_values[j] = trainData[idx, j]             # get the input values	
        for j in range(self.numOutputs):
          t_values[j] = trainData[idx, j + self.numInputs]   # get the corresponding class values
        self.computeOutputs(x_values, epoch)                 # results stored internally
        epochnum = 1

    ### BACKPROPAGATION PHASE BEGINS 
        epochnum = epoch
        # Compute output node signals
        for k in range(self.numOutputs):
          derivative = (1 - self.OutputNodes[k]) * self.OutputNodes[k]
          outputSignals[k] = derivative * (self.OutputNodes[k] - t_values[k])
        

        # Compute hidden-to-output weight gradients using output signals
        for j in range(self.numHidden):
          for k in range(self.numOutputs):
            hiddenOutputGradients[j, k] = outputSignals[k] * self.HiddenNodes[j]

        if(epochnum == 0):
          print("Hidden - Output layer weight matrix gradient: ")
          print(hiddenOutputGradients)
			
        # Compute output node bias gradients using output signals
        for k in range(self.numOutputs):
          OutputBiasGradients[k] = outputSignals[k] * 1.0  # 1.0 dummy input can be dropped
        if(epochnum == 0):
          print("Output layer bias matrix gradient: ")
          print(OutputBiasGradients)
        
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

        if(epochnum == 0):
          print("Hidden - Input layer weight matrix gradient: ")
          print(InputHiddenGradients)
			
        # Compute hidden node bias gradients using hidden signals
        for j in range(self.numHidden):
          HiddenBiasGradients[j] = hiddenSignals[j] * 1.0  # 1.0 dummy input can be dropped

        if(epochnum == 0):
          print("Hidden layer bias matrix gradient: ")
          print(HiddenBiasGradients)

        # Update input-to-hidden weights
        for i in range(self.numInputs):
          for j in range(self.numHidden):
            delta = -1.0 * learnRate * InputHiddenGradients[i,j]
            self.inputHiddenWeights[i, j] += delta

        if epochnum == 0:
          print("Delta value, i.e, deviation from initial value for the input-hidden weights: ")
          print(delta)
          print("Updated input-hidden weight matrix: ")
          print(self.inputHiddenWeights)

        # Update hidden node biases
        for j in range(self.numHidden):
          delta = -1.0 * learnRate * HiddenBiasGradients[j]
          self.HiddenLayerBias[j] += delta      
        if epochnum == 0:          
          print("Delta value, i.e, deviation from initial value for the hidden layer biases: ")
          print(delta)
          print("Updated hidden layer biases")
          print(self.HiddenLayerBias)
		  
        # Update hidden-to-output weights
        for j in range(self.numHidden):
          for k in range(self.numOutputs):
            delta = -1.0 * learnRate * hiddenOutputGradients[j,k]
            self.hiddenOutputWeights[j, k] += delta

        if epochnum == 0:
          print("Delta value, i.e, deviation from initial value for the hidden-output weights: ")
          print(delta)
          print("Updated hidden-output weight matrix: ")
          print(self.hiddenOutputWeights)
        # Update output node biases
        for k in range(self.numOutputs):
          delta = -1.0 * learnRate * OutputBiasGradients[k]
          self.OutputLayerBias[k] += delta

        if epochnum == 0:
          print("Delta value, i.e, deviation from initial value for the output layer biases: ")
          print(delta)
          print("Updated output layer biases")
          print(self.OutputLayerBias)
        epochnum = 1
    ### BACKPROPAGATION PHASE ENDS
        
      epoch += 1
	  
      if epoch % 10 == 0:
        mse = self.meanSquaredError(trainData)
        print("Epoch = " + str(epoch) + " Mean Squared Error = %0.4f " % mse)
  # end train
  
  def accuracy(self, tdata):  # train or test data matrix
    correct_amt = 0
    wrong_amt = 0
    input_vals = [None] * self.numInputs
    input_vals = np.asarray(input_vals, dtype=np.float32)
    output_vals = [None] * self.numOutputs
    output_vals = np.asarray(output_vals, dtype = np.float32)

    for i in range(0, len(tdata)):
      for j in range(0, self.numInputs):
        input_vals[j] = tdata[i][j]
      for j in range(0, self.numOutputs):
        output_vals[j] = tdata[i][j + self.numInputs]

      predicted_vals = self.computeOutputs(input_vals, 1)  
      mx = np.argmax(predicted_vals)  

      deviation = abs(output_vals[mx] - 1.0)
      if deviation >= 1e-3:
        wrong_amt += 1
      else:
        correct_amt += 1

    acc = correct_amt / (correct_amt + wrong_amt)
    return acc

  def meanSquaredError(self, tdata):  # on train or test data matrix

    initialSumError = 0.0
    input_vals = [0] * self.numInputs
    input_vals = np.asarray(input_vals, dtype=np.float32)
    output_vals = [0] * self.numOutputs
    output_vals = np.asarray(output_vals, dtype = np.float32)

    for i in range(0, len(tdata)):
      for j in range(0, self.numInputs):
        input_vals[j] = tdata[i][j]
      for j in range(0, self.numOutputs):
        output_vals[j] = tdata[i][j + self.numInputs]
    
    predicted_vals = self.computeOutputs(input_vals, 1)

    for i in range(self.numOutputs):
      error_squared = (output_vals[i] - predicted_vals[i])*(output_vals[i] - predicted_vals[i])
      initialSumError += error_squared

    initialSumError /= len(tdata)
    return initialSumError

  @staticmethod	  
  def softmax(outputLayerSums):
    result = [0] * len(outputLayerSums)
    result = np.asarray(result, dtype=np.float32)
    m = max(outputLayerSums)
    divisor = 0.0
    exponents = []
    for k in range(0, len(outputLayerSums)):
      expo = math.exp(outputLayerSums[k] - m)
      exponents.append(expo)
      divisor += expo
    for k in range(0, len(result)):
      result[k] =  exponents[k] / divisor
    return result

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
  print("\nStarting training")
  nn.train(trainDataMatrix, maxEpochs, learnRate)
  print("Training complete")
  
  accTrain = nn.accuracy(trainDataMatrix)
  accTest = nn.accuracy(testDataMatrix)
  
  print("\nAccuracy on train dataset = %0.4f " % accTrain)
  print("Accuracy on test data   = %0.4f " % accTest)
     
if __name__ == "__main__":
  main()

# end script

