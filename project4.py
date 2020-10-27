import numpy
import csv
import math
import random
import sys

def activation(value):
    return 1 / (1 + math.exp(-1 * value))

def activationDerivative(value):
    return (activation(value)*(1-activation(value)))


def trainModel(trainData, trainLabels):
    edges = []
    #learning rate used for weight updating
    learningRate = 0.5
    
    #number of hidden nodes to be used for the neural net
    numHiddenNodes = 100
    
    #edge weights going from the input layer to the hidden layer
    hiddenEdgeList = []
    
    #initialize weights to random number between 0 and 1 to each hidden node
    for i in range(numHiddenNodes):
        edgeList = []
        for j in range(len(trainData[i])):
            edgeList.append(random.uniform(-1,1))
        hiddenEdgeList.append(edgeList)
        
    #edge weights going from the hidden layer to the output layer
    outputEdgeList = []
    
    #initialize weights to random number between 0 and 1
    for i in range(len(hiddenEdgeList)):
        outputEdgeList.append(random.uniform(-1,1))
        
    #used to measure progress of training algorithm    
    progressCount = 0
    total = len(trainData)
    
    #train the model on each training data input set
    for i in range(len(trainData)):
        inputLayerMatrix = numpy.array(trainData[i])
        activatedInput = []
        for j in range(len(trainData[i])):
            activatedInput.append(activation(trainData[i][j]))
        
        #initialize the hidden layer
        hiddenLayer = []

        #calculate hidden layer values from inputs
        for j in range(numHiddenNodes):
            edgeMatrix = numpy.array(hiddenEdgeList[j])
            hiddenLayer.append(numpy.dot(inputLayerMatrix, edgeMatrix) + 1)
          
        #store the activated values of he hidden layer
        activatedHiddenLayer = []
        for i in range(len(hiddenLayer)):
            activatedHiddenLayer.append(activation(hiddenLayer[i]))
            
        #calculate output for network run
        outputEdgeMatrix = numpy.array(outputEdgeList)
        activatedMatrix = numpy.array(activatedHiddenLayer)
        outputLayer = numpy.dot(activatedMatrix, outputEdgeMatrix) + 1
        activatedOutputLayer = activation(outputLayer)
        
        #calculate error given the current input's class label
        error = trainLabels[i] - activatedOutputLayer
        
        #calculated the modified error for the hidden layer to output layer weight updating
        outputModError = error * activationDerivative(activatedOutputLayer)
        
         #calculate the modified error for updating weights from the input to the hidden layer
        hiddenModError = []
        for i in range(len(hiddenLayer)):
            modError = activationDerivative(activatedHiddenLayer[i]) * outputModError * outputEdgeList[i]
            hiddenModError.append(modError)
        
        #modify the weights moving from the hidden layer to the output layer
        for i in range(len(outputEdgeList)):
            outputEdgeList[i] = outputEdgeList[i] + (learningRate * activatedHiddenLayer[i] * outputModError)         
            
        #update the edge weights from the input layer to the hidden layer
        for i in range(len(hiddenEdgeList)):#for each input
            for j in range(len(hiddenEdgeList[i])):#for each input's edges
                hiddenEdgeList[i][j] = hiddenEdgeList[i][j] + (learningRate * activatedInput[j] * hiddenModError[i])
        
        progressCount += 1
        progress = (progressCount / total) * 100
        progress = str('%.2f'%progress)
        sys.stdout.write('\r' + "Percent Complete: " + progress + "%")
        sys.stdout.flush()
    edges.append(hiddenEdgeList)
    edges.append(outputEdgeList)
    return edges

def testModel(testData, testLabels, edgeList):  
    numHiddenNodes = 100
    output = 0
    numCorrect = 0
    total = len(testData)
    #used to measure progress of training algorithm    
    progressCount = 0
    total = len(testData)
    outputs = []
    #run the network on the test data
    for i in range(len(testData)):
        #calculate activated hidden node values
        hiddenValues = []
        input = numpy.array(testData[i])
        for j in range(numHiddenNodes):
            hiddenEdges = numpy.array(edgeList[0][j])
            value = activation(numpy.dot(input, hiddenEdges))
            hiddenValues.append(value)
        #calculate activated output
        hiddenInput = numpy.array(hiddenValues)
        edges = numpy.array(edgeList[1])
        output = activation(numpy.dot(hiddenInput, edges))
        outputs.append(output)
        #determine accuracy of output
        if(output >= 0.5 and testLabels[i] == 1.0):
            numCorrect += 1
        elif(output < 0.5 and testLabels[i] == 0.0):
            numCorrect += 1
        progressCount += 1
        progress = (progressCount / total) * 100
        progress = str('%.2f'%progress)
        sys.stdout.write('\r' + "Percent Complete: " + progress + "%")
        sys.stdout.flush()
    #calculate and return accuracy of data model
    accuracy = (numCorrect / total) * 100
    return str('%.2f'%accuracy)

#read file data
data = ['mnist_train_0_1.csv', 'mnist_test_0_1.csv']
trainLabels = []
testLabels = []
trainData = []
testData = []

print("Reading data from files...")
#read the training and testing data into memory
#appends the input data already normalized knowing that the minimum value is 0 and the maximum is 255
trainFile = open(data[0])
readFile = csv.reader(trainFile, delimiter = ',')
for row in readFile:
    rowData = []
    for i in range(len(row)):
        if i == 0:
            trainLabels.append(int(row[i]))
        else:
            rowData.append(int(row[i]) / 255)
    trainData.append(rowData)
            
testFile = open(data[1])
readFile = csv.reader(testFile, delimiter = ',')
for row in readFile:
    rowData = []
    for i in range(len(row)):
        if i == 0:
            testLabels.append(int(row[i]))
        else:
            rowData.append(int(row[i]) / 255)
    testData.append(rowData)
print("Done")

print("Training data model...")
#train the model
edgeList = trainModel(trainData, trainLabels)
print()
print("Done")


print("Testing Model...")
#test the model
accuracy = testModel(testData, testLabels, edgeList)
print()
print("Done")
print("Model Accuracy: " + accuracy + "%")