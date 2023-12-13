from Network import Network,Neuron
from MyEnum import Activation
import random as rnd

#carico i dati di test
with open("DataSet/mnist_test.csv","r") as file:
    evaluation = file.readlines()

#carico i dati di allenamento per la rete
with open("DataSet/mnist_train.csv","r") as file:
    train = file.readlines()

#Configurazione della rete
network = Network()
network.Add(None,784)
network.Add(activationType=Activation.Relu, nUnit = 15)
network.Add(activationType=Activation.SoftMax, nUnit = 10)

#Train della rete
ratio = 0.01
epochs = 3
okCount = 0
nonOkCount = 0
counterReset = 59000

for i in range(epochs):
    for i in range(len(train)):
        if i > 0 :
            trainerSample = train[i].split(",")
            label : int = int(trainerSample[0])      #Eticheetta corretta per il campione in fase di train
            trainerSample = trainerSample[1:]

            #Carico gli input da DataSet train
            for x in range(len(trainerSample)):
                network.layers[0][x].set_activation(float(trainerSample[x])/255)
            
            #Feed-forward e valutazione 
            if network.EvaluateSample(label) == True:
                okCount +=1 
            else:
                nonOkCount += 1

            #Retropropagazione con aggiornamento dei pesi
            network.Backpropagation(label, ratio)
    ratio *= 0.8
    okCount=0
    nonOkCount=0

for i in range (len(evaluation)):
    if i > 0:
        evaluationSample = evaluation[i].split(",")
        label = int(evaluationSample[0])
        evaluationSample = evaluationSample[1:]

        #Carico gli input da DataSet test
        for x in range(len(evaluationSample)):
            network.layers[0][x].set_activation(float(evaluationSample[x])/255)
        
        #Feed-forward e valutazione 
        if network.EvaluateSample(label) == True:
            okCount +=1 
        else:
            nonOkCount += 1

            
print("Accuracy:",format(okCount/(okCount+nonOkCount)*100,".2f"),"% | ",okCount+nonOkCount, "samples performed")           