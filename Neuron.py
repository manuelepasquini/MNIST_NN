from typing import List
from MyEnum import Activation
import MyMath
#from Neuron import Neuron

class Neuron:
    sum : float = None                  #Somma pesata degli input per i pesi
    activation : float = None           #Valore di attivazione del neurone
    activationType : Activation = None  #Tipo di attivazione del neurone
    weights : list[float] = []          #lista dei pesi delle connessioni del layer precedente
    bias : float = None                 #Bias del neurone
    error : float = None                #Errore del neurone
    layer : int = None                  #Layer index 
    index : int = None                  #index del Neurone
    LayerPrev :  []       #Lista dei neuroni del layer precedente connessi al neurone 
    LayerNext : []        #Lista dei neuroni del layer successivo connessi al neurone
    

    #definizione del costruttore
    def __init__(self,layer: int,index : int, activationType : Activation,bias : float,weights : list[float]):
        self.layer = layer
        self.index = index
        self.bias = bias
        self.weights = weights
        self.activationType = activationType
        
    def set_activation(self, value: float):
        self.activation = value

    #calcolo della somma pesata tra input del layer successivo per il peso delle connessione
    def calc_sum(self):
        somma : float = 0.0
        for neuron in self.LayerPrev:
            somma += float(neuron.activation) * float(self.weights[neuron.index])       #moltiplico il valore dell'output precedente con il peso della connessione 

        somma += self.bias      #Aggiungo Bias (iperparametro)
        self.sum = somma
    
    #Calcolo attivazione del neurone
    def calc_activation(self,actLayer : list):
        if self.activationType == Activation.Relu:
            self.activation = MyMath.Relu(self.sum)
        elif self.activationType == Activation.Sigmoid:
            self.activation = MyMath.Sigmoid(self.sum)
        elif self.activationType == Activation.SoftMax:
            layer_values : list[float] = [] #Lista del valore della somma pesata degli input dei neuroni del layer
            position : int          #Posizione del valore per cui calcolare funzione softmax
            for neuron in actLayer:
                layer_values.append(neuron.sum)
                if self.index == neuron.index:
                    position = neuron.index
            
            self.activation = MyMath.Softmax(position,layer_values)

    #Calcolo dell'errore del neurone
    def calc_error(self):
        e : float = 0.0
        for neuron in self.LayerNext:
            e += neuron.error * self.weights[neuron.index]

        if self.activationType == Activation.Sigmoid:
            self.error = MyMath.SigmoidDerivate(self.activation) * e
        elif self.activationType == Activation.Relu:
            self.error = MyMath.ReluDerivate(self.activation) * e
    
    #calcolo dell'errore del nerone in caso sia un neurone di output
    def calc_error_output(self,trueLabel : int):
        expected = 1 if trueLabel == self.index else 0
        if self.activationType == Activation.Sigmoid:
            self.error = MyMath.SigmoidDerivate(self.activation) * (expected - self.activation)
        elif self.activationType == Activation.SoftMax:
            self.error = MyMath.SoftmaxDerivate(self.activation) * (expected - self.activation)

    #Aggiornamento dei pesi della rete 
    def update(self,learningRate : float):
        self.bias += self.error * learningRate

        for neuron in self.LayerPrev:
            self.weights[neuron.index] += neuron.activation * self.error * learningRate

    #Reset i valori di esempio. Mantieni i pesi/bias
    def reset(self):
        self.sum = None
        self.activation = None
        self.error = None




