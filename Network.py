from Neuron import Neuron
from MyEnum import Activation
import random as rnd

class Network:
    layers : list[list[Neuron]] = []        #layer della rete

    def __init__(self) -> None:
        pass

    #Inizializzazione di un nuovo layer e impostazione dei collegamenti
    def Initialize(self) -> None:
        self.LayersReset()

        if len(self.layers) > 1:
            for i in range(len(self.layers)):               #sistemo un layeer alla volta
                if i == 0:                                  #Il primo layer carica solo il layer successivo
                    for unit in self.layers[i]:
                        unit.LayerNext = self.layers[i+1]

                elif i == (len(self.layers)-1):             #Il layer finale solo il precedente
                    for unit in self.layers[i]:
                        unit.LayerPrev = self.layers[i-1]

                else:                                       #I layers intermedi sia il precedente che il successivo
                    for unit in self.layers[i]:
                        unit.LayerPrev = self.layers[i-1]
                        unit.LayerNext = self.layers[i+1]

    #Reset delle connessioni tra layer
    def LayersReset(self) -> None:
        for layer in self.layers:
            for unit in layer:
                unit.LayerPrev = None
                unit.LayerNext = None

    #evaluete sample 
    def EvaluateSample(self, expectation : int) -> bool:
        for layer in self.layers[1:]:
            self.ResetEvaluate(layer=layer)     #reset del layer che vado ad valutare in questo momento 
            for neuron in layer:    #Calcolo i valori pesati dei neuroni
                neuron.calc_sum()
            for neuron in layer:    #Calcolo i valori di attivazione del neurone
                neuron.calc_activation(layer)

        #Controllo il risultato della valutazione
        evaluation : int
        maxValue : float = 0.0
        for unit in self.layers[len(self.layers)-1]:
            if unit.activation > maxValue:
                maxValue = unit.activation
                evaluation = unit.index
        
        return True if int(expectation) == int(evaluation) else False
            
    #Retropropagazione e apprendimento della rete
    def Backpropagation(self, expectation : int, learnRation : float) -> None:
            for i in range(len(self.layers)-1,0,-1):        #parto dall'ultimo layer fino al secondo (Il primo è il layer di input)
                for unit in self.layers[i]:
                    if i == (len(self.layers)-1):
                        unit.calc_error_output(expectation)
                    else:
                        unit.calc_error()

                    unit.update(learnRation)    #Aggiornamento dei pesi

    #Add layer in network
    def Add(self,activationType : Activation,nUnit : int):
        if len(self.layers) > 0:
            nLayers : int = len(self.layers)-1
            prevUnit = len(self.layers[nLayers]) #Numero di unità nel'ultimo layer presentee nella rete
        else:
            prevUnit = 0

        #layer = [Neuron(len(self.layers),_,activation,0,[format(rnd.uniform(-0.5,0.5),".3f") for _ in range(prevUnit)]) for _ in range(nUnit)]
        self.layers.append([Neuron(len(self.layers),_,activationType,0,[float(format(rnd.uniform(-0.5,0.5),".3f")) for _ in range(prevUnit)]) for _ in range(nUnit)])
        self.Initialize()

    #Reset dei valori di valutazione di campione 
    def ResetEvaluate(self, layer : list[Neuron]) -> None:
        for unit in layer:
            unit.reset()




    
