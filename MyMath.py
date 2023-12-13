import math

#definizione della funzione Relu x se x>0; 0 altrimenti
def Relu(x:float):
    return max(0,x)

#Funzione sigmoide. Mappa un valore dato tra 0 e 1
def Sigmoid(x:float):
    return 1 / (1 + math.pow(math.e,-x))

#funzione softmax. Mappa le probabilità per ogni valore data una lista di valori 
def Softmax(index : int, values : list[float]):
    dividend : float = 1*math.pow(10,-15)   #Annulla la divisione per 0
    for i in range(len(values)):
        dividend += math.pow(math.e,values[i])
    
    return math.pow(math.e,values[index])/dividend

#clacola la derivata della funzione Relu. l'input che passo è il valore di attivazione della funzione Relu
def ReluDerivate(x : float):
    return 1 if x > 0 else 0

#Calcola la derivata della funzione Sigmoide
def SigmoidDerivate(x : float):
    return x * (1 - x)

#calcola la derivata della funzione Softmax
def SoftmaxDerivate(x : float):
    return x * (1 - x)