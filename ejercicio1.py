import pandas as pd
from perceptron import perceptronSimple
from test import test
from graficar import graficar

#1-a

# Entrenamos el perceptron con el OR
df = pd.read_csv('icgtp1datos/OR_trn.csv', header=None)
historialWTrain, promErrorTrain = perceptronSimple(df)

# Testeamos el perceptron con el OR
dfTest = pd.read_csv('icgtp1datos/OR_tst.csv', header=None)
tamAux = len(historialWTrain)
promErrorTest = test(dfTest, historialWTrain[tamAux-1][:]) # Pasamos el df y el último W

##Comparamos los errores promedios -> El error promedio del test deberia ser bajo para poder decir que se cumple la generalizacion
print("Error promedio final de entrenamiento", promErrorTrain)
print("Error promedio de test", promErrorTest)

#-------------------------------------------------------------------------------------------------------------------------------------#

##Entrenamos el perceptron con el XOR
# dfXOR = pd.read_csv('icgtp1datos/XOR_trn.csv', header=None)
# historialWTrain, promErrorTrain = perceptronSimple(dfXOR)
#
# ##Testeamos el perceptron con el XOR
# dfTestXOR = pd.read_csv('icgtp1datos/XOR_tst.csv', header=None)
# tamAux = len(historialWTrain)
# promErrorTest = test(dfTestXOR, historialWTrain[tamAux-1][:])
#
# ##Comparamos los errores promedios -> El error promedio del test deberia ser bajo para poder decir que se cumple la generalizacion
# print("Error promedio final de entrenamiento", promErrorTrain)
# print("Error promedio de test", promErrorTest)

#--------------------------------------------------------------------------------------------------------------------------------------#

#1-b: Gráficas
#Grafica del OR
graficar(df, "OR")
#
# # #Grafica del XOR
# graficar(dfXOR, "XOR")

