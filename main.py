import numpy as np
import pandas as pd

df = pd.read_csv('icgtp1datos/train.csv', header=None)
dfTest = pd.read_csv('icgtp1datos/test.csv', header=None)
#dfTest = df.iloc[800:999, :]
#df = df.iloc[0:799, :]

# Hacemos dinamico las entradas
X = df.iloc[:, 0:-1]
yd = df.iloc[:, -1]
xTest = dfTest.iloc[:, 0:-1]
ydTest = dfTest.iloc[:, -1]

X = X.to_numpy()
yd = yd.to_numpy()
xTest = xTest.to_numpy()
ydTest = ydTest.to_numpy()

# 1 Genero un vector de pesos aleatorios entre -0.5 y 0.5
tamX = len(X)
tamAux = len(X[0])
tamxTest = len(xTest)
W = []
columnAux = (np.full((tamX, 1), -1))
columnAuxTest = (np.full((tamxTest, 1), -1))
X = np.hstack((columnAux, X))
aux = X[0][:]
xTest = np.hstack((columnAuxTest, xTest))

for i in range(0, tamAux + 1):
    W.insert(i, np.random.uniform(-0.5, 0.5))
y = []  # Y calculado
epocas = 0
promError = 1  # Promedio de errores (inicializamos en 1 para que no joda)
umbralError = 0.05  # El porcentaje de errores que tuvo en una epoca debe ser menor al 5%
maxEpocas = 20000
historialW = np.zeros(shape=(maxEpocas, 3))
j = 0
# Recorro todas las pruebas
while (epocas < maxEpocas and promError > umbralError):  # Si antes de haber completado las epocas, el error es ya es muy chico, lo corto
    for i in range(0, tamX):
        yaux = np.dot(W, X[i][:])  # Calculo el y[i]
        # Funcion signo
        if yaux > 0:
            y.insert(i, 1)
        else:
            y.insert(i, -1)
        # Actualizar pesos
        W = W + (0.01 / 2) * (yd[i] - y[i]) * X[i][:]
    historialW[j] = [W[0], W[1], W[2]]
    j = j + 1

    # Comparo las salidas y calculo la cantidad de errores en una epoca
    y = []
    error = 0
    for i in range(0, tamxTest):
        yaux = np.dot(W, xTest[i][:])
        if yaux > 0:
            y.insert(i, 1)
        else:
            y.insert(i, -1)
        if y[i] != ydTest[i]:
            error = error + 1

    promError = error / tamxTest
    # Sumar 1 epoca
    print("epocas:", epocas, " - promerror: ", promError)
    epocas = epocas + 1
    y = []

historialW = historialW[0:j]
print(W)