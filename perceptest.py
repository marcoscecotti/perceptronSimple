import numpy as np
import pandas as pd

def perceptest(df,dfTest):

    # Hacemos dinamico las entradas
    X = df.iloc[:, 0:-1]
    yd = df.iloc[:, -1]
    xTest = dfTest.iloc[:, 0:-1]
    ydTest = dfTest.iloc[:, -1]

    X = X.to_numpy()
    yd = yd.to_numpy()
    xTest = xTest.to_numpy()
    ydTest = ydTest.to_numpy()

    tamX = len(X)
    tamAux = len(X[0])
    tamxTest = len(xTest)

    #Le agrego la entrada -1 correspondiente al BIAS
    columnAux = (np.full((tamX,1), -1))
    X = np.hstack((columnAux, X))

    columnAuxTest = (np.full((tamxTest, 1), -1))
    xTest = np.hstack((columnAuxTest, xTest))

    # Genero un vector de pesos aleatorios entre -0.5 y 0.5
    W = []
    for i in range(0, tamAux+1):
        W.insert(i, np.random.uniform(-0.5, 0.5))
    y = [] # Y calculado

    # Condiciones de corte
    epocas = 0
    promError = 1 # Promedio de errores (inicializamos en 1 para que no joda)
    umbralError = 0 # El porcentaje de errores que tuvo en una epoca debe ser menor al 5%
    maxEpocas = 100

    historialW = np.zeros(shape=((maxEpocas*tamX)+1,len(W)))
    historialW[0] = W
    j=1
    # Recorro todas las pruebas
    while (epocas < maxEpocas and promError > umbralError): # Si antes de haber completado las epocas, el error es ya es muy chico, lo corto
        for i in range(0, tamX):
            yaux = np.dot(W, X[i][:])   #Calculo el y[i]
            #Funcion signo
            if yaux > 0:
                y.insert(i, 1)
            else:
                y.insert(i, -1)
            #Actualizar pesos
            auxSuma = 0.1*(yd[i]-y[i])*X[i][:]
            W = W + auxSuma
            historialW[j] = W
            j=j+1

        #Comparo las salidas y calculo la cantidad de errores en una epoca
        y = []
        error = 0
        for i in range(0, tamX):
            yaux = np.dot(W, X[i][:])
            if yaux > 0:
                y.insert(i, 1)
            else:
                y.insert(i, -1)
            if y[i] != yd[i]:
                error = error + 1

        promErrorTrain = error / tamX
        #Sumar 1 epoca
        #print("epocas:", epocas, " - promerror: ", promError)
        epocas = epocas + 1
        y = []
    historialW = historialW[0:j]
    #print(W)

    # Comparo las salidas y calculo la cantidad de errores en una epoca
    tamAuxW = len(historialW)
    W = historialW[tamAuxW-1]
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

    return error