import numpy as np
import pandas as pd

def test(dfTest,W):
    xTest = dfTest.iloc[:, 0:-1]
    ydTest = dfTest.iloc[:, -1]
    tamxTest = len(xTest)

    # Columna -1
    columnAux = (np.full((tamxTest, 1), -1))
    xTest = np.hstack((columnAux, xTest))

    #Comparo las salidas y calculo la cantidad de errores en una epoca
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
    return promError