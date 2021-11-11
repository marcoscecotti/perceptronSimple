import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import perceptronSimple

def graficar(df, titulo):
    historialW, _ = perceptronSimple(df)

    def x2(x1,w0,w1,w2):
        return (w0/w2) - (w1/w2)*x1

    x1 = range(-3,3)

    plt.title(titulo)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.axhline(0,color="black")
    plt.axvline(0,color="black")

    # 4 ©iruclos
    plt.plot(1,1,"ro")
    plt.plot(-1,1,"ro")
    plt.plot(1,-1,"ro")
    plt.plot(-1,-1,"ro")

    for i in range(0,len(historialW),2):
        w0 = historialW[i][0]
        w1 = historialW[i][1]
        w2 = historialW[i][2]
        plt.plot(x1, x2(x1, w0, w1, w2))
        plt.pause(0.5)

    plt.show()