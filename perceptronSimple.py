import numpy as np

def perceptronSimple(df):

    # Hacemos dinamico las entradas
    X = df.iloc[:, 0:-1]
    yd = df.iloc[:, -1]

    X = X.to_numpy()
    yd = yd.to_numpy()

    tamX = len(X)
    tamAux = len(X[0])

    #Le agrego la entrada -1 correspondiente al BIAS (umbral / parcialidad)
    columnAux = (np.full((tamX,1), -1))
    X = np.hstack((columnAux, X)) # reagrupamos

    # Genero un vector de pesos aleatorios entre -0.5 y 0.5
    W = []
    for i in range(0, tamAux+1):
        W.insert(i, np.random.uniform(-0.5, 0.5))
    y = [] # Y calculado

    # Condiciones de corte
    epocas = 0
    tasaAcierto = 1 # Promedio de errores (inicializamos en 1 para que no joda)
    tolerancia = 95 # El porcentaje de errores que tuvo en una epoca debe ser menor al 5%
    maxEpocas = 100

    # Armamos un historial
    historialW = np.zeros(shape=((maxEpocas * tamX) + 1, len(W)))
    historialW[0] = W
    j = 1 # Contador del historialW

    # Recorro todas las pruebas
    while (epocas < maxEpocas and tasaAcierto < tolerancia):
        for i in range(0, tamX): # Cantidad de pruebas
            yaux = np.dot(W, X[i][:])   #Calculo el y[i]
            # Funcion signo - función de activación
            if yaux > 0:
                y.insert(i, 1)
            else:
                y.insert(i, -1)

            #Actualizar pesos
            W = W + 0.1*(yd[i]-y[i])*X[i][:]

            # Agregamos al historial
            historialW[j] = W
            j = j+ 1

        # Comparo las salidas y calculo la cantidad de errores en una epoca
        y = [] # Reinicio las salidas calculadas
        acierto = 0
        for i in range(0, tamX): # Cantidad de pruebas
            yaux = np.dot(W, X[i][:])
            if yaux > 0:
                y.insert(i, 1)
            else:
                y.insert(i, -1)
            if y[i] == yd[i]:
                acierto+=1

        tasaAcierto = (acierto / tamX)*100

        # Sumar 1 epoca
        epocas = epocas + 1
        print("epocas:", epocas, " - Tasa de acierto: ", tasaAcierto)

        # reinicio salidas por época
        y = []

    historialW = historialW[0:j]
    return historialW, tasaAcierto
