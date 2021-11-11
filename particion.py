from perceptest import perceptest

def particion(indices, df):
    arrayErrores = []
    for i in range(len(indices)): # Cantidad de particiones
        train = df.iloc[indices[i][0], :]
        test = df.iloc[indices[i][1], :]
        error = perceptest(train, test) # Cantidad de errores (Perceptron + test)
        arrayErrores.insert(i, error)
    return arrayErrores