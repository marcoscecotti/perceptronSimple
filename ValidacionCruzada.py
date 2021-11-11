import pandas as pd
import numpy as np

# Size: Cantidad de pruebas
# n: Particiones de entrenamiento
# porcTrn: porcentaje que tomamos para training (80% => 80)

def validacionCruzada(size, n, porcTrn):
    indices = []
    for i in range(n):
        # randomizo
        vecrnd = np.random.choice(size, size, replace=False)
        nTrn = int(porcTrn * size / 100) # Obtengo el n para corte
        vectrn = vecrnd[0:nTrn]
        vectst = vecrnd[(nTrn):]
        dataset = [vectrn, vectst]
        indices.insert(i, dataset)
    return indices

