import pandas as pd
from ValidacionCruzada import validacionCruzada
from particion import particion
import numpy as np

archivo = pd.read_csv('icgtp1datos/spheres1d10.csv', header=None)
# archivo = pd.read_csv('icgtp1datos/spheres2d10.csv', header=None)
# archivo = pd.read_csv('icgtp1datos/spheres2d50.csv', header=None)
# archivo = pd.read_csv('icgtp1datos/spheres2d70.csv', header=None)

size = len(archivo) # Cantidad de pruebas
n = 5 # Particiones de entrenamiento
porcTrn = 80 # Relacion (training)

indices = validacionCruzada(size, n, porcTrn)
errores = particion(indices, archivo)

media = np.mean(errores)
desvio = np.std(errores)

print("Media:", media, " Desvio:", desvio)
print(errores) # Cantidad de errores por particion