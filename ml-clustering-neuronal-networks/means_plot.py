import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import utils
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


def pruebita(dataset):

	#Histograma que muestra la distribucion de ideologias segun variables
	#dataset.drop(['ideology'],1).hist()
	#plt.show()

	#Graficos que relacionan atributos pasado por parametro en vars, diferenciando las ideologias con colores
	sb.pairplot(dataset, hue='ideology',height=4,vars=['race', 'religion'],kind='scatter')
	plt.show()

	sb.pairplot(dataset, hue='ideology',height=4,vars=['income', 'education'],kind='scatter')
	plt.show()

	sb.pairplot(dataset, hue='ideology',height=4,vars=['income', 'education', 'age-group'],kind='scatter')
	plt.show()

	#Distribucion de income, education y age group segun ideologias en 3D
	#Se selecciono income, education y age grupo de forma aleatoria
	X = np.array(dataset[['income', 'education', 'age-group']])
	y = np.array(dataset['ideology'])
	X.shape
	fig = plt.figure()
	ax = Axes3D(fig)
	colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
	asignar=[]
	for row in y:
		asignar.append(colores[int(row)])
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
	plt.show()


