import numpy as np
import math
import matplotlib.pyplot as plt


def plotTH(THs, axis = [0, 20, 0, 20]):
	"""
	Esta funcion grafica una serie de transformaciones homogeneas 3*3
	de una lista THs. totalTH es la matriz que vamos a aplicar las th sucesivas, 
	mientras que p guardara las coordenadas (x,y,z) de cada th para poder
	graficarlas
	"""

	#Creamos la figura que utilizaremos para graficar en 3d
	fig, ax = plt.subplots()
	plt.xlabel('eje x')
	plt.ylabel('eje y')
	#Creamos las variables que vamos a utilizar 
	n = len(THs)
	totalTH = np.eye(3)
	# El origen con zeros								
	x = np.zeros(n+1)
	y = np.zeros(n+1)
	colors = ['r', 'b', ..., 'w']

	#Multiplicamos los eslabones sucesivamente y guardamos las coordenadas del vector de posicon p 
	for i in range(n):
		totalTH = np.dot(totalTH, THs[i])
		x[i+1] = totalTH[0][2]
		y[i+1] = totalTH[1][2]

		for j in range(2):	
			xr = totalTH[0][j]
			yr = totalTH[1][j]
			
			xp = totalTH[0][2]
			yp = totalTH[1][2] 
		
			ax.plot([xp, xp + xr], [yp, yp + yr], 'o', alpha=0.3)


	#Graficamos
	ax.plot(x, y, color = 'r', marker='o', linewidth = '5')
	plt.axis(axis);
	plt.show()
	return 

def plotPose(T, axis = [ -5, 10, -5, 10]):
	'''
	Esta funcion grafica una pose dada una matriz de transformacion homogenea T de 3*3. Para esto utiliza
	las columnas de la matriz R dentro de la th, cada columna es un vector de longitud 1 apuntando en la direccion
	de los componentes de la pose.
	'''
    #Configuramos la grafica
	plt.xlabel('eje x')
	plt.ylabel('eje y')
	plt.axis(axis)
    #Graficamos la pose
	for i in range(2):	
		xrot = T[0][i]
		yrot = T[1][i]

		x = T[0][2]
		y = T[1][2]  

		ax.plot([x, x + xrot], [y, y + yrot], 'o:r',	alpha=0.3)

	plt.show()
	return


def cinDir():
	"""Optional docstring."""
	return

def cinInv():
	"""Optional docstring."""
	return

def trasl(x, y):
	"""Esta funcion recibe las coordenas de una traslacion (x,y)
	y regresa una matriz homogenea con la traslacion aplicada* """

	tr = np.array([[1, 0, x],[0, 1, y],[0, 0, 1]])
	
	return tr

def rot(theta, tipo = 'rad'):
	"""Esta funcion regresa la matriz de transformacion
	homogenea 3*3 de una rotacion dado un angulo
	theta. Utiliza radianes por defecto, pero se pueden dar el
	angulo en grados"""

	if tipo == 'grados':
		theta = theta*math.pi/180

	ct = math.cos(theta)
	st = math.sin(theta)
	rx = np.array([[ct, -st, 0],[st, ct, 0], [0, 0, 1]])

	return rx



T = trasl(1, 5) 
B = trasl(0, 8)
H = rot(45,'grados')

print (np.dot(T,H))
#plotPose(np.dot(T,H))
plotTH([T, B, H])


# q = [0, -pi/4, -pi/4, pi/8, 0]

# cinDir(puma, q)