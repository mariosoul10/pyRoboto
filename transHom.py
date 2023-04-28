import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotTH(THs, axis = [0, 10, 0, 10]):
	"""
	Esta funcion grafica una serie de transformaciones homogeneas 4*4
	de una lista THs. totalTH es la matriz que vamos a aplicar las th sucesivas, 
	mientras que p guardara las coordenadas (x,y,z) de cada th para poder
	graficarlas
	"""

	#Creamos la figura que utilizaremos para graficar en 3d
	ax = plt.figure().add_subplot(projection='3d')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	#Creamos las variables que vamos a utilizar 
	n = len(THs)
	totalTH = np.eye(4)										
	x = np.zeros(n+1)
	y = np.zeros(n+1)
	z = np.zeros(n+1)

	#Multiplicamos las th sucesivamente y guardamos las coordenadas del vector de posicon p 
	for i in range(n):
		totalTH = np.dot(totalTH, THs[i])
		x[i+1] = totalTH[0][3]
		y[i+1] = totalTH[1][3]
		z[i+1] = totalTH[2][3]

		for j in range(3):	
			xr = totalTH[0][j]
			yr = totalTH[1][j]
			zr = totalTH[2][j]
			xp = totalTH[0][3]
			yp = totalTH[1][3] 
			zp = totalTH[2][3] 

			ax.plot([xp, xp + xr], [yp, yp + yr], [zp, zp +zr], 'o:r',	alpha=0.3)

	#Graficamos
	ax.plot(x, y, z, color = 'r', marker='o', linewidth = '5')
	plt.axis(axis)
	plt.show()
	return 

def plotPose(T):
	'''
	Esta funcion grafica una pose dada una matriz de transformacion homogenea T de 4*4. Para esto utiliza
	las columnas de la matriz R dentro de la th, cada columna es un vector de longitud 1 apuntando en la direccion
	de los componentes de la pose.
	'''

	ax = plt.figure().add_subplot(projection='3d')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	for i in range(3):	
		xr = T[0][i]
		yr = T[1][i]
		zr = T[2][i]
		xp = T[0][3]
		yp = T[1][3] 
		zp = T[2][3] 

		ax.plot([xp, xp + xr], [yp, yp + yr], [zp, zp +zr], 'o:r',	alpha=0.3)
	plt.axis([0, 10, 0, 10])
	plt.show()
	return

def cinDir():
	"""Optional docstring."""
	return

def cinInv():
	"""Optional docstring."""
	return

def trasl(x, y, z):
	"""Esta funcion recibe las coordenas de una traslacion (x,y,x)
	y regresa una matriz homogenea con la traslacion aplicada* """

	tr = np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])
	
	return tr

def rotx(theta, tipo = 'rad'):
	"""Esta funcion regresa la matriz de transformacion
	homogenea 4*4 de una rotacion en el eje x dado un angulo
	theta. Utiliza radianes por defecto, pero se pueden dar el
	angulo en grados"""

	if tipo == 'grados':
		theta = theta*math.pi/180

	ct = math.cos(theta)
	st = math.sin(theta)
	rx = np.array([[1, 0, 0, 0],[0, ct, -st, 0],[0, st, ct, 0],[0, 0, 0, 1]])

	return rx

def roty(theta, tipo = 'rad'):
	"""Esta funcion regresa la matriz de transformacion
	homogenea 4*4 de una rotacion en el eje y dado un angulo
	theta. Utiliza radianes por defecto, pero se pueden dar el
	angulo en grados"""

	if tipo == 'grados':
		theta = theta*math.pi/180

	ct = math.cos(theta)
	st = math.sin(theta)
	ry = np.array([[ct, 0, st, 0],[0, 1, 0, 0],[-st, 0, ct, 0],[0, 0, 0, 1]])
	
	return ry

def rotz(theta, tipo = 'rad'):
	"""Esta funcion regresa la matriz de transformacion
	homogenea 4*4 de una rotacion en el eje z dado un angulo
	theta. Utiliza radianes por defecto, pero se pueden dar el
	angulo en grados"""

	if tipo == 'grados':
		theta = theta*math.pi/180

	ct = math.cos(theta)
	st = math.sin(theta)
	rz = np.array([[ct, -st, 0, 0],[st, ct, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

	return rz

def DH(theta, d, a, alpha):
	"""Crea una matriz de transformacion homogenea 4*4
	a partir de los parametros Denavit Hartenberg theta, d, a, alpha 
	y el tipo de eslabon revolucion(R) o prismatico(P)
	
	"""
	ct = math.cos(theta)
	st = math.sin(theta)
	ca = math.cos(alpha)
	sa = math.sin(alpha)
	A = np.array([[ct, -1*st*ca, st*sa, a*ct],[st, ct*ca, -1*ct*sa, alpha*st],[0, sa, ca, d],[0, 0, 0, 1]])

	return A

def puma():
	'''
	Esta funcion regresa una lista con los parametros dh contantes
	y una lista con los parametros tipo de variable R para revolucion
	y P para prismatico 
	'''

	return


T = trasl(0, 0, 2) 
B = trasl(-2, 0, 0)
H = rotx(45,'grados')
G = roty(45, 'grados')


#plotPose(H)
plotTH([T, np.dot(B,G), T, H], [-5, 5, -5, 5])


# q = [0, -pi/4, -pi/4, pi/8, 0]

# cinDir(puma, q)