#!/usr/bin/env python
# coding: utf-8
'''
Genera nudos aleatorios de tres dimensiones a partir de la funciones:
x = A_x*cos(n_x*t + p_x)
y = A_y*cos(n_y*t + p_y)
z = A_z*cos(n_z*t + p_z)
El objetivo es obtener variedades en 3D con un género determinado.
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits import mplot3d

class knots:
	#Incializa la función
	#points: entero, número de puntos de la variedad
	#sc: entero, entero más pequeño que se obtiene en la distirbución
	#noise: flotante, parámetro para introducir ruido blanco
	def __init__(self, points=2000, sc=10, noise=0.0):
		super().__init__()
		self.points = int(points/2)
		self.sc = sc
		self.noise = noise
		self.knot = None

	#Genera un nudo aleatorio en base a los parámetros:
	#i: entero
	#j: entero
	#k: entero
	def create_random_knot(self,i,j,k):
	    #Valores de A en cada dimensión
	    Ax = np.random.rand(i)
	    Ay = np.random.rand(j)
	    Az = np.random.rand(k)

	    #Valores de n en cada dimensión
	    nx = np.random.randint(0,self.sc,size=i)
	    ny = np.random.randint(0,self.sc,size=j)
	    nz = np.random.randint(0,self.sc,size=k)

	    #Valores de p en cada dimensión
	    px = np.random.rand(i)
	    py = np.random.rand(j)
	    pz = np.random.rand(k)

	    #Función para generar cada dimensión del nudo
	    x = lambda t: np.array([Ax@np.cos(nx*t + px)])
	    y = lambda t: np.array([Ay@np.cos(ny*t + py)])
	    z = lambda t: np.array([Az@np.cos(nz*t + pz)])

	    #Dominio del nudo, círculo [-pi, pi]
	    S1 = np.linspace(-np.pi, np.pi, self.points)

	    #Genera las dimensiones del nudo aleatorio
	    x_t = np.array([x(t) for t in S1]).reshape(self.points)
	    y_t = np.array([y(t) for t in S1]).reshape(self.points)
	    z_t = np.array([z(t) for t in S1]).reshape(self.points)
	    
	    #Guarda el nudo
	    self.knot = np.vstack((x_t, y_t, z_t)).T
	    #Regresa las funciones que generan el nudo
	    return x,y,z


	#Genera un número de barras determinado en base al género
	#genus: entero > 0, determina el género de la variedad.
	def create_bars_random_knot(self,i,j,k,genus=1):
	    n = genus
	    #Obtiene un nudo aleatorio y ajusta sus dimensiones
	    x,y,z = self.create_random_knot(i,j,k)
	    x_t, y_t, z_t = self.knot[:,0],self.knot[:,1], self.knot[:,2]
	    x_t, y_t, z_t = x_t.reshape(self.points,1), y_t.reshape(self.points,1), z_t.reshape(self.points,1)
	    
	    #Genera las raíces de la unidad de acuerdo al género
	    #se generan los puntos antipodas para unir las barras
	    #Agrega ruido blanco para mover las raíces
	    points_pos = np.array([np.pi*k/n for k in range(0,n)]) + self.noise*np.random.normal(0.0,1.0, size=n)
	    points_neg = np.array([-np.pi*k/n for k in range(1,n+1)])[::-1] + self.noise*np.random.normal(0.0,1.0, size=n)
	    
	    #Aplica la función que genera los nudos a las raíces
	    pos_r_x = np.array([x(r) for r in points_pos])
	    pos_r_y = np.array([y(r) for r in points_pos])
	    pos_r_z = np.array([z(r) for r in points_pos])
	    neg_r_x = np.array([x(r) for r in points_neg])
	    neg_r_y = np.array([y(r) for r in points_neg])
	    neg_r_z = np.array([z(r) for r in points_neg])
	    
	    #Genera las barras a partir de la imagen de las raíces de la unidad
	    T = np.linspace(0,1,self.points)
	    bar_x= np.vstack([t*pos_r_x+(1-t)*neg_r_x for t in T])
	    bar_y= np.vstack([t*pos_r_y+(1-t)*neg_r_y for t in T])
	    bar_z= np.vstack([t*pos_r_z+(1-t)*neg_r_z for t in T])
	    #Une las barras al nudo para obtener una nueva variedad (nudo con barras)
	    mainfold_x, mainfold_y, mainfold_z = np.vstack((x_t,bar_x)), np.vstack((y_t,bar_y)), np.vstack((z_t,bar_z))
	    
	    #Guarda el nudo
	    self.knot = np.hstack((mainfold_x, mainfold_y, mainfold_z))
	    #return mainfold_x, mainfold_y, mainfold_z
	
	#Función para visualizar el nudo
	#color: color del nudo en el plot
	#bars_color: color de las barras en el plot
	def plot_knot(self,size=1,color='black',bars_color='black'):
		try:
			ax = plt.gca(projection='3d')
			ax.scatter(self.knot[:,0][:self.points], self.knot[:,1][:self.points], self.knot[:,2][:self.points],s=size,c=color)
			ax.scatter(self.knot[:,0][self.points:], self.knot[:,1][self.points:], self.knot[:,2][self.points:],s=size,c=bars_color)
			plt.show()
		except:
			raise Exception("No se ha generado el nudo")
