from knots_generator import knots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pickle import dump, load
from knots_generator import knots

#Función de normalización
f = lambda m : m / (np.linalg.norm(m)+1e-100)

#Función para generarl circulos en un punto
def addTube(U, V, radius=1, segments=10):
	#Bases normal y binormal
	Ux, Uy, Uz = U[0], U[1],U[2]
	Vx, Vy, Vz = V[0], V[1],V[2]

	tube = []
	for i in range(segments):
		#Determina los tubso
		θ = 2*np.pi*i/segments  # theta
		dx = radius*(np.cos(θ)*Ux + np.sin(θ)*Vx)
		dy = radius*(np.cos(θ)*Uy + np.sin(θ)*Vy)
		dz = radius*(np.cos(θ)*Uz + np.sin(θ)*Vz)
		#Add tube
		tube.append([dx,dy,dz])
		#tube.append([dx+0.1,dy+0.1,dz+0.1])
		#tube.append([dx+0.2,dy+0.2,dz+0.2])
		
	return(np.array(tube))

class knot_points_cloud:
	"""
	Genera la nube de puntos a partir de un nudo.
	Args:
		points (numpy array): puntos del nudo
	"""
	def __init__(self): #points
		super().__init__()
		self.points = None #points
		#Objeto de nudo
		self.knot = None
		#derivadas
		self.dX, self.ddX = None, None
		#Marco de senet-ferret
		self.T, self.N, self.B = None, None, None
		#numero de puntos
		self.n = len(points)
		#Alcance
		self.r = 0.001
		#Nudo gordo
		self.fat_knot = None
		
	def get_points(self,npoints=1000,genus=1,noise=0.1,i=1,j=1,k=2):
		"Función para generar un punto"
		knot_generator = knots(points=npoints, noise=noise, sc=6)
		knot_generator.create_bars_random_knot(i,j,k,genus=genus)
		self.knot = knot_generator
		self.points = knot_generator.knot

	def frenet_serret(self):
		"Función para generar el marco de Frenet-Serret desde una nube de puntos (parametrizada)"
		# Calculate the first and second derivative of the points
		self.dX = np.apply_along_axis(np.gradient, axis=0, arr=self.points)
		self.ddX = np.apply_along_axis(np.gradient, axis=0, arr=self.dX)
		#Calcula la tangente normalizando dX
		self.T = np.apply_along_axis(f, axis=1, arr=self.dX)
		#Calcula y normaliza binormal
		B = np.cross(self.dX, self.ddX)
		self.B = np.apply_along_axis(f, axis=1, arr=B)
		#Cálcula y normaliza vector normal
		N = np.cross(self.B, self.T)
		self.N = np.apply_along_axis(f, axis=1, arr=N)
		
	def plot_frenet_serret(self,arrow_length=0.01,size=(10,10)):
		"Function to plot the Frenet Serret frame"
		#Genera los axis
		fig = plt.figure(figsize=size)
		ax = fig.add_subplot(111,projection='3d')
		#Plotea los puntos del nudo
		ax.plot(self.points[:,0], self.points[:,1], self.points[:,2], 'o', markersize=10, color='g', alpha=0.5)
		ax = fig.gca(projection='3d')
		
		#Función para plotear marcos
		def get_arrow(vector,center,c='black'):  
			#Meshea los puntos
			x, y, z = np.meshgrid(vector[0],vector[1],vector[2])
			#Meshea los centros de las flechas
			x_c, y_c, z_c = np.meshgrid(center[0],center[1],center[2])
			# Make the direction data for the arrows
			u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
			v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
			w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z))  
			#Plotea los puntos centrados en los puntos
			ax.quiver(x_c, y_c, z_c, u, v, w, length=arrow_length,normalize=True,color=c)

		#Plotea cada marco centrado en el punto del nudo
		for i,x in enumerate(self.points):
		  get_arrow(self.B[i],x, c='blue')
		  get_arrow(self.N[i],x, c='red')
		  get_arrow(self.T[i],x)
		plt.show()
		
	def get_radius(self):
		"Función para calcular alcance"
		import open3d as o3d
		#Puntos en formato o3d
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(self.points)
		#Estioma las normales
		pcd.estimate_normals()
		#Calcula las distancias
		distances = pcd.compute_nearest_neighbor_distance()
		#Calcula el reach (alcance)
		self.r = np.max(distances)/2
		
	def create_fat_knot(self, num_circ=10):
		"Función para añadir circulos a los nudos"
		#Chrea marco de frenet-serret
		self.frenet_serret()
		
		#Obtiene el alcance
		self.get_radius()
			
		#Acumula los círculos
		fatten_knot = []
		for i,x in enumerate(self.points):
			#Crea circulos con marco frenet-serret
			circ = addTube(self.N[i], self.B[i], self.r, num_circ) + x
			#Junta los círuclos
			fatten_knot.append(list(circ))
			
		#Crea el nudo engordado
		self.fat_knot = np.array(fatten_knot).reshape(self.n*num_circ,3)
		
	def plot_fat_knot(self, size=(10,10), c='red'):
		"función para plotear los puntos del nudo gordo"
		#Crear 3d plot
		fig = plt.figure(figsize=size)
		ax = fig.add_subplot(111, projection='3d')
		#Poner puntos del nudo flaco
		ax.plot(self.points[:,0], self.points[:,1], self.points[:,2], 'o', markersize=2, color='g', alpha=0.5)
		#Si no hay nudo gord
		if type(self.fat_knot) == 'NoneType':
			pass
		else:
			#Plotear el nudo gordo
			ax.plot(self.fat_knot[:,0], self.fat_knot[:,1], self.fat_knot[:,2], 'o', markersize=2, color=c, alpha=1)
		
		plt.show()
		
	def save(self, path):
		knot_object = (self.fat_knot, self.points,self.n,self.r,self.dX,self.ddX,self.T,self.N,self.B)
		fname = open(path,'wb')
		dump(knot_object,fname)
		fname.close()
		
	def load(self,path):
		fname = open(path,'rb')
		self.fat_knot, self.points, self.n, self.r, self.dX, self.ddX, self.T, self.N, self.B = load(fname)
		fname.close()
		
