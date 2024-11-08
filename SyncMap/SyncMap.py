import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import pickle


class SyncMap:

	def __init__(self, input_size, dimensions, adaptation_rate):

		self.organized = False
		self.space_size = 10
		self.dimensions = dimensions
		self.input_size = input_size
		self.syncmap = np.random.rand(input_size, dimensions)
		self.adaptation_rate = adaptation_rate
		self.std_devs = np.std(self.syncmap, axis=0)

	def inputGeneral(self, x):
		plus = x > 0.1
		minus = ~ plus

		sequence_size = x.shape[0]
		for i in range(sequence_size):

			vplus = plus[i, :]
			vminus = minus[i, :]
			plus_mass = vplus.sum()
			minus_mass = vminus.sum()

			if plus_mass <= 1:
				continue

			if minus_mass <= 1:
				continue
			center_plus = np.dot(vplus, self.syncmap) / plus_mass
			center_minus = np.dot(vminus, self.syncmap) / minus_mass

			dist_plus = distance.cdist(center_plus[None, :] / self.std_devs, self.syncmap / self.std_devs, 'euclidean')
			dist_minus = distance.cdist(center_minus[None, :] / self.std_devs, self.syncmap / self.std_devs, 'euclidean')

			dist_plus = np.transpose(dist_plus)
			dist_minus = np.transpose(dist_minus)
			update_plus = vplus[:, np.newaxis]*((center_plus - self.syncmap)/dist_plus)
			update_minus = vminus[:, np.newaxis]*((center_minus - self.syncmap)/dist_minus)
			update = update_plus - update_minus

			self.syncmap += self.adaptation_rate * update

			maximum = self.syncmap.max()
			self.syncmap = self.space_size*self.syncmap/maximum

	def input(self, x):

		self.inputGeneral(x)

		return

		print(x.shape)
		plus = x > 0.1
		minus = ~ plus
		sequence_size = x.shape[0]
		for i in range(sequence_size):
			vplus = plus[i, :]
			vminus = minus[i, :]
			plus_mass = vplus.sum()
			minus_mass = vminus.sum()
			if plus_mass <= 1:
				continue

			if minus_mass <= 1:
				continue
			center_plus = np.dot(vplus, self.syncmap) / plus_mass

			center_minus = np.dot(vminus, self.syncmap) / minus_mass
			update_plus = vplus[:, np.newaxis]*(center_plus - center_minus)
			update_minus = vminus[:, np.newaxis]*(center_minus - center_plus)
			update = update_plus + update_minus
			self.syncmap += self.adaptation_rate * update

			maximum = self.syncmap.max()
			self.syncmap = self.space_size*self.syncmap / maximum

	def organize(self):

		self.organized = True
		self.labels = DBSCAN(eps=3, min_samples=2).fit_predict(self.syncmap)

		return self.labels

	def activate(self, x):
		'''
		Return the label of the index with maximum input value
		'''

		if self.organized == False:
			print("Activating a non-organized SyncMap")
			return

		# maximum output
		max_index = np.argmax(x)

		return self.labels[max_index]

	def plotSequence(self, input_sequence, input_class,filename="plot.png"):

		input_sequence = input_sequence[1:500]
		input_class = input_class[1:500]

		a = np.asarray(input_class)
		t = [i for i, value in enumerate(a)]
		c = [self.activate(x) for x in input_sequence]

		plt.plot(t, a, '-g')
		plt.plot(t, c, '-.k')

		plt.savefig(filename, quality=1, dpi=300)
		plt.show()
		plt.close()

	def plot(self, color=None, save=False, filename="plot_map.png"):

		if color is None:
			color = self.labels

		print(self.syncmap)
		if self.dimensions == 2:
			ax = plt.scatter(self.syncmap[:, 0], self.syncmap[:, 1], c=color)

		if self.dimensions == 3:
			fig = plt.figure()
			ax = plt.axes(projection='3d')

			ax.scatter3D(self.syncmap[:, 0], self.syncmap[:, 1], self.syncmap[:, 2], c=color)

		if save == True:
			plt.savefig(filename)

		plt.show()
		plt.close()

	def save(self, filename):
		"""save class as self.name.txt"""
		file = open(filename+'.txt', 'w')
		file.write(pickle.dumps(self.__dict__))
		file.close()

	def load(self, filename):
		"""try load self.name.txt"""
		file = open(filename+'.txt', 'r')
		dataPickle = file.read()
		file.close()

		self.__dict__ = pickle.loads(dataPickle)


