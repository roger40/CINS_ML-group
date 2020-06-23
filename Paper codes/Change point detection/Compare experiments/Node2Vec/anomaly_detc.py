import os
import re

from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score

class data_loader:
	"""docstring for data_loader"""
	def __init__(self, file_path):
		self.file_path = file_path

	def load_data(self):
		file_list = os.listdir(self.file_path)
		graph_list = []
		for i in range(len(file_list)):
			file_name = '%s/graph%d.txt' % (self.file_path, i)
			fileObject = open(file_name, 'r')
			str = fileObject.read()
			str_list = re.split('{|:|"', str)
			str_list = list(filter(None, str_list))
			# if i==0:
			# 	print(str_list)
			num_node = int(str_list[-2])
			# print(num_node)
			graph_dic = {}
			for j in range(num_node):
				key = int(str_list[2*j])
				# print(type(key))
				value = []
				str_d_list = re.split(r'[ },\]\[]', str_list[2*j+1])
				str_d_list = list(filter(None, str_d_list))
				for k in range(len(str_d_list)):
					value.append(float(str_d_list[k]))
				graph_dic[key] = value
			graph_list.append(graph_dic)

		graph_all = []
		# print(graph_list[0][99])
		for i in range(len(graph_list)):
			graph_list_all = []
			for temp in graph_list[i].items():
				key, value = temp
				for j in range(len(value)):
					graph_list_all.append(value[j])
			graph_all.append(graph_list_all)

		return graph_all

def cluster_dbscan(file_path):
	# file_path = './embedding_results/results_simulate_100n'
	dl = data_loader(file_path)
	graph_list = dl.load_data()
	# print(len(graph_list))
	ep = 2
	ms = 2
	y_pred = DBSCAN(eps=ep, min_samples=ms).fit_predict(graph_list)
	print("eps=", ep)
	print("min_samples=", ms)
	print("y_pred:", y_pred)
	anomal = []
	for i in range(len(y_pred)-1):
		if y_pred[i+1] != y_pred[i]:
			anomal.append(i+1)

	print("anomal:", anomal)


def cluster_spectral(file_path):
	dl = data_loader(file_path)
	graph_list = dl.load_data()

	true_list = [7, 12, 15, 16, 17, 20, 21]
	true_anomal = []
	for i in range(len(graph_list)):
		if i in true_list:
			true_anomal.append(1)
		else:
			true_anomal.append(0)

	y_pred = SpectralClustering(n_clusters=3, gamma=0.1).fit_predict(graph_list)

	print("y_pred", y_pred)
	anomal = []
	anomal_list = []
	anomal_list.append(0)
	for i in range(len(y_pred)-1):
		if y_pred[i+1] != y_pred[i]:
			anomal.append(i+1)
			anomal_list.append(1)
		else:
			anomal_list.append(0)
	print("anomal:", anomal)

	f1 = f1_score(true_anomal, anomal_list, average='binary')
	print("f1_score:", f1)


if __name__ == '__main__':
	file_path = './embedding_results/results_simulate_24g'
	# cluster_dbscan(file_path)
	cluster_spectral(file_path)
