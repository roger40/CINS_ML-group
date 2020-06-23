import matplotlib.pyplot as plt

import networkx as nx
import pickle
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import time
import scipy.io as scio
#from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import SpectralClustering

#import helper libraries
from dynamicgem.utils      import graph_util, plot_util, dataprep_util

"""
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
"""
#import the methods
from dynamicgem.embedding.ae_static    import AE
'''
from dynamicgem.embedding.dynamicTriad import dynamicTriad
from dynamicgem.embedding.TIMERS       import TIMERS
from dynamicgem.embedding.dynAE        import DynAE
from dynamicgem.embedding.dynRNN       import DynRNN
from dynamicgem.embedding.dynAERNN     import DynAERNN
'''


# output directory for result
outdir = './2019-7-8'#'./zttOutput'
intr='./intermediate'
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(intr):
    os.mkdir(intr)  
	

#Generate the dynamic graph
#testDataType = 'enron'
#file = "../data/enron.mtx"
#num_nodes = 147
#500epoch-Time:280.62259244918823
#300epoch-Time:238.98669481277466
#testDataType = 'toyF'#'mit'
#file = "./zttInput/DynGemToy.mtx"#DynGemMit.mtx"
#num_nodes =9 #94

#testDataType = 'mit'
#file = "../data/mit.mtx"
#num_nodes =94
#500epoch-Time:291.62789845466614
#300epoch0Time:248.606543302536

#Simulate
#testDataType = 'simulate24'
#file = '../data/simulate24.mtx'
#num_nodes = 100
#500epoch-Time: 119.90309453010559
#300epoch-Time:97.93286418914795

#testDataType = 'simulate'
#file = '../data/simulate.mtx'
#num_nodes = 100
#500epoch-Time:23.38923954963684
#300epoch-Time:18.28303813934326

#testDataType = 'sbm'
#file = './sbm_10g_1000n_0.1i_0.01o.mtx'
#num_nodes = 1000

#Senate  array
testDataType = 'senate'
file = '../data/senate.mtx'
num_nodes = 222
#300epoch-Time:41.04161334037781

#SBM
#testDataType = 'sbm10g_50n'
#file = './SBM_10g_50n.mtx'
#num_nodes = 50
#testDataType = 'sbm30g_100n'
#file = './SBM_30g_100n.mtx'
#num_nodes = 100

#testDataType = 'sbm50g_100n'
#file = './SBM_50g_100n.mtx'
#num_nodes = 100



samples = scio.mmread(file)#.todense()
num_sample = len(samples)
print(num_sample)
graphs = []
for index in range(num_sample):
    sample = samples[index]
    sample.shape = (num_nodes, num_nodes)
    graphs.append(nx.from_numpy_matrix(sample))


# parameters for the dynamic embedding
# dimension of the embedding
#dim_emb  = 20
#dim_emb = 10
# lookback = 2

dim_emb = 20
#dim_emb = 64 #simulate

testDataType = testDataType + str(dim_emb)
if not os.path.exists(os.path.join(outdir, testDataType)):
    os.mkdir(os.path.join(outdir, testDataType))

#AE Static
# xeta is learning ratio
embedding = AE(d            = dim_emb, 
                 beta       = 5, 
                 nu1        = 1e-6, 
                 nu2        = 1e-6,
                 K          = 3,   # hidden_layer_num
                 n_units    = [500, 300, ],  
                 n_iter     = 300, #500 # epoch
                 xeta       = 1e-3, #1e-4, #1e-6, # lr
                 n_batch    = num_nodes, #100,
                 modelfile  = ['./intermediate/enc_modelsbm.json',
                             './intermediate/dec_modelsbm.json'],
                 weightfile = ['./intermediate/enc_weightssbm.hdf5',
                             './intermediate/dec_weightssbm.hdf5'])
embs  = []
t1 = time.time()
error= []
#outdir = outdir+testDataType
#ae static
for temp_var in range(num_sample):
    print('time :', temp_var)
#  try:
    emb, _= embedding.learn_embeddings(graphs[temp_var])
    if np.inf in emb or np.nan in emb:
       print('there is error!!!')
    #np.savetxt(outdir+'/'+testDataType+'/' +str(temp_var)+'.txt', emb)
    #emb.shape = (1, num_nodes*dim_emb)
    #embs.append(list(emb)[0])
 # except:
  #  print(temp_var)
  #  error.append(temp_var)
#np.savetxt('./output/matrix/error.txt', error)
#print(np.array(embs).shape)
t2 = time.time()
print('###############TIME##############:')
print(t2-t1)

#with open(outdir, 'w') as fp:
#    for emb in embs:
#        for e in emb:
#            fp.write(str(e)+"\t")
#        fp.write('\n')


'''
#y_pred = DBSCAN(eps = 1e-6).fit_predict(embs)
y_pred = cluster.s_cluster(np.array(embs))
print("anomaly:\n", y_pred)
for i in range(num_sample-1):
    t = i + 1
    if y_pred[t-1] != y_pred[t]:
        print(t)
'''
l = int(np.sqrt(2*num_sample))
anomal = []

for n in range(2, l+1):
 for g in [0.01, 0.1, 1, 10]:
   
   y_pred = SpectralClustering(n_clusters=n, gamma=g).fit_predict(embs)
   print(n, g, "prediction:\n", y_pred)
   an = []
   for i in range(1, len(y_pred)):
     t = i
     if y_pred[t-1] != y_pred[t]:
        an.append(t)
   print("anomaly: ", an)
   anomal.append(an)
#np.savetxt(outdir+'/'+testDataType+'_pred'+'.txt', anomal)

