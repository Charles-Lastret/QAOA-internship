from datetime import datetime
from qiskit import *
import qiskit as qk
from qiskit.providers.aer import QasmSimulator
import networkx as nx
import numpy as np
import math
from Utilities import getNumberOfControlledGates
from QAOAMaxKCutBinary_original import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import json as js
import os.path
from tqdm import tqdm 


###################################### Tests utilities ###############################################################################

class Saver(): 


    def __init__(self, folder):
        self.input_folder = os.path.join(os.getcwd(),folder)
        os.mkdir(self.input_folder)
        self.input_path = self.input_folder + '/results.json'
        self.file = open(self.input_path, 'w')
        # print("input_path =", self.input_path)
        
    def save(self, input_dict):
        with open(self.input_path) as fp:  
            listObj = []
            listObj.append(input_dict)
            fp.close()
        with open(self.input_path, 'w') as json_file:
            js.dump(listObj, json_file, indent=4)#,   separators=(',',': '))
            json_file.close()

                    

def approx_ratio(k_cuts):
    alpha= 0
    a = np.array([.0, .0, .878567, .836008, .857487, .876610, .891543, .903259, .912664, .920367, .926642], dtype=float)
    if(2 <= k_cuts <= 10):
        alpha= a[k_cuts]
    else:
        raise ValueError('k > 10')
    return alpha


def maximum_cost(k_cuts, cost): # problème : obtenir les valeurs de coût de GW !
    assert(2<= k_cuts <= 10)
    max_cost = cost/approx_ratio(k_cuts)
    return abs(max_cost)


def number_of_CX_gates(Quantum_circuit):
    n = getNumberOfControlledGates(Quantum_circuit)[0]
    return n
    


###################################### Graph generators for tests ########################################################################


def one_clique_graph(nb_nodes, clique_size, with_random_weights =False): 
    assert(nb_nodes >= clique_size and clique_size > 1)

    while True:
        G =nx.complete_graph(clique_size)
        for i in range(clique_size, nb_nodes):
            G.add_node(i)
            G.add_edge(i, np.random.randint(0, high= clique_size, size=None, dtype=int))
        if with_random_weights:
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = np.random.random()
        else:
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = 1
        break

    return G

# Creates a graph with a priscribed number of cliques with fixed number of nodes per clique. Connected, no self-node edge
def many_clique_graph(nb_nodes, nb_cliques, clique_size, with_random_weights =False):
    assert(nb_nodes >= clique_size * nb_cliques)
    assert(clique_size > 1)
    
    while True:
        G= nx.empty_graph()
        for k in range(nb_cliques):
            G = nx.disjoint_union(G,nx.complete_graph(clique_size))
            for j in range(clique_size): # Spans clique nodes
                for i in range(k*clique_size + j): # Spans nodes of the k-th clique graph 
                    if(k*clique_size + j - 1 > 0):
                        a = np.random.randint(0, high= k*clique_size + j - 1, size=None, dtype=int)
                        if(i != a):
                            G.add_edge(i, a)
        for s in range(clique_size * nb_cliques, nb_nodes):
            G.add_node(s)
            if(s>1):
                rand= np.random.randint(0, high= s - 1, size=None, dtype=int)
                H = G
                H.add_edge(s, rand)
                max = list(nx.enumerate_all_cliques(H))[-1]
                if(len(max) < clique_size):
                    G = H
        if with_random_weights:
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = np.random.random()
        else:
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = 1
        break

    return G

###################################### Parameter tested : Clique size ##############################################################


def test_clique_size(G, k_cuts, shots =1000, depth= 1, noisemodel= None):
    alpha = 0
    output_dict= {}

    test = QAOAMaxKCutOnehot() 
    params = {'G' : G, 'k_cuts' : k_cuts}
    gamma_bounds = 2*np.pi*np.array([1 for i in range(depth)])
    beta_bounds= np.pi*np.array([1 for i in range(depth)])
    simulator = QasmSimulator()

    angles = test.random_init(gamma_bounds,beta_bounds,depth) # angles = test.interp(angles) à corriger
    angles = test.local_opt(angles, backend= simulator, shots= 100, depth =depth, noisemodel=noisemodel, params= params, method='Powell')
   
    C = test.createCircuit(angles, depth, params)
    cx = number_of_CX_gates(C)
    job = qk.execute(C,simulator, shots=shots)
    loss = test.loss(angles, backend = simulator, depth= depth, shots = shots, noisemodel=noisemodel, params= params)
    
    meas_stat = test.measurementStatistics(job, params)
    alpha = meas_stat[0]/maximum_cost(k_cuts, loss)
    output_dict ={'alpha': alpha, 'angles': angles.tolist(), 'cost': loss, 'cx': cx}

    return output_dict



def simulation_clique_size(nb_nodes, clique_size_range, k_cuts_range, r, noisemodel, with_random_weights):
    assert(1 <= r)
    folder = f"scs_{nb_nodes}_{clique_size_range }_{k_cuts_range }_{r}_{noisemodel}_{with_random_weights}_{datetime.now()}/"
    print(folder)
    saver= Saver(folder)
    res_dict = {}
    for i in clique_size_range:
        G = one_clique_graph(nb_nodes, i, with_random_weights = with_random_weights)
        name = f"clique de taille {i}"
        graph_path = os.path.join(folder, f"{name}" + ".gml")
        nx.write_gml(G, graph_path)
        
        for k in tqdm(k_cuts_range):
            aux_dict = test_clique_size(G ,k_cuts = k, shots= 100, depth =r, noisemodel=noisemodel)
            aux_dict.update({'k':k})
            aux_dict.update({'graph_path': graph_path})
            res_dict[name + f", k={k}"] = aux_dict
            saver.save(res_dict)



###################################### Parameter tested : Number of cliques ###########################################################


def test_number_of_cliques(G, k_cuts, shots=1000 , depth = 1, noisemodel= None):

    output_dict= {}

    test = QAOAMaxKCutOnehot()
    params = {'G' : G, 'k_cuts' : k_cuts}
    gamma_bounds = 2*np.pi*np.array([1 for i in range(depth)])
    beta_bounds= np.pi*np.array([1 for i in range(depth)])
    simulator = QasmSimulator()

    angles = test.random_init(gamma_bounds,beta_bounds,depth) # angles = test.interp(angles) à corriger
    angles = test.local_opt(angles, backend= simulator, shots= 100, depth =depth, noisemodel=noisemodel, params= params, method='Powell')
   
    C = test.createCircuit(angles, depth, params)
    job = qk.execute(C,simulator, shots=shots)
    cx = number_of_CX_gates(C)
    loss = test.loss(angles, backend = simulator, depth= depth, shots = shots, noisemodel=noisemodel, params= params)
    
    meas_stat = test.measurementStatistics(job, params)
    alpha = meas_stat[0]/maximum_cost(k_cuts, loss)
    output_dict ={'angles': angles.tolist(), 'cost': loss, 'cx': cx}

    return output_dict



def simulation_number_of_cliques(G, nb_nodes, nb_cliques, k_cuts_range, r, clique_size, noisemodel, with_random_weights):
    
    folder = f"snc_nb_nodes{nb_nodes}_nb_cliques{nb_cliques}_k{k_cuts_range }_clique_size{clique_size}_{datetime.now()}/"
    print(folder)
    saver= Saver(folder)
    res_dict = {}
    name = f"nombre de cliques = {nb_cliques}"
    for k in tqdm(k_cuts_range):
            aux_dict = test_number_of_cliques(G ,k_cuts = k, shots= 1000, depth =r, noisemodel=noisemodel)
            aux_dict.update({'k':k})
            res_dict[name + f", k={k}"] = aux_dict
            saver.save(res_dict)


def test_erdos_renyi(k_cuts_range):
    folder = f"Erdos-Renyi_k{k_cuts_range}_{datetime.now()}/"
    saver= Saver(folder)
    res_dict = {}
    G=nx.read_gml("Erdos-Renyi.gml")
    for k in tqdm(k_cuts_range):
            aux_dict = test_number_of_cliques(G ,k_cuts = k, shots= 1000, depth =1, noisemodel=None)
            aux_dict.update({'k':k})
            res_dict[f"k={k}"] = aux_dict
            saver.save(res_dict)
    return 



