import string
from qiskit import *
import numpy as np
from scipy.optimize import minimize

from Utilities import start_or_retrieve_job

class QAOABase:

    def __init__(self):#,params = None):
        """
        init function that initializes member variables
        :param params: additional parameters
        """
        self.E=None
        self.Var=None
        self.current_depth=0 # depth at which local optimization has been done
        self.angles_hist={}
        self.costval={}
        self.gamma_grid=None
        self.beta_grid=None

        self.g_it=0
        self.g_values={}
        self.g_angles={}

################################
# functions to be implemented:
################################

    def cost(self, str, params):
        """
        implements the cost function
        :param string: a binary string
        :param params: additional parameters
        :return: a scalar value
        """

        raise NotImplementedError

    def createCircuit(self, angles, depth, params={}):
        """
        implements a function to create the circuit
        :param params: additional parameters
        :return: an instance of the qiskti class QuantumCircuit
        """
        raise NotImplementedError

################################
# generic functions
################################

    def loss(self, angles, backend, depth, shots, noisemodel=None, params={}):
        """
        loss function
        :param params: additional parameters
        :return: an instance of the qiskit class QuantumCircuit
        """
        self.g_it+=1
        circuit = []
        circuit.append(self.createCircuit(angles, depth, params=params))

        if backend.configuration().local:
            job = execute(circuit, backend=backend, noisemodel=noisemodel, shots=shots)
        else:
            job = start_or_retrieve_job('name'+"_"+str('opt_iterations'), backend, circuit, options={'shots' : shots}) # job = start_or_retrieve_job(name+"_"+str(opt_iterations), backend, circuit, options={'shots' : shots}) 

        e,v = self.measurementStatistics(job, params=params)

        self.g_values[str(self.g_it)] = -e
        self.g_angles[str(self.g_it)] = angles.copy()

        return -e

    def measurementStatistics(self, job, params):
        """
        implements a function for expectation value and variance
        :param job: job instance derived from BaseJob
        :return: expectation and variance
        """
        jres=job.result()
        counts_list=jres.get_counts()
        if isinstance(counts_list, list): # checks the list is not empty
            expectations = []
            variances = []
            for i, counts in enumerate(counts_list):
                n_shots = jres.results[i].shots
                E = 0
                E2 = 0
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    cost = self.cost(string[::-1], params)
                    E += cost*counts[string]
                    E2 += cost**2*counts[string]
                if n_shots == 1:
                    v = 0
                else:
                    E/=n_shots
                    E2/=n_shots
                    v = (E2-E**2)*n_shots/(n_shots-1)
                expectations.append(E)
                variances.append(v)
            return expectations, variances
        else:
            n_shots = jres.results[0].shots
            E = 0
            E2 = 0
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                cost = self.cost(string[::-1], params)
                E += cost*counts_list[string]
                E2 += cost**2*counts_list[string]
            if n_shots == 1:
                v = 0
            else:
                E/=n_shots
                E2/=n_shots
                v = (E2-E**2)*n_shots/(n_shots-1)
            return E,v

    def hist(self, angles, backend, shots, noisemodel=None, params={}):
        depth=int(len(angles)/2)
        circ=self.createCircuit(angles, depth, params=params)
        if backend.configuration().local:
            job = execute(circ, backend, shots=shots)
        else:
            job = start_or_retrieve_job("hist", backend, circ, options={'shots' : shots})
        return job.result().get_counts()

    def random_init(self, gamma_bounds, beta_bounds, depth):
        """
        Enforces the bounds of gamma and beta based on the graph type.
        :param gamma_bounds: Parameter bound tuple (min,max) for gamma
        :param beta_bounds: Parameter bound tuple (min,max) for beta
        :return: np.array on the form (gamma_1, beta_1, gamma_2, ...., gamma_d, beta_d)
        """
        if(len(gamma_bounds) > 1):
            gamma_list = np.random.uniform(gamma_bounds[0],gamma_bounds[1], size=depth)
        else:
            gamma_list = gamma_bounds[0]

        if(len(gamma_bounds) > 1):
            beta_list = np.random.uniform(beta_bounds[0],beta_bounds[1], size=depth)
        else:
            beta_list = beta_bounds[0]

        initial = np.empty((np.size(gamma_list) + np.size(beta_list))) # np.empty((gamma_list.size + beta_list.size,), dtype=gamma_list.dtype)
        initial[0::2] = gamma_list
        initial[1::2] = beta_list

        # gamma_list = np.random.uniform(gamma_bounds[0],gamma_bounds[1], size=depth)
        # beta_list = np.random.uniform(beta_bounds[0],beta_bounds[1], size=depth)
        # initial = np.empty((gamma_list.size + beta_list.size,), dtype=gamma_list.dtype)
        # initial[0::2] = gamma_list
        # initial[1::2] = beta_list

        return initial


    def interp(self, angles):
        """
        INTERP heuristic/linear interpolation for initial parameters
        when going from depth p to p+1 (https://doi.org/10.1103/PhysRevX.10.021067)
        E.g. [0., 2., 3., 6., 11., 0.] becomes [2., 2.75, 4.5, 7.25, 11.]
        :param angles: angles for depth p
        :return: linear interpolation of angles for depth p+1
        """
        depth=len(angles)
        tmp=np.zeros(len(angles)+2)
        tmp[1:-1]=angles.copy()
        w=np.arange(0,depth+1)
        return w/depth*tmp[:-1] + (depth-w)/depth*tmp[1:]

    def sample_cost_landscape(self, backend, shots, noisemodel=None, params={}, verbose=True, angles={"gamma": [0,2*np.pi,20], "beta": [0,2*np.pi,20]}):
        if verbose:
            print("Calculating Energy landscape for depth p=1...")

        depth=1

        tmp=angles["gamma"]
        self.gamma_grid = np.linspace(tmp[0],tmp[1],tmp[2])
        tmp=angles["beta"]
        self.beta_grid = np.linspace(tmp[0],tmp[1],tmp[2])

        if backend.configuration().local:
            circuits=[]
            for beta in self.beta_grid:
                for gamma in self.gamma_grid:
                    #params['name'] = str(beta_n)+"_"+str(gamma_n)
                    circuits.append(self.createCircuit(np.array((gamma,beta)), depth, params=params))
            job = execute(circuits, backend, shots=shots)
            e, v = self.measurementStatistics(job, params=params)
            self.E = -np.array(e).reshape(angles["beta"][2],angles["gamma"][2])
            self.Var = np.array(v).reshape(angles["beta"][2],angles["gamma"][2])
        else:
            self.E = np.zeros((angles["beta"][2],angles["gamma"][2]))
            self.Var = np.zeros((angles["beta"][2],angles["gamma"][2]))
            b=-1
            for beta in self.beta_grid:
                b+=1
                g=-1
                for gamma in self.gamma_grid:
                    g+=1
                    params['name'] = str(b)+"_"+str(g)
                    circuit = self.createCircuit(np.array((gamma,beta)), depth, params=params)  # createCircuit(np.array((gamma,beta)), depth, params=params)
                    job = start_or_retrieve_job('name'+"_"+str(b)+"_"+str(g), backend, circuit, options={'shots' : shots})  # job = start_or_retrieve_job(name+"_"+str(b)+"_"+str(g), backend, circuit, options={'shots' : shots})
                    e,v = self.measurementStatistics(job, params=params)
                    self.E[b,g] = -e[0]
                    self.Var[b,g] = -v[0]

        #self.current_depth=1
        if verbose:
            print("Calculating Energy landscape done")

    def get_current_deptgh(self):
        return self.current_depth

    def local_opt(self, angles0, backend, shots, depth, noisemodel=None, params={}, method='Powell'): 
        """
        :param angles0: initial guess
        """
        bounds =()
        for j in range(depth):
            bounds += ((0,2*np.pi),(0, np.pi))
            
        depth=int(len(angles0)/2)

        res = minimize(self.loss, x0 = angles0, method = method,
                       args=(backend, depth, shots, noisemodel, params), bounds= bounds)

        return res.x

    def increase_depth(self, backend, shots, noisemodel=None, params={}, method='Nelder-Mead'):

        repeats=1

        if self.current_depth == 0:
            if self.E is None:
                self.sample_cost_landscape(backend, shots, noisemodel=noisemodel, params=params, angles={"gamma": [0,2*np.pi,20], "beta": [0,2*np.pi,20]})
            ind_Emin = np.unravel_index(np.argmin(self.E, axis=None), self.E.shape)
            angles0=np.array((self.gamma_grid[ind_Emin[1]], self.beta_grid[ind_Emin[0]]))
            self.angles_hist['d1_initial']=angles0
        else:
            gamma=self.angles_hist['d'+str(self.current_depth)+'_final'][::2]
            beta=self.angles_hist['d'+str(self.current_depth)+'_final'][1::2]
            gamma_interp=self.interp(gamma)
            beta_interp=self.interp(beta)
            angles0=np.zeros(2*(self.current_depth+1))
            angles0[::2]=gamma_interp
            angles0[1::2]=beta_interp
            self.angles_hist['d'+str(self.current_depth+1)+'_initial']=angles0

        self.g_it=0
        self.g_values={}
        self.g_angles={}

        #best_res=None
        for rep in range(repeats):
            if rep>0:
                gamma_bounds=(0,np.pi)
                beta_bounds=(0,np.pi)
                angles0 = self.random_init(gamma_bounds, beta_bounds, self.current_depth+1)
            res = self.local_opt(angles0, backend, shots, noisemodel=noisemodel, params=params, method=method)
            print("rep=",rep, ":", res.fun)

        ind = min(self.g_values, key=self.g_values.get)
        self.angles_hist['d'+str(self.current_depth+1)+'_final']=self.g_angles[ind]
        self.costval['d'+str(self.current_depth+1)+'_final']=self.g_values[ind]
        #if not res.success:
        #    raise Warning("Local optimization was not successful.", res)


        self.current_depth+=1