'''
This code generates a sparse support vector machine for classification (for hyperspectral images)
'''

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split # to split the dataset
from numpy.random import seed, rand           # generate random floating point values
from numpy import array                 # to use norm
from numpy.linalg import norm
from scipy import sparse as sp
import time                           # time evaluation
import copy
import json
import random


class SSVM():
    def __init__(self, Lambda = 0.5, beta = 1.96, delta = 1.96, eps = 1.0e-1, seedn=1,
                verbose=False, balanced=False):
        self.Lambda = Lambda
        self.beta = beta
        self.delta = delta
        self.eps = eps
        self.seed = seedn
        self.verbose=verbose
        self.balanced = balanced
    
    def fit(self, X, y, do_continue = False, track = False):
        #I made a wrapper function so that we can implement 1-v-1 and 1-v-rest
        #1-v-rest is the initial default
        self.n_groups = len(set(y))
        if self.n_groups>2:
            #1-v-rest is the initial default
            self.svectors = {}
            if track:
                self.tracks = {}
            self.groups = set(y)
            for group in self.groups:
                print("training group {}".format(group))
                is_group = y==group
                print("{} in group".format(is_group.sum()))
                ya = np.zeros_like(y, dtype=np.int16)
                ya[:] = -1
                ya[is_group] = 1
                
                if self.balanced:
                    indx = np.arange(len(ya)) 
                    ui = indx[ya<1]
                    random.shuffle(ui)
                    keep = np.zeros_like(ya)
                    keep[ya==1] = 1
                    for j in range((ya==1).sum()):
                        keep[ui[j]] = 1
                    keep[ya==1] = 1
                else:
                    keep = np.zeros_like(ya)
                if do_continue:
                    self.x = self.svectors[group]
                print((np.array(ya)[keep==1]==1).sum(),(np.array(ya)[keep==1]==-1).sum())
                print((np.array(ya)==1).sum(),(np.array(ya)==-1).sum())
#                 print(np.array(ya)[keep==1]==1,np.array(ya)[keep==1]==-1)
                '''
                print((ya==1).sum(),(ya==-1).sum(),(ya==0).sum(),(keep==1).sum(),(keep==-1).sum(),(keep==0).sum())
                ya vector has 1's for the given class and rest are -1's, 
                while keep vector has all 0's...then why do we need keep vector?
                '''
#                 self._fit_1(np.array(X)[keep==1], np.array(ya)[keep==1], do_continue = do_continue, track = track)
                self._fit_1(np.array(X), np.array(ya), do_continue = do_continue, track = track)
                self.svectors[group] = copy.deepcopy(self.x)
                if track:
                    self.tracks[group] = copy.deepcopy(self.track_record)
        # keeping the 'else' for backwards compatibility
        else:
            self._fit_1(X,y)
    
    
    def _fit_1(self, X, y, do_continue = False, track = False):
        # at the top of the file, X_train (and X_test) are transposed, whereas
        # it seems more normal to input them into the algorithm as they are
        # we can adjust this as seems fitting
        
        # LP matrix ----AA,bb,c ---- from ---X,y,delta---
        pSVM, NSVM = X.transpose().shape
        self.pSVM = pSVM
        self.NSVM = NSVM
        c1 = np.array([np.ones(2*pSVM, dtype=int)])
        c2 = np.array([self.Lambda*np.ones(NSVM)])
        c3 = np.array([np.zeros(2, dtype = int)])
        c = np.concatenate((c1, c2, c3), axis=None)
        c = np.array([c])
        c = c.transpose()
        #print(c.shape)

        #     AA
        AA1 = -y.transpose()*X.transpose()
        AA2 = -np.identity(NSVM)
        AA3 = np.array([y])
        AA  = np.concatenate((AA1.transpose(), -AA1.transpose(),
                              AA2, -AA3.transpose(), AA3.transpose()), axis=1)
        #print(AA.shape)

        #     bb
        bb = np.array([-np.ones(NSVM)])
        bb = bb.transpose()
        #print(bb.shape)

        #Least-squares matrices A,b,C,d from LP(AA, bb, c)
        ptrain, Ntrain = AA.shape
        #print(ptrain, Ntrain)

        #     C
        C1 = np.concatenate((AA, np.zeros((ptrain,ptrain), dtype = 'int')), axis=1)     # int
        #C1sp = sp.lil_matrix(np.concatenate((AA, 
        #                                     np.zeros((ptrain,ptrain), dtype=np.float32)),
        #                                    axis=1))
        sum_of_rows_C1 = np.array(np.sum(np.abs(C1)**2, axis=-1))**(1./2)
        C1sp = sp.lil_matrix(C1 / sum_of_rows_C1[:, np.newaxis])
        C2 = np.concatenate((np.zeros((Ntrain,Ntrain), dtype=int),-AA.transpose()), axis=1)
        sum_of_rows_C2 = np.array(np.sum(np.abs(C2)**2, axis=-1))**(1./2)
        #C2sp = sp.lil_matrix(np.concatenate((np.zeros((Ntrain,Ntrain), dtype=int),
        #                                    -AA.transpose()), axis=1),
        #                                    dtype=np.float32)
        #C = np.concatenate((C1, C2), axis=0)
        C2sp = sp.lil_matrix(C2 / sum_of_rows_C2[:, np.newaxis])
        sum_of_rows_C = np.concatenate((sum_of_rows_C1, sum_of_rows_C2), axis=0)
        
        Csp = sp.vstack([C1sp, C2sp])
        #Csp = Csp.tocsr()
        #print(C.shape)

        #     d
        d = np.concatenate((bb, c), axis=0)
        #print(d.shape)

        #     A
        A = np.concatenate((c.transpose(), bb.transpose()), axis=1)
        #print(A.shape)

        #     b
        b = np.array([0])
        
        p, n = Csp.shape
        m, n = A.shape
        #print(p, n, m)

        # to normalize each rows of A
        sum_of_rows_A = np.sum(np.abs(A)**2, axis=-1)**(1./2)
        A = A / sum_of_rows_A[:, np.newaxis]
        b = b / sum_of_rows_A[:, np.newaxis]

        # to normalize each rows of C
        #sum_of_rows_C = np.array(np.sum(np.abs(Csp)**2, axis=-1))**(1./2)
        #print(Csp.shape, sum_of_rows_C.shape)
        #Csp = Csp.multiply(1 / sum_of_rows_C[:, np.newaxis])
        #print(Csp, Csp.shape)
        #Csp = Csp.transpose()
        d = d / sum_of_rows_C[:, np.newaxis]
        
        #Csp = sp.lil_matrix(C)
        Csp = Csp.tocsr()
        #print(Csp)
        
        seed(self.seed)
        # generate random numbers between 0-1
        
        # why is this called x? shouldn't it be called 'w' or something like that?
        if do_continue:
            x = self.x
        else:
            x = np.array([rand(n)])
            x = x.transpose() ## why is this '5' here?
            print(y)
            middle = (X[y==1].mean(axis=0)-X[y==-1].mean(axis=0))
            #print(middle)
            x[:self.pSVM,0] = np.maximum(middle, 0)
            x[self.pSVM:2*self.pSVM,0] = np.maximum(-middle, 0)
            x[2*self.pSVM+self.NSVM:2*self.pSVM+self.NSVM+2,0]=0
            x *= 5

        #print(x.shape)
        if track:
            track_record = []
            self.verbose = True # override default
        
        #    SSP-LS

        k=0
        Maximum = []
        h1 = np.dot(A, x) - b
        h2 = np.maximum(np.array(Csp.dot(x)) - d, 0)
        Max = norm(np.concatenate((h1, h2), axis=None))
        
        start = time.time()
        
        var = np.arange(p)
        np.random.shuffle(var)

        while Max > self.eps:
            #i = np.random.choice(m, size=1, replace=False)          # To choose row block of A
            #Axi = A[i, :]
            #bxi = b[i]
            r = A@x# - bxi

            # Optimality step
            if r==0:
                v=x
            else:
                #alpha = (self.delta*(r**2)) / (r*norm(Axi))**2
                v = x - self.delta*(A.transpose()*r)

            if k%p==0:
                np.random.shuffle(var)
            j = [var[k%p]]
            #print("j is {}".format(j))
            #j = np.random.choice(p, size=1, replace=False)         # To choose row block of C
            #print(Csp[j,:].todense())
            Cxi = np.array(Csp[j, :].todense())
            #print(Cxi.shape)
            #print(C[j,:])
            dxi = d[j[0]]
            #print(dxi.shape)
    
            # Feasibility step
            #print(x.shape)
            x = v - (self.beta*np.maximum(0, Cxi@v-dxi)*Cxi.transpose())#/(norm(Cxi)**2)
            #print(x.shape)
      
            # Projection Step 
            x = np.maximum(x, 0)

            # Stopping criteria
            if k%max(m,p) == 0:
                h1 = np.dot(A, x) - b
                h2 = np.maximum(np.array(Csp.dot(x)) - d, 0)
                Max = norm(np.concatenate((h1, h2), axis=None))
                if self.verbose:
                    rem_bands = (np.abs(x[:pSVM] - x[pSVM:2*pSVM])>0.1).sum()
                    print(Max, "{} remaining bands".format(rem_bands))
                Maximum = np.append(Maximum, Max)   
                if track:
                    entry = {}
                    entry['k'] = k
                    entry['Max'] = Max
                    entry['rem_bands'] = rem_bands
                    entry['pcorrect'] = (self._predict_gen(X, x)==y).sum()/len(y)
                    entry['l1'] = np.abs(x[:pSVM] - x[pSVM:2*pSVM]).sum()
                    track_record.append(entry)
                    print(entry['pcorrect'], (self._predict_gen(X, x)==1).sum()/len(y),
                         (y==1).sum()/len(y))
                    
        
        
            k=k+1 

    
        end = time.time()
        self.x = x
        self.duration = end - start
        self.epochs = len(Maximum)
        if track:
            self.track_record = track_record
        
    def predict(self, X):
        # the 1-v-rest prediction style
#         self.n_groups = len(set(y))
        
        if self.n_groups>2:
            #1-v-rest is the initial default
            self.prediction = -np.ones(X.shape[0])
            scores = np.zeros((self.n_groups, X.shape[0]))
            group_order = []
            for i, group in enumerate(self.svectors):
                group_order.append(group)
                self.x = self.svectors[group]
                scores[i] = self._predict(X)
            prediction = np.argmax(scores, axis=0)
            for i, j in enumerate(group_order):
                self.prediction[prediction==i] = j
        # keeping the 'else' for backwards compatibility
        else:
            self.prediction = self._predict_1(X)
        return self.prediction
    
    def _predict(self, X):
        pSVM = self.pSVM
        NSVM = self.NSVM
        dist = -(self.x[2*pSVM+NSVM]-self.x[2*pSVM+NSVM+1])
        labels = (X@(self.x[:pSVM] - self.x[pSVM:2*pSVM])-dist)#>=dist
        #labels = 2*labels - 1
        return labels[:,0]
        
    def _predict_1(self, X):
        
        '''old version'''
        pSVM = self.pSVM
        NSVM = self.NSVM
        dist = -(self.x[2*pSVM+NSVM]-self.x[2*pSVM+NSVM+1])
        labels = X@(self.x[:pSVM] - self.x[pSVM:2*pSVM])>=dist
        labels = 2*labels - 1
        return labels[:,0]
    
    def _predict_gen(self, X, sv):
        pSVM = self.pSVM
        NSVM = self.NSVM
        dist = -(sv[2*pSVM+NSVM]-sv[2*pSVM+NSVM+1])
        labels = X@(sv[:pSVM] - sv[pSVM:2*pSVM])-dist
        labels = 1*np.sign(labels)
        return labels[:,0]
                          
    def _predict_b(self, X, bound = 0.1):
        pSVM = self.pSVM
        NSVM = self.NSVM
        dist = -(self.x[2*pSVM+NSVM]-self.x[2*pSVM+NSVM+1])
        sv = self.x[:pSVM] - self.x[pSVM:2*pSVM]
        sel_bands = (np.abs(sv) > bound)[:,0]
        labels = X[:,sel_bands]@(sv[sel_bands])-dist#>=dist
        #labels = 2*labels - 1
        return labels[:,0]                      

    def predict_sparse(self, X, bound = 0.1):
        # the 1-v-rest prediction style
        if self.n_groups>2:
            #1-v-rest is the initial default
            self.prediction = -np.ones(X.shape[0])
            scores = np.zeros((self.n_groups, X.shape[0]))
            group_order = []
            for i, group in enumerate(self.svectors):
                group_order.append(group)
                self.x = self.svectors[group]
                scores[i] = self._predict_b(X, bound=bound)
            prediction = np.argmax(scores, axis=0)
            for i, j in enumerate(group_order):
                self.prediction[prediction==i] = j
        # keeping the 'else' for backwards compatibility
        else:
            self.prediction = self._predict_1(X)
        return self.prediction
    
    def _predict_ba(self, X, bands = 1):
        pSVM = self.pSVM
        NSVM = self.NSVM
        dist = -(self.x[2*pSVM+NSVM]-self.x[2*pSVM+NSVM+1])
        sv = self.x[:pSVM] - self.x[pSVM:2*pSVM]
        sel_bands = np.argsort(np.abs(sv).flatten())[-bands:]
        labels = X[:,sel_bands]@(sv[sel_bands])-dist#>=dist
        #abels = 2*labels - 1
        return labels[:,0]   
                          
    def predict_bands(self, X, bands = 1):
        # the 1-v-rest prediction style
        if self.n_groups>2:
            #1-v-rest is the initial default
            self.prediction = -np.ones(X.shape[0])
            scores = np.zeros((self.n_groups, X.shape[0]))
            group_order = []
            for i, group in enumerate(self.svectors):
                group_order.append(group)
                self.x = self.svectors[group]
                scores[i] = self._predict_ba(X, bands=bands)
            prediction = np.argmax(scores, axis=0)
            for i, j in enumerate(group_order):
                self.prediction[prediction==i] = j
        # keeping the 'else' for backwards compatibility
        else:
            self.prediction = self._predict_1(X)
        return self.prediction
                          
    def _w(self):
        pSVM = self.pSVM
        NSVM = self.NSVM
        if self.n_groups>2:
            return {group:(self.svectors[group][:pSVM] - self.svectors[group][pSVM:2*pSVM]) \
                    for group in self.svectors}
        else:
            return (self.x[:pSVM] - self.x[pSVM:2*pSVM])
        
    def save(self, filename):
        if filename[-4:]=='json':
            saving_dict = {}
            try:
                saving_dict['tracks'] = {str(i):self.tracks[i] for i in self.tracks}
            except AttributeError:
                print('no tracks to save')
            saving_dict['svectors'] = {str(i):self.svectors[i] for i in self.svectors}
            saving_dict['parameters'] = {
                    'Lambda' : self.Lambda,
                    'beta' : self.beta,
                    'delta' : self.delta,
                    'eps' : self.eps,
                    'seed' : self.seed,
                    'pSVM' : self.pSVM,
                    'NSVM' : self.NSVM,
                    'training_duration' : self.duration,
                    'epochs' : self.epochs
                }
            print(saving_dict)
            with open(filename, 'w') as outfile:
                json.dump(saving_dict, outfile, cls=NpEncoder)
        else:
            print('failed to save')
            
    def load(self, filename):
        with open(filename) as json_file:
            data = json.load(json_file)
        try:
            self.tracks = {int(i):data['tracks'][i] for i in data['tracks']}
        except KeyError:
            print('no tracks found')
        self.svectors = {}

        for v in data['svectors']:
            self.svectors[int(v)]= np.array(data['svectors'][v]).reshape((-1,1))
        self.n_groups = len(self.svectors)
            
        self.Lambda = data['parameters']['Lambda']
        self.beta = data['parameters']['beta']
        self.delta = data['parameters']['delta']
        self.eps =  data['parameters']['eps']
        self.seed = data['parameters']['seed'] 
        self.pSVM = data['parameters']['pSVM'] 
        self.NSVM = data['parameters']['NSVM'] 
        self.duration = data['parameters']['training_duration'] 
        self.epochs = data['parameters']['epochs']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
