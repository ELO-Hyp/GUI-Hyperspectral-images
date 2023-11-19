import numpy as np
from scipy import linalg
from scipy import sparse
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.linalg import pinv

def lpp(img, r=0, wType = 0):
    """ Locality Preserving Projection (LPP)
    ---------------------------------------
    Inputs: img - size(noPixels x noBands) 
            r   - the reduced dimension (r<<noBands)
            wType - 0: simple, the weights are 1
                  - 1: heat kernel weighting
    ---------------------------------------
    """
    epsilon = .00001                       # to quantify how "close" two points are
    t = 1                                  # Heat kernel parameter for when calculating the weights
    noPixels = img.shape[0]
    if r == 0:
        r = img.shape[-1]
    
    #Create the graph matrix and add weights:
    """
    W = np.zeros((noPixels, noPixels), dtype = int)
    # This version is without using the sklearn, but takes to much time
    if wType == 0:
        for i in range(noPixels):
            for j in range(i,noPixels):
                if (img[i,:] - img[j,:])@(img[i,:] - img[j,:]).T < epsilon: 
                    W[i,j] = 1
                    W[j,i] = 1
    else:
        for i in range(noPixels):
            for j in range(i,noPixels):
                ijDist = (img[i,:] - img[j,:])@(img[i,:] - img[j,:]).T 
                if ijDist <= epsilon: 
                    W[i,j] = np.exp( -ijDist/ t)
                    W[j,i] = W[i,j]
                    
    """
    
    # ---------------------------------------------------------
    if img.shape[0] > 50000:
        # Specify the subsampling factor (e.g., 0.1 for 10% subsampling)
        subsampling_factor = 0.1

        # Determine the number of rows to keep based on the subsampling factor
        num_rows_to_keep = int(subsampling_factor * img.shape[0])

        # Randomly select rows to keep
        selected_rows = np.random.choice(img.shape[0], num_rows_to_keep, replace=False)

        # Subsampled matrix hyp_img
        img = img[selected_rows, :]

    # ----------------------------------------------------------
    noNbrs = 2
    nbrs = NearestNeighbors(n_neighbors= noNbrs,algorithm='auto')
    nbrs.fit(img)
    if wType == 0:
        W = kneighbors_graph(nbrs, noNbrs,
                                 mode='connectivity', include_self=True)
    else:
        W = kneighbors_graph(nbrs, noNbrs,
                                 mode='distance', include_self=True)
        W.data = np.exp(-W.data ** 2 / t)
            
    #W = W.toarray()   #returnse the dense representation of the matrix
    #W = np.maximum(W, W.T) #to symetrize W
    W = ((W+W.T)-np.abs(W-W.T))/2 # to symmetrize W
    
    #Compute matrices
    D = sparse.diags(np.array(W.sum(1))[:,0])
    L = D - W 
    # Solve the GEP: img.T@L@img@ v = lbd* img.T@D@img@ v
    # Eig decomposition of img.T@ D@ img
    AUX = sparse.csr_matrix.dot(img.T, sparse.csr_matrix.dot( D, img))
    S, U = linalg.eigh(AUX)

    # Combine  img.T@L@ img and img.T@D@ img on left hand side via decomposition of img.T@D@img
    S[S <= 0] = np.inf
    Sinv = 1. / np.sqrt(S)
    AUX2 = sparse.csr_matrix.dot(img.T, sparse.csr_matrix.dot( L, img))
    W = Sinv[:, None] * np.dot(U.T, np.dot(AUX2, U)) * Sinv
    evals, evecs = linalg.eigh(W)
    
    evecs = np.dot(U, Sinv[:, None] * evecs)
    #redImg = img@evecs

    return evecs #redImg

class LPP():
    def __init__(self, n_bands=0, w_type=0):
        self.n_bands = n_bands
        self.w_type = w_type

    def fit(self, img):
        if self.n_bands == 0:
            self.n_bands = img.shape[-1]
        self.proj = lpp(img, self.n_bands, self.w_type)

    def set_r(self, r):
        if r == -1:
            r = self.n_bands
        return r

    def transform(self, img, r=-1):
        r = self.set_r(r)
        return self.proj[:r,:]@img.T

    def inverse_transform(self, tr_img, r=-1):
        r = self.set_r(r)
        inv = pinv(self.proj[:r])
        return (inv@tr_img).transpose()

    def endecode(self, img, r=-1):
        r = self.set_r(r)
        return self.inverse_transform(self.transform(img, r), r)

    def save(self, file_name):
        np.savez(file_name,
                 lpp_proj=self.proj,
                 w_type=[self.w_type])
        
    def trained_weights(self):
        props = {
            'projection' : self.proj
        }
        return props

    def load(self, file_name):
        loaded = np.load(file_name)
        self.proj = loaded['lpp_proj']
        self.n_bands = self.proj.shape[-1]
        self.w_type = loaded['w_type'][0]
        
    def load_from_props(self, projection):
        #meta parameters already defined, should only be used when they are known
        self.proj = projection
        self.n_bands = self.proj.shape[-1]