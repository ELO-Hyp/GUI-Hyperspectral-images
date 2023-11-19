def rcgp(A, r):
    """Random Coordinate Gradient Descent (RCGD)
     Problem formulation:
      min_{U,V} 1/2 ||A -VU||^2 + lambda/2 || I_r - V.TV ||^2 
 
     Input:  A[b x N] - input matrix, N - no of pixels & b - no. of bands
             r[1 x 1] - rank 
     Output: U[r x N], V[b x r] - factorization matrices: A = VU  
    """
    b, pixels = A.shape

    #Initialization
    crit = 1
    
    lbd = 1000
    tol = 1e-2

    U = np.random.rand(r, pixels)    
    V = np.zeros((b,r))
    for i in range(r):
        V[np.random.randint(b), i] = np.random.rand(1)
        
    V = V/np.linalg.norm(V, axis = 0)
    
    VV = V.T@V 
    O_error = np.linalg.norm(np.eye(r) - VV, 'fro')
    #Compute objective function
    objF_old= 0.5* np.linalg.norm(A - V@U, 'fro')**2 + lbd/2 * O_error**2

    while crit>tol: 
        
        if k%2 == 0:    
            # Update U, fix V
            VV = V.T@V;
            Lu = 0.51* np.linalg.norm(VV, 'fro')
            U = np.maximum( U - 1/Lu * (VV@ U - V.T@A), 0) 
        else:
            #Update V, fix U
            Lu = np.linalg.norm(V, 'fro')**2
            UU = U@U.T
            Lv = 0.51* np.linalg.norm(UU, 'fro')   
           
            C = V@UU - A@U.T + 2* lbd* V@(VV - np.eye(r))
            
            p = Lu + Lv/(12*lbd)
            q = np.linalg.norm(C, 'fro')/(12*lbd)
            rt = np.sqrt(p**3/27 + q**2/4);
            alpha = (q/2 + rt)**(1/3) - (-q/2 + rt)**(1/3)
            beta = 12* lbd *(Lu + alpha**2 )+ Lv
            V = np.maximum(V - 1/beta *C, 0)
            VV = V.T@V
            
            # Compute objective function
            O_error = np.linalg.norm(np.eye(r) - VV, 'fro')
            print("O_error", O_error)
            objF= 0.5* np.linalg.norm(A - V@U, 'fro')**2 + lbd/2 * O_error**2
            crit = objF_old - objF
            objF_old = objF
            print("crit:",crit)
        k+=1
    return U
