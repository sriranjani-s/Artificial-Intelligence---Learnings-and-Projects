import numpy as np

def execute_pivot(A,r,s):
    A[r,:] = A[r,:]/A[r,s]
    for i in range(A.shape[0]):
        if i!=r:
            A[i,:] = A[i,:] - A[i,s]*A[r,:]
    return

def simplex_phase_2(A):                    
    while True:    
        pivots = {}
        for j in range(1,A.shape[1]-1):
            cj = A[0,j]
            if cj>0:
                for i in range(1,A.shape[0]):
                    bi = A[i,-1]
                    if A[i,j]>0:
                        pivots[(i,j)] = bi/A[i,j]                    
        if len(pivots)==0:
            break        
        pivot = min(pivots.keys(), key=(lambda k: pivots[k]))
        execute_pivot(A,pivot[0],pivot[1])
    return

def determine_basis(D):
    basis = []
    epsilon = 1e-6
    for j in range(1,D.shape[1]):
        if np.abs(D[0,j])<epsilon:
            zeros = sum(abs(D[1:D.shape[0],j])<epsilon)
            ones = sum(abs(D[1:D.shape[0],j]-1)<epsilon)
            if (ones==1) and (zeros==D.shape[0]-2):
                basis.append(j)
    return basis

def simplex_phase_1(A):
    eb = np.sum(A[1:A.shape[0],-1])
    eA = np.sum(A[1:A.shape[0],1:A.shape[1]-1],axis=0)
        
    a = np.append(np.append(np.append([1], eA), np.zeros((1,A.shape[0]-1))), eb) 
    
    B = np.append(A[1:A.shape[0],0:A.shape[1]-1], np.eye(A.shape[0]-1), axis=1)
    C = np.append(B, A[1:A.shape[0],-1].reshape(B.shape[0],1), axis=1)
    D = np.append(a.reshape(1,len(a)),C,axis=0)    
    
    simplex_phase_2(D)
    
    epsilon = 1e-6
    if not abs(D[0,-1])<epsilon:
        print("No feasible solution!")
        return 
    
    while True:
        basis = determine_basis(D)        
        finished = True
        for j in basis:
            if j>=A.shape[1]-1:
                for i in range(1,D.shape[0]):
                    if abs(D[i,j]-1)<epsilon:
                        k = np.argmax(np.abs(D[i,1:A.shape[1]-1]))+1
                        execute_pivot(D,i,k)
                        finished = False
        if finished:
            break


    E = np.append(D[:,0:A.shape[1]-1], D[:,-1].reshape(D.shape[0],1), axis=1)

    for i in range(1,E.shape[1]-1):
        if i in basis:
            E[0,E.shape[1]-1] -= A[0,i]*D[np.argmax(D[:,i]),-1]
        else:
            E[0,i] = A[0,i]

    return E

def main():
    # diet problem, Lab 10_a
    A = np.array([[1,-9,-7,0,0,0,0],
                  [0,2,4,-1,0,0,12],
                  [0,5,3,0,-1,0,15],
                  [0,4,1,0,0,-1,8]], dtype=np.float32)

       
    print(A)
    
    A = simplex_phase_1(A)
    simplex_phase_2(A)

    print(np.array(100*A, dtype=np.int32)/100.0)

main()
