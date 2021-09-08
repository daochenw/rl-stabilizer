import numpy as np

I = np.eye(2);
X = np.array([[0, 1], [1, 0]]); 
Y = np.array([[0, -1j], [1j, 0]]);
Z = np.array([[1, 0], [0, -1]]);

pauli = {0:I, 1:X, 2:Y, 3:Z}
coefficient = {0: 1, 1: -1, 2: 1j, 3: -1j}

def gs_prj(base, target):
    if np.allclose(np.sum((base.H*base)), 0):
        return 0*target
    return np.sum((base.H*target)) / np.sum((base.H*base)) * base

def gram_schmidt(vectors):
    dim = vectors[0].size
    n = len(vectors)
    # if not check_lin_independence(vectors):
    #     return None
    V = np.matrix(np.zeros([dim,n], dtype=np.complex_))
    U = np.matrix(np.zeros([dim,n], dtype=np.complex_))
    for i in range(len(vectors)):
        V[:,i] = vectors[i]
    for i in range(len(vectors)):
        U[:,i] = V[:,i]
        for j in range(i):
            U[:,i] -= gs_prj(U[:,j], V[:,i])
    for i in range(len(vectors)):
        norm = np.linalg.norm(np.matrix(U[:,i]), 2)
        if np.allclose(norm, 0):
            U[:,i] *= 0
        else:    
            U[:,i] /= norm
    return [np.matrix(U[:,i]) for i in range(len(vectors))]

def orthogonal_projector(vectors):
    vectors = [np.transpose([vectors[i]]) for i in range(len(vectors))]
    dim  = len(vectors[0])
    ortho_vecs = gram_schmidt(vectors)
    A = np.matrix(np.zeros([dim, len(ortho_vecs)], dtype=np.complex_))
    for i in range(len(ortho_vecs)):
        A[:,i] = ortho_vecs[i]
    P = A*A.H
    return P

def tensor(*args):
    matrices = list(args)
    out = matrices.pop(0)
    while matrices:
        out = np.kron(out, matrices.pop(0))
    return np.array(out)

def number_to_base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]