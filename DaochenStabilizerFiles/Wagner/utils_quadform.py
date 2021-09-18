import numpy as np
import fieldmath


X = np.array([[0, 1],[1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
Ddict = {(0,0): 0, (0,1): 2, (1,0): 4,(1,1): 6}
Jdict = {0:0, 1:4}
ketbit = {0:np.array([1,0]),1:np.array([0,1])}


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


def bits_to_number(binary):
    number = 0
    for b in binary:
        number = (2 * number) + b
    return number


def number_to_padbase2(n, k):
    if n == 0:
        return [0]*k
    digits = []
    while n:
        digits.append(int(n % 2))
        n //= 2
    nopad = digits[::-1]
    return [0]*(k-len(nopad))+nopad


def tensor(matrices):
    out = matrices.pop(0)
    while matrices:
        out = np.kron(out, matrices.pop(0))
    return out


def row_reduce_gf2(A,n):
    n = len(A)
    B = fieldmath.Matrix(n,n,fieldmath.PrimeField(2))

    for i in range(n):
        for j in range(n):
            B.set(i,j,int(A[i,j]))

    B.reduced_row_echelon_form()

    for i in range(n):
        for j in range(n):
            A[i,j]=B.get(i,j)
            
    return A[~np.all(A == 0, axis=1)]


def convert_Dbits(x, real=False):
    if real:
        return 4*x
    else:
        assert len(x)%2 == 0
        return [Ddict[(x[2*i],x[2*i+1])] for i in range(int(len(x)/2))]

    
def convert_Jbits(x,n):
    assert len(x) == n*(n-1)/2
    x = [Jdict[x[i]] for i in range(len(x))]
    tri = np.zeros((n, n))
    tri[np.triu_indices(n, 1)] = x
    # do NOT add the lower triangular part to make it symmetric a that is wrong by a factor of 2!
    return tri


def bits_to_stab(bits,n,chi,real=False):
    bits = np.array(bits,dtype=int)
    basis = []
    
    if real:
        m = 3/2*np.power(n,2)+3/2*n
    else:
        m = 3/2*np.power(n,2)+5/2*n
    assert m%1 == 0
        
    m = int(m)
    
    assert len(bits) == chi*m
    
    for i in range(chi):
        bitarray = bits[i*m:(i+1)*m]
        assert len(bitarray) ==  m
        h = bitarray[:n]
        G = bitarray[n:n+np.power(n,2)]
        if real: 
            D = bitarray[n+np.power(n,2):n+np.power(n,2)+n]
            J = bitarray[n+np.power(n,2)+n:len(bitarray)] 
        else:
            D = bitarray[n+np.power(n,2):n+np.power(n,2)+2*n]
            J = bitarray[n+np.power(n,2)+2*n:len(bitarray)]
        
        G = np.reshape(G,[n,n])
        G = row_reduce_gf2(G,n)
        k = len(G)
        
        if real: 
            D = convert_Dbits(D[:k],real)
        else:
            D = convert_Dbits(D[:2*k],real)

        Jmatrix = convert_Jbits(J,n)[:k,:k]
        state = 0 
        
        for vecx_number in range(np.power(2,k)):
            vecx = number_to_padbase2(vecx_number,k)
            qx = np.mod(np.dot(vecx,D) + np.dot(vecx,Jmatrix.dot(vecx)),8)
            x = h + sum([G[i,:]*vecx[i] for i in range(k)])
            x = np.mod(x,2)
            ketx = tensor([ketbit[x[i]] for i in range(n)])
            state = state+np.exp(1j*np.pi/4*qx)*ketx
            
        state = state/np.power(2,k/2)
        basis.append(state)

    return tuple(basis)