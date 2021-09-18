import numpy as np

X = np.array([[0, 1],[1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
pauli_dict = {(0,0): np.eye(2), (0,1): Z, (1,0): X, (1,1): Y}
sign_dict = {0: 1, 1: -1}

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

def binary_to_number(binary):
    number = 0
    for b in binary:
        number = (2 * number) + b
    return number

def gf2_rank(rows):
    """
    Find rank of a matrix over GF2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank

def create_gamma_matrix(n_qubits):
    return np.kron(np.array([[0,1],[1,0]]), np.eye(n_qubits))

def commute(check_matrix, gamma_matrix):
    matrix_product = np.mod(np.matmul(np.matmul(check_matrix,gamma_matrix),np.transpose(check_matrix)),2)
    all_zeros = not np.any(matrix_product)
    return all_zeros

def lin_indep(check_matrix,n_qubits):
    rows = [int(binary_to_number(check_matrix[i,:])) for i in range(n_qubits)]
    return n_qubits == gf2_rank(rows.copy())

# can be potentially faster!
def stabilizer_to_state(check_matrix, sign, n_qubits):
    identity = np.eye(np.power(2,n_qubits))
    Proj = identity
    for i in range(n_qubits):
        P = 1
        for j in range(n_qubits):
            Pj = pauli_dict[(check_matrix[i,j],check_matrix[i,n_qubits+j])]
            P = np.kron(P,Pj)
        P = sign_dict[sign[i]]*P
#         print('P',i,'=',repr(P))
        Proj = np.matmul(identity+P,Proj)
        
    Proj_rank = np.linalg.matrix_rank(Proj)
    assert Proj_rank == 1
    
    eigvals, eigvecs = np.linalg.eigh(Proj)
    # the : is very important below!
    state = eigvecs[:,int(np.argwhere(np.isclose(eigvals,np.power(2,n_qubits))))]
    assert np.isclose(np.linalg.norm(state),1)
    
    return state

def bitarray_to_basis(x, n_qubits, chi):
    basis = []
    for i in range(chi):
#         print('i =',i)
        sign = x[:n_qubits]
#         print('sign = ', sign)
        x = x[n_qubits:]
        check = x[:2*np.power(n_qubits,2)]
        check_matrix = np.reshape(check,(n_qubits,2*n_qubits))
#         print('check_matrix = ', check_matrix)
        if not commute(check_matrix, create_gamma_matrix(n_qubits)) or not lin_indep(check_matrix,n_qubits):
            return -1
        assert commute(check_matrix, create_gamma_matrix(n_qubits))
        assert lin_indep(check_matrix,n_qubits)
        basis.append(stabilizer_to_state(check_matrix, sign, n_qubits))
        x = x[2*np.power(n_qubits,2):]
    assert len(x) == 0  
    return tuple(basis)