def print_latex_matrix(A):
    n = A.shape[0]
    m = A.shape[1]
    print('\\begin{bmatrix}')
    for i,row in enumerate(A):
        line = '    '
        for j,col in enumerate(row):
            line += str(col)
            if j < m-1:
                line += ' & '
        if i < n-1:
            line += ' \\\\'
        print(line)
    print('\\end{bmatrix}')
    print('')

if __name__ == "__main__":
    import numpy as np
    A = np.array([[0.8,0.1],[0.1,0.8]])
    B = np.array([[1.0,0.0],[0.0,1.0]])
    a = np.array([[0.9]])
    Aa = np.array([[0.0,1.0],[1.0,0.0]])[:,:,np.newaxis]
    b = np.array([[0.0]])
    Bb = np.array([[0.0,0.0],[0.0,0.0]])[:,:,np.newaxis]
    Q = np.eye(2)
    R = np.eye(2)
    S0 = np.eye(2)

    print_latex_matrix(A)
    print_latex_matrix(B)
    print_latex_matrix(Q)
    print_latex_matrix(R)