import numpy as np

def read_matrix(file):
    with open(file) as f:
        n = int(f.readline())
        matrix = []

        for _ in range(n):
            matrix.append(list(map(int, f.readline().split())))

    return np.array(matrix)

A = read_matrix("data/matrix_a.txt")
B = read_matrix("data/matrix_b.txt")
C_cpp = read_matrix("data/result_cpp.txt")

C_py = np.dot(A,B)

if np.array_equal(C_cpp,C_py):
    print("Verification successful")
else:
    print("Results do not match")