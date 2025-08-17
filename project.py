"""
EE 793 Cryptography Course Project
LWE-based Cryptanalysis using Integer programming
References: https://eprint.iacr.org/2023/1162.pdf

Authors: Kirthan Kamble(22b1229), Jatin Singhal(22b1277), Akshat Singhvi(22b1231), Preetish Sathawane(22b1217) 
"""

# Imports
import numpy as np     # to handle and public key matrices and arrays
import cvxpy as cp     # to solve linear programming problem
from math import floor # utility function to round down the value


def modular_matrix_inverse(A, q):
    """
    Computes the inverse of matrix A modulo q using the adjugate method.
    """
    n = A.shape[0]
    detA = round(np.linalg.det(A)) % q

    try:
        det_inv = pow(int(detA), -1, q)
    except ValueError:
        raise ValueError(f"Matrix is not invertible modulo {q}")

    # Compute adjugate matrix
    adj = np.zeros_like(A, dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            cofactor = ((-1) ** (i + j)) * round(np.linalg.det(minor))
            adj[j, i] = cofactor % q  # Note transpose for adjugate

    return (det_inv * adj) % q


def main():
    n = 10          # Number of equations / rows
    m = 15          # Number of unknowns / columns
    q = 17          # Modulus (prime number)

    # Generate LWE Data
    np.random.seed(0)
    A = np.random.randint(0, q, size=(n, m))            # Public matrix
    s = np.random.randint(0, q, size=(n,))              # Secret vector
    e_max, e_min = q // 4 - 1, -q // 4 + 1              # Error bounds
    e = np.random.randint(e_min, e_max + 1, size=(m,))  # Noise vector
    t = (np.dot(s, A) + e) % q                          # LWE equation

    print("[+] Public matrix A:\n", A)
    print("[+] Secret vector s:", s)
    print("[+] Target vector t:", t)

    # section 3.1 of the paper
    A0, A1 = A[:, :n], A[:, n:]         # Partition A: A = [A0 | A1]
    t0, t1 = t[:n], t[n:]               # Partition t similarly

    A0_inv = modular_matrix_inverse(A0, q)

    # section 3.2 of the paper
    def psi(v):
        return np.dot(np.dot(v, A0_inv), A1) % q

    def phi(v):
        return (psi(v) + t1 - psi(t0)) % q

    # Lemma 1 of the paper
    W = psi(np.eye(n, dtype=int))      # Shape (n, m-n)
    U = phi(np.zeros(n, dtype=int))    # Shape (m-n,)

    # linear programming problem
    x = cp.Variable(m, integer=True)           # Approximate error vector
    f = cp.Variable(m - n, integer=True)       # Integer slack variables

    objective = cp.Minimize(cp.sum_squares(x)) # Objective: minimize squared norm of x

    constraints = [                            # Constraints from Proposition 2
        cp.matmul(x[:n], W) + U - x[n:] == q * f,
        x >= e_min,
        x <= e_max,
        f >= -floor((n * (q - 1) * e_max - e_min + q - 1) / q),
        f <= floor((e_max - n * (q - 1) * e_min) / q)
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    recovered_e = np.round(x.value).astype(int)
    s_hat = np.dot((t0 - recovered_e[:n]), A0_inv) % q

    print("\n[+] Recovered secret key s_hat:", s_hat)
    print("[+] Actual error vector e:      ", e)
    print("[+] Recovered error vector x:   ", recovered_e)

if __name__ == "__main__":
    main()