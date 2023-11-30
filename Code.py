import numpy as np
from typing import Tuple

"""
#Ejercicio 2.1
def naive_forward_elimination(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    # TODO: your code here! ----------------------------------

        Ab = np.column_stack((A, b))

    # L = number of unknowns
        L = len(A)

        for p in range(L - 1):
            for r in range(p + 1, L):
            # Identify pivot, pivot's row, subpivot, and subpivot's row
                pivot = Ab[p, p]
                subpivot = Ab[r, p]

            # row r = pivot*row r - subpivot*row p
            Ab[r, p:] = Ab[r, p:] - (subpivot / pivot) * Ab[p, p:]

    # Extract At (upper triangular part) and Bt
        At = Ab[:, :-1]
        Bt = Ab[:, -1]

        return At, Bt                                         

#Ejercicio 3.1
def forward_elimination(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    
    # TODO: your code here! ----------------------------------

    n = A.shape[0]                                      
    x = np.zeros(n)                                     
    UnaSolucion = False                                   
    n = len(A)  # Number of unknowns

    for p in range(n - 1):
        for r in range(p + 1, n):
            # Calculate the multiplier
            multiplier = A[r, p] / A[p, p]

            # Update the elements in row r
            A[r, p:] -= multiplier * A[p, p:]
            b[r] -= multiplier * b[p]
                                            
    if UnaSolucion:                                        
        return A, b, False                             
    else:   
        return A, b, True                              

#Ejercicio 1.1
def backtracking(At: np.ndarray, bt: np.ndarray) -> np.ndarray:

       # Get the number of rows in At
    L = len(At)

    # Preallocate memory for the solution vector x
    x = np.zeros(L)

    # Solve for the last element of x
    x[-1] = bt[-1] / At[-1, -1]

    # Loop through the rows in reverse order
    for r in range(L - 2, -1, -1):
        sum_term = 0.0
        for i in range(r + 1, L):
            sum_term += At[r, i] * x[i]
        x[r] = (bt[r] - sum_term) / At[r, r]

    return x  
"""""

import numpy as np
from typing import Tuple

def naive_forward_elimination(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    n = A.shape[0]
    x = np.zeros(n)

    for row in range(n - 1):
        main_diag_element = A[row, row]
        for next_row in range(row + 1, n):
            factor = A[next_row, row] / main_diag_element
            A[next_row, row:] = A[next_row, row:] - factor * A[row, row:]
            b[next_row] = b[next_row] - factor * b[row]

    return A, b

def forward_elimination(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:

    n = A.shape[0]
    x = np.zeros(n)
    
    is_singular = False

    for col in range(n - 1):
        max_abs_val_row = col + np.argmax(np.abs(A[col:, col]))
        if np.abs(A[max_abs_val_row, col]) <0:
            is_singular = True
            break

        A[[col, max_abs_val_row]] = A[[max_abs_val_row, col]]
        b[col], b[max_abs_val_row] = b[max_abs_val_row], b[col]

        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row, col:] = A[row, col:] - factor * A[col, col:]
            b[row] = b[row] - factor * b[col]

    if is_singular:
        return A, b, False
    else:
        return A, b, True

def backtracking(At: np.ndarray, bt: np.ndarray) -> np.ndarray:
    n = At.shape[0]
    x = np.zeros(n)

    x[n - 1] = bt[n - 1] / At[n - 1, n - 1]

    for row in range(n - 2, -1, -1):
        sum_term = np.dot(At[row, row + 1:], x[row + 1:])
        x[row] = (bt[row] - sum_term) / At[row, row]

    return x
                                     
if __name__ == "__main__":
    
    # EX 2.2
    # Test naive_forward_elimination with a system of > 4 dimensions
    A = np.array([[2, 1, -1, -7], 
                  [1, 3, 3, 9], 
                  [3, 2, 4, 2],
                  [3, 2, 4, 1]], 
                  dtype=float)
    
    b = np.array([1, 4, 6, 2],
                 dtype=float)

    print("\033[1m" + "TESTING Exercise 2.2 Naive forward elimination" + "\033[0m")
    print("A = \n", A)
    print("b = \n", b)

    At, bt = naive_forward_elimination(A, b)
    
    print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places

    # EX 1.2
    # Testing backtracking with a system of > 4 dimensions
    print("TESTING Exercise 1.2 Backtracking: ", np.round(backtracking(At, bt), 2), "\n")

    print("\033[1m" + "---------------------------------------------" + "\033[0m")

    # EX 3.3
    # Test forward_elimination
    A = np.array([[1, 2, 3], 
                  [1, 2, 2],
                  [0, 1, 2]], 
                  dtype=float)
    
    b = np.array([14, -1, 8],
                dtype=float)
    
    print("\033[1m" + "TESTING Exercise 3.3 Forward elimination" + "\033[0m")
    print("A = \n", A)
    print("b = \n", b)

    At, bt, singular = forward_elimination(A, b)

    print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places
    print("singular = ", singular)

    print("Backtracking: ", np.round(backtracking(At, bt), 2), "\n")

    # Test naive_forward_elimination
    #A = np.array([[1, 2, 3], 
    #              [1, 2, 2],
    #              [0, 1, 2]], 
    #              dtype=float)
    
    #b = np.array([14, -1, 8],
    #            dtype=float)
    
    #print("\033[1m" + "TESTING Exercise 3.3 Forward elimination" + "\033[0m")
    #print("A = \n", A)
    #print("b = \n", b)

    #At, bt = naive_forward_elimination(A, b)

    #print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    #print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places

    # --- EXPLICACION: ---
    # No podemos utilizar la funcion naive_forward_elimination porque este 
    # sistema de ecuaciones generará una excepcion:

        # RuntimeWarning: divide by zero encountered in scalar divide
        # factor = A[j, p] / pivot                    
        # Calculate the factor to make elements under the pivot a_pp equal to 0
        # RuntimeWarning: invalid value encountered in multiply
        # A[j, p:] = A[j, p:] - factor * A[p, p:]     
        # Update the row A[j, :] based on the equation

    # Lo cual significa que la variable pivot es un numero muy pequeño o cero.
    # Este problema lo solucionamos en la funcion forward_elimination comprobando
    # que el pivote no sea menor que 1e-15, el cual es un numero muy pequeño.

    print("\033[1m" + "---------------------------------------------" + "\033[0m")

    # Test forward_elimination
    A = np.array([[1, 2, -1, 3], 
                  [2, 0, 2, -1],
                  [-1, 1, 1, -1],
                  [3, 3, -1, 2]], 
                  dtype=float)
    
    b = np.array([-8, 13, 8, -1],
                dtype=float)

    # EX 3.2
    print("\033[1m" + "TESTING Exercise 3.2 Forward elimination with one solution" + "\033[0m")
    print("A = \n", A)
    print("b = \n", b)

    At, bt, singular = forward_elimination(A, b)

    print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places
    print("singular = ", singular)

    print("Backtracking: ", np.round(backtracking(At, bt), 2), "\n")

    print("\033[1m" + "---------------------------------------------" + "\033[0m")

    # EX 3.4
    # Test forward_elimination
    A = np.array([[1, -1, 1, -2], 
                  [2, 1, -1, -1],
                  [1, -4, 4, -5],
                  [1, 5, -5, 4 ]], 
                  dtype=float)
    
    b = np.array([2, 1, 5, -4],
                dtype=float)

    print("\033[1m" + "TESTING Exercise 3.4 Forward elimination with multiple solutions" + "\033[0m")
    print("A = \n", A)
    print("b = \n", b)

    At, bt, singular = forward_elimination(A, b)
    
    print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places
    print("singular = ", singular)

    # Test naive_forward_elimination
    #A = np.array([[1, -1, 1, -2], 
    #              [2, 1, -1, -1],
    #              [1, -4, 4, -5],
    #              [1, 5, -5, 4 ]], 
    #              dtype=float)
    
    #b = np.array([2, 1, 5, -4],
    #            dtype=float)

    #print("\033[1m" + "TESTING Exercise 3.4 Naive forward elimination with multiple solutions" + "\033[0m")
    #print("A = \n", A)
    #print("b = \n", b)

    # At, bt = naive_forward_elimination(A, b)
    
    #print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    #print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places

    # --- EXPLICACION: ---
    # En este caso sucede lo mismo que en la prueba del apartado 3.3 y 3.4,
    # este sistema de ecuaciones generará una excepcion:

        # RuntimeWarning: invalid value encountered in scalar divide
        # factor = A[j, p] / pivot                    
        # Calculate the factor to make elements under the pivot a_pp equal to 0

    print("\033[1m" + "---------------------------------------------" + "\033[0m")
    
    A = np.array([[1, -2, -2, 1], 
                  [1, 1, 1, -1],
                  [1, -1, -1, 1],
                  [6, -3, -3, 2]], 
                  dtype=float)
    
    b = np.array([4, 5, 6, 32],
                dtype=float)

    # EX 3.4
    print("\033[1m" + "TESTING Exercise 3.4 Forward elimination with zero solution" + "\033[0m")
    print("A = \n", A)
    print("b = \n", b)

    At, bt, singular = forward_elimination(A, b)
    
    print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places
    print("singular = ", singular)

    #A = np.array([[1, -2, -2, 1], 
    #              [1, 1, 1, -1],
    #              [1, -1, -1, 1],
    #              [6, -3, -3, 2]], 
    #              dtype=float)
    #
    #b = np.array([4, 5, 6, 32],
    #            dtype=float)

    #print("\033[1m" + "TESTING Exercise 3.4 Naive forward elimination with zero solution" + "\033[0m")
    #print("A = \n", A)
    #print("b = \n", b)

    #At, bt = naive_forward_elimination(A, b)

    #print("At = \n", np.round(At, 2))  # Round matrix to 2 decimal places
    #print("bt = \n", np.round(bt, 2))  # Round vector to 2 decimal places

    # --- EXPLICACION: ---
    # En este caso sucede lo mismo que en la prueba del apartado 3.3 y 3.4,
    # este sistema de ecuaciones generará una excepcion:

        # RuntimeWarning: invalid value encountered in scalar divide
        # factor = A[j, p] / pivot                    
        # Calculate the factor to make elements under the pivot a_pp equal to 0
