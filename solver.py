
import argparse
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import  gmres, spilu, LinearOperator
import prepare
import math

A_TEST = csc_matrix(([20, -5, -5, 15, -5, -5, 15, -5, -5, 15, -5, -5, 10], ([0,0,1,1,1,2,2,2,3,3,3,4,4], [0,1,0,1,2,1,2,3,2,3,4,3,4])), shape=(5, 5))
B_TEST = [1100, 100, 100,100,100]
np.set_printoptions(suppress=True, linewidth=200, precision=2)

def main():
    
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--mat", type=str, default="\"random\"", help="Matrix: \"random\" | \"banded\" | \"<matrix>\" | . If a matrix isnt given, the diagonal is nonzero to [almost] ensure invertibility. Matrix in form \"x,x,x;x,x,x...\"")
    parser.add_argument("--csc_input", type=str, default="\"dense\"", help="Input to build the csr matrix, only applicable when --mat=<matrix>. Either \"dense\" or the rowIdxs and colIdxs if <matrix> is the data. For the latter the format is x,x,x;x,x,x")
    parser.add_argument("--bands", type=str, default="dense", help="Left and or upper coordinates of bands, only applicable when --mat=banded. Format is \"r,r,r,..;c,c,c,..\"...")
    parser.add_argument("--image", type=str, default="random", help="Image of Ax, either \"random\" or x,x,x,x,x,x...")
    parser.add_argument("--size", type=int, default=1000000, help="Height of square matrix, only applicable when --mat=random or banded.")
    parser.add_argument("--density", type=float, default=0.001, help="Ratio of non-diagonal non-zero elements to element count, only applicable when --mat= random.")
    parser.add_argument("--mat_lower", type=float, default=-20, help="Lower bound of elements, only applicable when --mat=\"random\".")
    parser.add_argument("--mat_upper", type=float, default=20, help="Upper bound of elements, only applicable when --mat\"random\".")
    parser.add_argument("--image_lower", type=float, default=-20, help="Lower bound of elements, only applicable when --image=\"random\".")
    parser.add_argument("--image_upper", type=float, default=20, help="Upper bound of elements, only applicable when --image\"random\".")
    
    args = parser.parse_args()

    mat = args.mat
    csc_input = args.csc_input
    size = args.size
    density = args.density
    mat_lower = args.mat_lower
    mat_upper = args.mat_upper
    image_lower = args.image_lower
    image_upper = args.image_upper
    image = args.image
    bands = args.bands
    
    A = None
    b = None
    
    if mat == "random":
        print("Generating random matrix...")
        A = prepare._csc_rand(size, density, mat_lower, mat_upper)
    elif mat == "banded":
        print("Generating random [banded] matrix with set band positions...")
        A = prepare._csc_from_bands(bands, size, mat_lower, mat_upper)
        
    else:
        if csc_input == "dense":
            A = prepare._csc_from_dense(prepare.parse_matrix(mat))
        else:
            A = prepare._csc_parse(mat, csc_input, size)
            print(A.todense())
            
    
    if image == "random":
        b = prepare._img_rand(size, image_lower, image_upper)
    else:
        b = prepare._img_parse(image)
        
    
    x = None
    if mat == "random" or mat == "banded":
        x, _ = solve(A, b, random_upper=mat_upper)
        print(f"Solution norm: {np.linalg.norm(x)}")
    else:
        x, _ = solve(A, b, random_upper=None)
        print("Solution: ", x)
        
    if A.shape[0] < 21:
        print("A:")
        print(A.todense())
        print("b:")
        print(b)
        print("x:")
        print(x)
    



    
    
def solve(A: csc_matrix, b, random_upper=None):
    
    print("Solving...")
    print("Generating preconditioner...")
    # ILU preconditioner
    ilu = None
    
    if random_upper != None:
        i = 1
        while i < A.shape[0]:
            try:
                ilu = spilu(A)
                break   
            except Exception as e:
                k = (-1)**(i + 1) * math.floor(i/2)
                print(f"Singular, setting diagonal {k} from main.")
                A.setdiag(2 * random_upper, k)
                i+=1
    else:
        ilu = spilu(A)
        
    print("Preconditioning done.")
    print("Starting gmres...")
                
    x = None
    residual = None
    
    try:
        x, _ = gmres(A, b, M=LinearOperator(A.shape, ilu.solve) )
        residual = b - A.dot(x)
        print(f"Solution residual: {np.linalg.norm(residual)}")
    except Exception as e:
        print("Error: ", e)

    return x, residual





if __name__ == "__main__":
    main()