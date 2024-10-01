import numpy as np
from scipy.sparse import csc_matrix


def _csc_rand(size: int, density: float, lower: float, upper: float) :
    
    num_non_zero = int(size*size*density)
    
    idxs = np.random.randint(0, size, (2, num_non_zero))
    data = np.random.rand(num_non_zero)
    
    A = csc_matrix((lower + (upper - lower)* data, idxs), shape=(size, size))

    return A

def _csc_from_bands(bands: str, size: int, lower: float, upper: float):
    
    [rows_str, cols_str] = bands.split(";")
    rows_str = "" if rows_str == "" else rows_str.split(",")
    cols_str = "" if cols_str == "" else cols_str.split(",")

    rows = [int(row.strip()) for row in rows_str]
    cols = [int(col.strip()) for col in cols_str]

    rowIdxs = np.concatenate([np.arange(row, size) for row in rows] + [np.arange(0, size-col) for col in cols])
    colIdxs = np.concatenate([np.arange(0, size-row) for row in rows] + [np.arange(col, size) for col in cols])
    
    data = lower + (upper - lower) * np.random.rand(len(rowIdxs))
    
    A = csc_matrix((data, (rowIdxs, colIdxs)), shape=(size, size))
    
    return A

def _csc_from_dense(mat):
    return csc_matrix(mat)

def _csc_parse(mat: str, csr_input: str, size: int) :
    [rows_str, cols_str] = csr_input.split(";")
    rows_str = "" if rows_str == "" else rows_str.split(",")
    cols_str = "" if cols_str == "" else cols_str.split(",")
    rows = [int(row.strip()) for row in rows_str]
    cols = [int(col.strip()) for col in cols_str]
    data_str = mat.split(",")
    data = [int(datum.strip()) for datum in data_str]
    
    A = csc_matrix((data, (rows, cols)), shape=(size, size))
    return A

def _img_rand(size: int, lower: float, upper: float) :
    return lower + (upper - lower)* np.random.rand(size)
   
def _img_parse(image: str): 
    img_str = image.split(",")
    img = [int(datum.strip()) for datum in img_str]
    return img



def parse_matrix(matrix_str):
    return np.array(parse(matrix_str))

def parse(matrix_str):
    rows = matrix_str.split(';')
    matrix_lst = [row.split(',') for row in rows]
    matrix_int = [[int(string.strip()) for string in row] for row in matrix_lst]
    return matrix_int