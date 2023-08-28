import numpy as np
from scipy.linalg import toeplitz
import scipy.signal as signal

"""
  Toeplitz, doubly blocked Toeplitz def ppt
"""


"""
  Lehet vektor is a bemenet, ekkor mx képzéssel indul a program: vector_to_matrix(). Hasonlóan a kimenet is lehet bármilyen formájú.
"""
# gerjesztés mx-a
U = np.array([[1 , 2, 3], [4, 5, 6]])

# impulzusválasz mx-a
H = np.array([[10 , 20], [30, 40]])


# gerjesztés és impulzusválasz mx sorainak és oszlopainak száma
U_row_num, U_col_num = U.shape
H_row_num, H_col_num = H.shape

# kimenet mx sorainak és oszlopainak száma
output_row_num = U_row_num + H_row_num - 1
output_col_num = U_col_num + H_col_num - 1

# imp.válasz mx-nak nullákkal való feltöltése, hogy ugyanakkora legyen mint a kimeneti mátrix
H_zero_padded = np.pad(H,

  # (H első sora elé hány sort tegyen, H utolsó sora után hány sort tegyen)
  ((output_row_num - H_row_num , 0),

  # (H első oszlopa elé hány oszlopot tegyen, H utolsó oszlopa után hány oszlopot tegyen)
  (0, output_col_num - H_col_num)),

  #mivel töltse ki
  'constant', constant_values =0)

  #print(H_zero_padded)


"""
ppt-n kép
Kis Toeplitz mx-ok létrehozása: H_zero_padded mx minden sorából csinálunk egy kis Toeplitz mx-ot,
  melynek annyi oszlopa van, mint a gerjesztés mx-nak,
  annyi sora, amennyi oszlopa a H_zero_paddednek <- Toeplitz mx készítésének módjából következően
  és toeplitz_list-ben tároljuk őket

Ezek a scipy.linalg könyvtár toeplitz() fv-ével készülnek:
  1) Egy sora a H_zero_padded-nek (toeplitz()-ben a c paraméter) bemegy a fv-be és a fv ebből (ahogy defben láttuk) elkészíti
  2) Még a készülő Toeplitz mx-nak az első sorát is megadjuk (r paraméter a toeplitz()-ban), ezzel lesz olyan, amilyet szeretnénk
"""

toeplitz_list = []
# utolsó sortól az elsőig iteráció, H_zero_padded.shape[0]-1-től, -1-ig (ezzel már nem kezd új ciklust), -1-es lépésekben
for i in range(H_zero_padded.shape[0]-1, -1, -1):
    c = H_zero_padded[i, :]   # H i-edik sorának másolása
    r = np.r_[c[0], np.zeros(U_col_num - 1)] # A készülő kis Toeplitz mx első sora: c[0] és nullák konkatenálása (numpy.r_() fv-el)
    # toeplitz() fv a scipy.linalg könyvtárból
    toeplitz_m = toeplitz(c,r)
    toeplitz_list.append( toeplitz_m )
    #print('H '+ str(i)+'\n', toeplitz_m )


"""
ppt: saját doubly blocked elrendezés
"""

# INDEXEK TÁROLÁSÁHOZ
# 1-től 3-ig számsorozat (1,2,3), tehát c[i]=i+1, i=0,1,2
c = range(1, H_zero_padded.shape[0]+1)

# indexek tárolásához használt Toeplitz mx első sora: [1, 0]
r = np.r_[c[0], np.zeros( U_row_num -1, dtype=int)]
doubly_indices = toeplitz(c, r)
#print('doubly indices \n', doubly_indices )


"""
Doubly blocked Toeplitz feltöltése a kicsi Toeplitz-ekkel
"""
# Kicsi toeplitz-ek (F0 (lila), F1 (zöld), F2 (halvány-narancssárga)) alakja (fent bevezetett):
# H_zero_padded.shape[1] x U_col_num
  # H_zero_padded.shape[1] = H_zero_padded oszlopainak száma
  # U_col_num              = gerjesztés mx oszlopainak száma
toeplitz_shape = (H_zero_padded.shape[1], U_col_num)

# doubly-blocked hxw-s mx lesz, ahol
# h = kis Toeplitz-ek sorainak száma x kis Toeplitz-ek darabszáma (w = ..., a doubly blocked Toeplitz-unk kialakítása miatt)
h = toeplitz_shape[0]*doubly_indices.shape[0]
w = toeplitz_shape[1]*doubly_indices.shape[1]
doubly_blocked_shape = [h, w]
# egyelőre nullákkal töltjük fel
doubly_blocked = np.zeros(doubly_blocked_shape)

# doubly-blocked feltöltése a kis Toeplitz-ekkel
b_h, b_w = toeplitz_shape # minden kis Toeplitz mx alakja
for i in range(doubly_indices.shape[0]): # 0-tól doubly_indices.shape[0]-1-ig
    for j in range(doubly_indices.shape[1]): # 0-tól doubly_indices.shape[1]-1-ig
        # kezdő magasság és szélesség (i megadja, hogy mennyi mx van fölötte; j megadja, hogy mennyi mx van tőle balra, ennyivel kell tolni a kezdőpontot)
        start_i = i * b_h
        start_j = j * b_w
        # befejező magasság és szélesség (végpont: a kezdőponttól 1 kis Toeplitz mx terjedelmet kell rajta tolni magasságban és szélességben is)
        end_i = start_i + b_h
        end_j = start_j + b_w
        doubly_blocked[start_i : end_i, start_j : end_j] = toeplitz_list[doubly_indices[i,j]-1]
#print( doubly_blocked )


"""
ppt kép a vektorializálásról
"""
def matrix_to_vector (input):
    input_h , input_w = input.shape
    output_vector = np.zeros( input_h * input_w , dtype=input.dtype)
    # tengelyesen tükrözi az input mx-ot a középső sorára
    input = np.flipud(input)
    for i, row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row

    return output_vector

# U mx vektorializálása, hogy össze tudjuk szorozni a doubly blocked Toeplitz-al
vectorized_U = matrix_to_vector(U)

# mx szorzással (matmul() fv a numpy könyvtárból) kapjuk az eredményt
result_vector = np.matmul(doubly_blocked , vectorized_U)
#print('result vector: ', result_vector )


"""
ppt kép
"""
def vector_to_matrix (input , output_shape):
    output_h , output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range( output_h ):
        st = i* output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # visszatükrözés
    output=np.flipud(output)
    return output


# konvolúció eredménye
result_matrix = vector_to_matrix(result_vector, (U_row_num + H_row_num - 1, U_col_num + H_col_num - 1))
print('my result matrix: \n', result_matrix)


# összehasonlítás a convolve2d() fv eredményével
result = signal.convolve2d (U, H, "full")
print('\nresult with convolve2d(): \n', result)