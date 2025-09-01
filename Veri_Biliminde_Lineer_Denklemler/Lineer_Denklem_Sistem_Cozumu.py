import numpy as np 

#Katsayılar matrisi A 
A= np.array([[2,3,-1],
           [4,1,2],
           [-1,2,3]])

#Sabit terimler vektörü
b= np.array([1,-2,3])

A_ters= np.linalg.inv(A)
X=np.matmul(A_ters,b)

print("Denklem sisteminin çözümü:")
print("x_1 =",X[0])
print("x_1 =",X[1])
print("x_1 =",X[2])

def lineer_denklem_sistem_cozumu(katsayilar,sabit_terimler):
    try:
        A= np.array(katsayilar)
        b= np.array(sabit_terimler)
        A_ters= np.linalg.inv(A)
        X=np.matmul(A_ters,b)
        return X
    except np.linalg.LinAlgError: 
        return "Bu denklem sisteminin çözümü yok !"

katsayilar=[[2,3,-1],[4,1,2],[-1,2,3]]
sabit_terimler= [1,-2,3]

cozum= lineer_denklem_sistem_cozumu(katsayilar,sabit_terimler)

print("Denklem sisteminin çözümü:")
print("x_1 =",X[0])
print("x_1 =",X[1])
print("x_1 =",X[2])

x= np.linalg.solve(A,b)

print("Denklem sisteminin çözümü:")
print("x_1 =",X[0])
print("x_1 =",X[1])
print("x_1 =",X[2])
