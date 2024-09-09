import numpy as np

array = np.array([1,2,3,4,5,6])
print(array)
matrix = np.array([[1,2,3],[4,5,6]]) 
print(matrix) 
print(array.shape) 
print(array.dtype) 
array2 = np.array(["hello",'hi','bye'])  
print(array.dtype) 

a = np.array([1,2,3])
b = np.array([9,8,7])
print(a+b)
print(a*b)
print(np.sqrt(a)) 
print(np.exp(a)) 

c = np.array([4,6,8,9,1])
print(c[0])
print(c[4])
print(c[1:3])

array2 = np.array([[1,2,3],[4,5,6]])
print(array2.reshape(3,2)) 

zeros = np.zeros((2,3))
print(zeros) 
ones = np.ones((2,3))
print(ones) 
identity = np.eye(4)
print(identity) 

randomNumbers = np.random.random((2,3)) 
print(randomNumbers) 

a1 = np.array([[1,2],[3,4]])
a2 = np.array([[5,6],[7,8]])
print(np.dot(a1,a2)) 

d = np.array([1,2,3,4,5])
print(np.sum(d)) 
print(np.std(d)) 
print(np.mean(d))  

e = np.array([0,1,2,3,4,5,6,7,8])
print(e.reshape(3,3)) 
e = np.arange(9).reshape((3,3)) 
print(e)

f = np.random.rand(2,2) 
print(f) 

g = np.arange(15).reshape((5,3))
h = np.arange(6).reshape((3,2)) 
m = np.matmul(g,h)
print(m) 

m1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]) 
m2 = np.array([[1,2],[3,4],[5,6]]) 
m3 = m1@m2
print(m3) 

a = np.array([1,2,3,4,5])
b = np.array([[3],[5],[7]])
print(a+b)

array = np.array([10,20,30,40,50])
index = np.array([0,2,4])
print(array[index])

array = np.array([2,5,6,8,2,3,9,6,7])
print(array[array > 3]) 

array = np.array([1,2,3,4,5,6])
print(np.sum(array))
print(np.min(array)) 
print(np.max(array)) 
print(np.mean(array)) 


array = np.array([[1,2,3],[4,5,6]])
print(np.sum(array, axis=0))
print(np.sum(array,axis=1))

from numpy.linalg import inv,det 

matrix = np.array([[1,2],[3,4]])
print(inv(matrix)) 
print(det(matrix) )

random_number = np.random.rand(3,3)
print(random_number)

random = np.random.randn(3,3)
print(random) 

array = np.array([1,2,3,4,5,6,7,8,9,10])
np.save('aur_bhai.npy', array)
loaded_array = np.load('aur_bhai.npy')
print(loaded_array) 

def my_function(x):
    return x**2 + x*2 + 1

vect_func = np.vectorize(my_function)
array = np.array([1,2,3,4]) 
print(vect_func(array))

random = np.random.rand(4,4)
print(random)
print(np.min(random, axis=0)) 
print(np.max(random, axis=0)) 
print(np.min(random, axis=1)) 
print(np.max(random, axis=1)) 


matrix = np.ones((5,5))
matrix[1:-1,1:-1] = 0
print(matrix)  

array = np.array([1,2,3,4,5,6])
print(array[-1]) 

random = np.random.randn(5,5) 
print(random) 

matrix1 = np.random.randn(2,2)
matrix2 = np.random.randn(2,2)
multiply = np.matmul(matrix1, matrix2) 
print(multiply) 

checkerboard = np.tile(np.array([[1,0],[0,1]]),(4,4)) 
print(checkerboard) 

array = np.array([1,2,3])
array2 = np.array([4,5,6])
add = np.add(array,array2)
print(add)
print(array+array2)

minus = np.subtract(array,array2)
print(minus)
print(array-array2)

multiply = np.multiply(array,array2)
print(multiply)
print(array@array2)
print(np.matmul(array,array2))

random = np.random.randint(1,100, size = 20)
print(random)
print(np.mean(random))
print(np.min(random))
print(np.std(random))
print(np.max(random)) 
print(np.mean(random)) 
print(np.var(random)) 
print(np.median(random)) 

matrix = np.array([[1,2],[4,5]])
vector = np.array([7,8])
solution = np.linalg.solve(matrix,vector)
# print(solution) 

data = np.array([1, 2, np.nan, 4, 5, np.nan, 7])
mean_value = np.nanmean(data)
data[np.isnan(data)] = mean_value

print("Data with Missing Values Replaced:\n", data)

print('This is some code that i learned from Chat GPT.')