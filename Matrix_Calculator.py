import numpy as np

# Ask for matrix dimensions
while True:
    try:
        rows = int(input("Enter number of rows: "))
        cols = int(input("Enter number of columns: "))
        break
    except ValueError:
        print("Invalid input! Please enter integers for rows and columns.")

# Input matrix
matrix = []
for i in range(rows):
    while True:
        try:
            row = list(map(float, input(f"Enter row {i+1} (separate elements with space): ").split()))
            if len(row) != cols:
                raise ValueError
            matrix.append(row)
            break
        except ValueError:
            print(f"Invalid input! Please enter {cols} numbers for row {i+1}.")

# Convert to numpy array
try:
    matrix = np.array(matrix)
except Exception as e:
    print(f"Error converting to numpy array: {e}")

# Matrix operations
try:
    print("Input Matrix:")
    print(matrix)
    print("\nMatrix Transpose:")
    print(matrix.T)
    if rows == cols:
        print("\nMatrix Inverse:")
        print(np.linalg.inv(matrix))
        print("\nMatrix Determinant:")
        print(np.linalg.det(matrix))
    else:
        print("Not a square matrix, cannot compute inverse or determinant")
except Exception as e:
    print(f"Error performing matrix operations: {e}")