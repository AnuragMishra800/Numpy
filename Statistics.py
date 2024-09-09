import numpy as np

while(True):
    def calculator():
        print("It is a basic Statistics Calculator.")

        user = input("Enter a list of numbers seperated by space: ")
        number = list(map(int, user.split())) 

        data = np.array(number)

        choice = int(input(" Enter 1 for Average\n Enter 2 for Median\n Enter 3 for Variance\n Enter 4 for Standard Deviation: "))

        match choice:
            case 1:
                print("The average is: ",np.mean(data))
            case 2:
                print("The median is: ",np.median(data))
            case 3:
                print("The variance is: ",np.var(data))
            case 4:
                print("The Standard Deviation is: ",np.std(data))
            case _:
                print("Enter the invalid choice.") 
    calculator()      