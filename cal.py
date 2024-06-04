import numpy as np
#Calculate the mean and standard division of the obtained results
def calculate_mean_and_std(numbers):
    numbers_array = np.array(numbers)  
    mean = np.mean(numbers_array)      
    std_dev = np.std(numbers_array)    
    return mean, std_dev


numbers = [100,100,98.5]
mean, std_dev = calculate_mean_and_std(numbers)
print("Mean:", mean)
print("Standard Deviation:", std_dev)
