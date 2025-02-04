import random
import time
import string
import numpy as np

def generate_custom_id():
    return np.random.randint(int(1e+8))

def progress_bar(it,num_it):
    """
    A simple progress bar for the console.
    
    Parameters
    ----------
    it : int
        The current iteration.
    num_it : int
        The total number of iterations.
    """
    percentage = (it / (num_it - 1)) * 100
    print(f"\r{int(percentage)}%|", end="")
    print(int(percentage/5)*"■",end="")
    print((20-int(percentage/5))*"□"+"|",end="")


def binary_sum(n):
    sum_rule=[]
    for i in range(n):
        j=i+1
        num_sum=0
        while j%2 == 0:
             num_sum+=1
             j=j/2
             if j==0:
                break
        sum_rule.append(num_sum)
    return sum_rule