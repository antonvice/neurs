#! main.py

#Imports
import numpy as np
import pandas as pd
from matplotlib.pyplot import pyplot as plt

#import mnist digit recognition dataset
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data)
