# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:28:47 2019

@author: Zenith
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

address = "//ad.uillinois.edu/engr-ews/zzhou64/Documents/GitHub/SMA-HullTrading-Practicum/Data/Cleaned"

train_x = address + "train_x.txt"
train_y = address + "train_y.txt"
test_x = address + "test_x.txt"
test_y = address + "test_y.txt"