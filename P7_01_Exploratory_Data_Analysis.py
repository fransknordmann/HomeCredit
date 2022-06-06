'''
P7_01_Exploratory_Data_Analysis

Data exploration and preparation for classification
'''
# Import default libraries
import pandas as pd
import numpy as np
import re
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Imports scikit learn
from sklearn.preprocessing import LabelEncoder 

# tools for execution time estimates
from datetime import datetime, timedelta
# Remove some warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas parameters
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)
# Matplotlib and sns visual parameters
sns.set_palette("Set2")
sns.set_style("white")
sns.set_context("paper")
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11

# Read files from Sources directory

df = pd.read_csv("./Sources/application_train.csv")

print(df.info())

column_list = df.columns.values.tolist()
for column_name in column_list:
    print(f'col = {column_name}, unique val = {len(df[column_name].unique())}')
