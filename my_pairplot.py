'''pairplot with autmatic axis limits to catch fraction of population'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def my_pairplot(df):
    names = df.colmuns.values
    g = sns.pairplot(df)
    for i in range(len(names)-1):
        g.axes[i+1,0].set_xlim()