import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file = 'training_metrics/epoch_metrics.txt'

df = pd.read_csv(file)
metrics = ['Loss', 'Fscore', 'Iou Score']

count = 0
for i in range(1,len(df.columns),2):
    fig, ax = plt.subplots()
    ax.plot(df.loc[:,'epoch'],df.loc[:,df.columns[i]],label='Training')
    ax.plot(df.loc[:,'epoch'],df.loc[:,df.columns[i+1]],label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metrics[count])
    ax.set_title(metrics[count])
    ax.legend()
    plt.savefig('plots/' + metrics[count] + '.png')
    count+=1
