import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from attributes import DATASET_ATTRS ,DATASET_FILE, DATASET_FILE_NUM, DATASET_HEADERS, DATASET_CLASS


from scipy import stats
from sklearn import preprocessing


def get_analysis(dataset, attr_names, class_name):
    possible_values = {}

    # Posibles valores de cada atributo
    for attr in attr_names:
        possible_values[attr] = dataset[attr].unique()

    # Cantidad de valores faltantes por atributo
    missing_values_per_attr = dataset.isna().sum()

    # Cantidad de valores faltantes
    missing_values = dataset.isna().sum().sum()

    # Cantidad de instancias con valores faltantes
    missing_instances = dataset.shape[0] - dataset.dropna().shape[0] # dropna() remueve las filas con valores NaN

    analysis = {
        'instances': dataset.shape[0],      # Cantidad de instancias totales del dataset
        'attributes': dataset.shape[1],     # Cantidad de atributos totales del dataset
        'class_distribution': dataset[class_name].value_counts(),   # Cantidad de instancias por cada idelogía
        'possible_values': possible_values,     # Posibles valores que puede tomar cada atributo
        'missing_values_per_attr': missing_values_per_attr,
        'missing_values': missing_values,
        'missing_instances': missing_instances
    }
    
    return analysis


def get_plots(dataset):

    scatter_plots = ['income','age-group']
    bar_plots = ['race', 'religion','education','ideology']

    for attr in bar_plots:
        values = dataset[attr].value_counts()
        x = values.index
        y = values.array
        plt.bar(x, y)
        plt.suptitle(attr)
        plt.xticks(range(min(x), max(x) + 1))
        plt.show()

    for attr in scatter_plots:
        values = dataset[attr].value_counts()
        plt.scatter(values.index, values.array)
        plt.suptitle(attr)
        plt.show()

    attrs = copy.deepcopy(DATASET_ATTRS)
    attrs.remove('case-id')

    for attr in attrs:
        for val in dataset[attr].unique():
            x = dataset[dataset[attr] == val]['ideology'].value_counts().index
            y = dataset[dataset[attr] == val]['ideology'].value_counts()
            plt.bar(x, y, color='blue')
            plt.suptitle("%s = %s" % (attr, val))
            plt.xticks(range(min(x), max(x) + 1))
            plt.savefig('graficas/%s_%s_ideology' % (attr, val))
            
    #combine plots
    # attr = 'race'
    # val = 1
    # x = dataset[dataset[attr] == val]['ideology'].value_counts().index
    # y = dataset[dataset[attr] == val]['ideology'].value_counts()
    # plt.bar(x, y, color='blue', alpha=0.3)
    # x = dataset[dataset[attr] == val]['ideology'].value_counts().index
    # y = dataset[dataset[attr] == val]['ideology'].value_counts()

    # val = 2
    # x = dataset[dataset[attr] == val]['ideology'].value_counts().index
    # y = dataset[dataset[attr] == val]['ideology'].value_counts()
    # plt.bar(x, y, color='red', alpha=0.7)
    # plt.suptitle("%s = %s" % (attr, val))
    # plt.xticks(range(int(min(x)), int(max(x)) + 1))
    # plt.show()

def kdeplot_vs_ideology(dataset_num,attrs):
    for attr in attrs:
        for val in dataset_num['ideology'].unique():
                lo=dataset_num[dataset_num['ideology'] == val][attr]
                sns.kdeplot(lo,shade="true",legend=False)
                plt.legend(['ideology=%s'%val for val in dataset_num['ideology'].unique()])
                plt.xlabel(attr)
        plt.show()
        #plt.savefig('graficas/%s_ideology' % (attr))

def histogram_vs_ideology(dataset_num,attrs):
    for attr in attrs:
        for val in dataset_num['ideology'].unique():
                lo=dataset_num[dataset_num['ideology'] == val][attr]
                sns.distplot(lo,kde=False,)
                plt.legend(['ideology=%s'%val for val in dataset_num['ideology'].unique()])
                plt.xlabel(attr)
        plt.show()
        #plt.savefig('graficas/%s_ideology' % (attr))

def catplot_swarm_ideology(dataset_num):
    sns.catplot(x='race',y='religion',data=dataset_num,kind='swarm',hue='ideology')
    plt.show()
    sns.catplot(x='race',y='education',data=dataset_num,kind='swarm',hue='ideology')
    plt.show()
    sns.catplot(x='race',y='age-group',data=dataset_num,kind='swarm',hue='ideology')
    plt.show()

def mean_var_std(dataset_num):
     print(dataset_num.agg([np.mean, np.var,np.std]))

def histogramas_ideology(dataset_num,attrs):
    for attr in attrs:
        for val in dataset_num['ideology'].unique():
            x=dataset_num[dataset_num['ideology'] == val][attr].value_counts().index
            y=dataset_num[dataset_num['ideology'] == val][attr].value_counts()
            plt.bar(x, y)
            plt.legend(['ideology=%s'%val for val in dataset_num['ideology'].unique()])
        plt.suptitle("%s" % (attr))
        plt.show()

def histogramas_gral(dataset_num,attrs):
    for i in attrs:
        for attr in attrs:
            for val in dataset_num[attr].unique():
                x = dataset_num[dataset_num[attr] == val][i].value_counts().index
                y = dataset_num[dataset_num[attr] == val][i].value_counts()
                plt.bar(x, y, color='blue')
                plt.suptitle("%s = %s" % (attr, val))
                #plt.xticks(range(min(x), max(x) + 1))
                plt.savefig('graficas/%s_%s_%s' % (attr, val,i))

def Box_plot(dataset_num,attrs):
    for i in attrs:
        sns.catplot(x=i,y='ideology',kind='box',data=dataset_num,orient='h')
        plt.show()

