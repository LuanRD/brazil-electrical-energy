import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
Class to obtain the null values ​​of each variable.
Attributes:
source: str
data: str
"""
class Null:
    
    def __init__(self, source, data):
        self.__source = source
        self.__data = data
        
    @property
    def source(self):
        return self.__source
    
    @property
    def data(self):
        return self.__data
    
    """
    Function that returns the null values ​​of the variable.    """
    def null(self):
        null_values = self.__data.query(f'{self.__source} == 0')[self.__source]
        if len(null_values) > 0:
            return self.__data.query(f'{self.__source} == 0')[self.__source]
        else:
            return null_values

"""
Class that selects the non-null values ​​of each variable.
Attributes:
source: str
data: str
"""            
class Selection:

    def __init__(self, source, data):
        self.__source = source
        self.__data = data
        
    @property
    def source(self):
        return self.__source
    
    @property
    def data(self):
        return self.__data
    
    """
    Function that selects the non-null values ​​of each variable.
    """
    def select(self, dataframe=None):

        if dataframe == True:

            if len(Null(f'{self.__source}', self.__data).null() > 0):
                self.__source = self.__data.query(f'{self.__source} > 0')
            else:
                self.__source = self.__data
            return self.__source

        else:
            if len(Null(f'{self.__source}', self.__data).null() > 0):
                self.__source = self.__data.query(f'{self.__source} > 0')[self.__source]
            else:
                self.__source = self.__data[f'{self.__source}']
            return self.__source

"""
Class that selects the non-null values ​​of the variable in the "sum" dataframe.
Attributes:
source: str
data: str
sum: str
""" 
class Sum_Selection:
    
    def __init__(self, source, sum):
        self.__source = source
        self.__sum = sum
        
    @property
    def source(self):
        return self.__source
    
    @property
    def sum(self):
        return self.__sum
    
    """
    Function that selects the non-null values ​​of the variable in the "sum" dataframe.
    """
    def select(self):
        if len(self.__sum.query(f'{self.__source} == 0')[self.__source]) > 0:
            self.__source = self.__sum.query(f'{self.__source} > 1')
        else:
            self.__source = self.__sum
        return self.__source

"""
Class that selects the non-null values ​​of each variable in the "sum_perc" dataframe.
Attributes:
source: str
data: str
sum: str
sum_perc: str
""" 
class Sum_Selection_Perc:
    
    def __init__(self, source, sum, sum_perc):
       self.__source = source
       self.__sum = sum
       self.__sum_perc = sum_perc
        
    @property
    def source(self):
        return self.__source
    
    @property
    def sum(self):
        return self.__sum
    
    @property
    def sum_perc(self):
        return self.__sum_perc
    
    """

    Function that selects the non-null values ​​of the variable in the "sum_perc" dataframe.    """
    def select(self):
          l = []
          for i in self.__sum.query(f'{self.__source} > 1').index:
               l.append(i)
               data = self.__sum_perc.loc[l[0]:l[-1]]
          return data       

"""
Class that aggregates functions referring to the statistical analysis of each variable.
Attributes:
source: str
data: str
""" 
class Stats:
    
    def __init__(self, source, data):
        self.__source = source
        self.__data = data
        
    @property
    def source(self):
        return self.__source
    
    @property
    def data(self):
        return self.__data
    
    """
    Function that returns the description of the variable's statistical parameters.
    """
    def description(self):
        describe = (Selection(f'{self.__source}', self.__data).select().describe())
        return pd.DataFrame(describe)
        
    """
    Function that returns the outlier values ​​of the variable.
    """
    def get_outliers(self):
        FIQ = Selection(f'{self.__source}', self.__data).select().describe()['75%'] - Selection(f'{self.__source}', self.__data).select().describe()['25%']
        inf = Selection(f'{self.__source}', self.__data).select().describe()['25%'] - 1.5*FIQ
        sup = Selection(f'{self.__source}', self.__data).select().describe()['75%'] + 1.5*FIQ
    
        out_inf = pd.DataFrame(self.__data.query(f'{self.__source} < {inf}')[self.__source])
        out_sup = pd.DataFrame(self.__data.query(f'{self.__source} > {sup}')[self.__source])

        if (len(out_inf) > 0) & (len(out_sup) > 0):
            return out_inf, out_sup
        elif (len(out_sup) > 0) & (len(out_inf) == 0):
            return out_sup
        elif (len(out_sup) == 0) & (len(out_inf) > 0):
            return out_inf
        else:
            return print('Sem Outliers')

"""
Class that aggregates functions related to plotting the graphs of each variable.
Attributes:
source: str
data: str
sum: str
sum_perc: str
""" 
class Charts:
    
    def __init__(self, source, data, sum, sum_perc):
        self.__source = source
        self.__data = data
        self.__sum = sum
        self.__sum_perc = sum_perc
        
    @property
    def source(self):
        return self.__source
    
    @property
    def data(self):
        return self.__data
    
    @property
    def sum(self):
        return self.__sum
    
    @property
    def sum_perc(self):
        return self.__sum_perc

    """
    Function that plots a "boxplot" analyzing the complete dataset.
    """
    def boxplot(self):
    
        # Chart Plot
        fig, ax = plt.subplots(figsize=(20,10))
        ax = sns.boxplot(data=Selection(f'{self.__source}', self.__data).select(), orient='h', palette='gist_heat_r')
    
        # Personalization
        ax.tick_params(labelsize=16)
        title = self.__source.replace('_',' ')
        ax.set_title(f'{title} (GWh)',fontsize=24)
        ax.set_xlabel('Dispatched Energy (GWh)', fontsize=18)
        sns.color_palette("GnBu", as_cmap=True)

        return plt.show()

    """
    Function that plots "boxplots" referring to each month.
    """
    def boxplot_monthly(self):
        
        # Chart Plot
        fig, ax = plt.subplots(figsize=(20,10))
        ax = sns.boxplot(data=Selection(f'{self.__source}', self.__data).select(dataframe=True), y=self.__source, x='month', orient='v', palette='gist_heat_r')
        
        # Personalization
        ax.tick_params(labelsize=16)
        title = self.__source.replace('_',' ')
        ax.set_title(f'{title} (GWh)',fontsize=24)
        ax.set_xlabel('Month', fontsize=18)
        ax.set_ylabel('Dispatched Energy (GWh)', fontsize=18)
        sns.color_palette("YlOrBr", as_cmap=True)
        
        return plt.show()

    """
    Function that plots a line graph.
    """    
    def lineplot(self):
        
        # Chart Plot
        fig,ax = plt.subplots(figsize=(20,10), dpi= 100)
        sns.lineplot(data=Selection(f'{self.__source}', self.__data).select(), palette='gist_heat_r', color='darkred')
    
        # Personalization
        title = self.__source.replace('_',' ')
        plt.title(f'{title} (GWh)', fontsize=22)
        plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.grid(axis='both', alpha=.3)
        ax.set_xlabel('')
        ax.set_ylabel('')
            
        # Edge Removal
        plt.gca().spines["top"].set_alpha(0.0)    
        plt.gca().spines["bottom"].set_alpha(0.3)
        plt.gca().spines["right"].set_alpha(0.0)    
        plt.gca().spines["left"].set_alpha(0.3)
    
        return plt.show()
    
    """
    Function that plots a column chart together with a line chart, both with separate y-axes.
    """
    def mixedplot(self):       
   
        # Chart Plot
        fig, ax1 = plt.subplots(figsize=(20,10))
        sns.barplot(data = Sum_Selection(f'{self.__source}', self.__sum).select(), x=Sum_Selection(f'{self.__source}', self.__sum).select().index.year.astype('string'), y=self.__source, alpha=0.5, ax=ax1, color = 'orangered')
        ax2 = ax1.twinx()
        sns.lineplot(data = Sum_Selection_Perc(f'{self.__source}', self.__sum, self.__sum_perc).select(), x = Sum_Selection(f'{self.__source}', self.__sum).select().index.year.astype('string'), y = f'{self.__source}_perc', marker='o', sort = False, ax=ax2, color='darkred')

        # Personalization
        title = self.__source.replace('_',' ')
        plt.title(f'Yearly Evolution: {title}', fontsize=22)
        plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.grid(axis='both', alpha=.3)
        ax1.set_xlabel('')
        ax1.set_ylabel('Dispatched Energy (GWh)', fontsize=18)
        ax2.set_ylabel('Percentage Change (%)', fontsize=18)
    
        # Line indicating the percentage y-axis
        plt.axhline(c='black', ls='--')
    
    """
    Function that plots a stacked area graph.
    Attribute:
    selection: pandas.DataFrame
    """
    @classmethod
    def areaplot(cls, selection):
    
        # Creation of the Data Selection DataFrame
        col = selection.columns
        n = len(col)-1

        l = []
        labels = []
        for j in range(len(col)-1):
            perc = selection[col[j+1]]/selection[col[0]]*100
            l.append(perc)
            labels.append(col[j+1])
            for k in range(len(labels)):
                labels[k] = labels[k].replace('_', ' ')
        
            selection = pd.concat([selection, perc], axis=1)
            selection = selection.rename(columns = {0: f'{col[j+1]}_per'})

        selection = selection.reset_index()
    
        # Chart Plot
        fig,ax = plt.subplots(figsize=(18,9.8), dpi= 100)
        colors = sns.color_palette('gist_heat_r', n)
        plt.stackplot(selection.date, l, labels=labels, colors=colors)
       
        # Personalization
        title = col[0].replace('_',' ')
        plt.title(f'Percent Decomposition: {title} (%)', fontsize=22)
        plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.grid(axis='both', alpha=.3)
        ax.legend(frameon=False, loc=9, ncol=n, fontsize='large')      
        ax.set_xlabel('')
        ax.set_ylabel('')
    
        # Edge Removal
        plt.gca().spines["top"].set_alpha(0.0)    
        plt.gca().spines["bottom"].set_alpha(0.3)
        plt.gca().spines["right"].set_alpha(0.0)    
        plt.gca().spines["left"].set_alpha(0.3)
    
        return plt.show()

"""
Class that aggregates functions referring to the evaluation functions of the correlation between the variables.
Attributes:
selection: pandas.DataFrame
""" 
class Correlation:
    
    def __init__(self, selection):
        self.__selection = selection
        
    @property
    def selection(self):
        return self.__selection
    
    """
    Function that return graphs of correlation between variables in matrix form, so that they are plotted:
    Inferior to the diagonal - Scatter plot between two variables;
    Diagonal - Histogram of that variable;
    Superior to the diagonal - Distribution graph between two variables of the KDE type (Kernel Density Estimate).
    """
    def pairgrid(self):
    
        g = sns.PairGrid(self.__selection, diag_sharey=False)
        g.map_upper(sns.scatterplot, color='darkred')
        g.map_lower(sns.kdeplot, cmap = 'Reds')
        g.map_diag(sns.kdeplot, color='darkred')
    
    """
    Function that returns a heat map of selected variables.
    """
    def heatmap(self):
        
        corrv=np.corrcoef(self.__selection, rowvar=False)
        mask = np.triu(np.ones_like(np.corrcoef(corrv, rowvar=False)))
        fig, ax = plt.subplots(figsize=(10,6), dpi= 100)
        heatmap = sns.heatmap(corrv, annot=True, linewidths=.5, xticklabels = self.__selection.columns, yticklabels = self.__selection.columns, fmt='.2g', mask=mask, ax=ax)

    """
    Function that returns specific correlation values ​​between variables.
    Attributes:
    number: float (correlation value used for comparison)
    relation: str('>' or '<')
    """
    def select_corr(self, number, relation):    
        corrr = self.__selection.corr().values
        col = self.__selection.columns
        la = []
        lb = []
        lc = []
        r, c = self.__selection.shape
        for i in range(c):
            for j in range(i+1, c):
                if relation == '>':
                    if corrr[i, j] > number:
                        la.append(col[i])
                        lb.append(col[j])
                        lc.append((corrr[i, j]))
                if relation == '<':
                    if corrr[i, j] < number:
                        la.append(col[i])
                        lb.append(col[j])
                        lc.append((corrr[i, j]))
        dfe = pd.DataFrame({
            'Variable 1': la,
            'Variable 2': lb,
            'Correlation': lc
        })
        dfe = dfe.sort_values(by='Correlation', ascending=False)
        dfe.reset_index(drop=True, inplace=True)
        return dfe