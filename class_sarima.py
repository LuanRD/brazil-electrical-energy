import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import itertools
import warnings
from classes_eda import Selection
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

"""
Class that aggregates the functions necessary for the implementation of the SARIMA model.
Attributes:
data: str
source: str
"""
class Sarima:
    
    def __init__(self, data, source):
        self.__data = data
        self.__source = source

    @property
    def data(self):
        return self.__data
        
    @property
    def source(self):
        return self.__source

    @property
    def selection(self):
        return self.__selection

    @property
    def order(self):
        return self.__order

    @property
    def seasonal_order(self):
        return self.__seasonal_order

    @property
    def train(self):
        return self.__train

    @property
    def test(self):
        return self.__test

    @property
    def results(self):
        return self.__results

    @property
    def prediction_results(self):
        return self.__prediction_results

    @property
    def prediction(self):
        return self.__prediction

    """
    Function that returns the result of the ADF and KPSS tests.
    Attribute:
    diff: int
    """     
    def adf_kpss(self, diff):
        
        var = []
        adf = []
        pvalor_adf = []
        kpss_stat = []
        pvalor_kpss = []
        is_sig_adf = []
        is_sig_kpss = []

        if diff == 0:
            if (len(self.__data.query(f'{self.__source} == 0')[self.__source]) > 0):
                selection = self.__data.query(f'{self.__source} > 0')[self.__source]
            else:
                selection = self.__data[self.__source]
                
        elif diff == 1:
            if (len(self.__data.query(f'{self.__source} == 0')[self.__source]) > 0):
                selection = np.diff(self.__data.query(f'{self.__source} > 0')[self.__source], 1)
            else:
                selection = np.diff(self.__data[self.__source])
                
        else:
            if (len(self.__data.query(f'{self.__source} == 0')[self.__source]) > 0):
                selection = np.diff(self.__data.query(f'{self.__source} > 0')[self.__source], 2)
            else:
                selection = np.diff(self.__data[self.__source], 2)
                
        results_adf = adfuller(selection, autolag='AIC')
        results_kpss = kpss(selection, regression="c", nlags="auto")
        var.append(self.__source)
        adf.append(results_adf[0])
        pvalor_adf.append(results_adf[1])
        kpss_stat.append(results_kpss[0])
        pvalor_kpss.append(results_kpss[1])
        is_sig_adf.append(results_adf[1] < 0.05)
        is_sig_kpss.append(results_kpss[1] > 0.05)
                
        return pd.DataFrame({'Energy Source': var, 'ADF': adf, 'P-Valor ADF': pvalor_adf, 'Stationary ADF': is_sig_adf, 'KPSS': kpss_stat, 'P-Valor KPSS': pvalor_kpss, 'Stationary KPSS': is_sig_kpss})

    """
    Function that returns graphs of the decomposition of a Time Series in its trend, seasonality and residuals.
    """     
    def seasonal_decompose(self):

        result = seasonal_decompose(Selection(f'{self.__source}', self.__data).select())
        plt.rcParams.update({'figure.figsize': (14,14)})
        result.plot()

        return plt.show()

    """
    Function that selects the SARIMA model parameters with lower AIC values ​​and creates the SARIMA model. She returns the
    optimal parameters for the model and the time required for them to be calculated.
    """     
    def sarima_model(self):

        if (len(self.__data.query(f'{self.__source} == 0')[self.__source]) > 0):
            self.__selection = self.__data.query(f'{self.__source} > 0')[self.__source]
        else:
            self.__selection = self.__data[self.__source]
    
        # Start of the process
        start_time = time.time()
    
        # Separation of test and training datasets
        self.__train = self.__selection[:int(len(self.__selection)*0.825)]
        self.__test = self.__selection[int(len(self.__selection)*0.825):]
    
        # Selection of the best parameters
        p = q = range(0,5)
        d = range(1,2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
        results_table = pd.DataFrame(columns=['pdq','seasonal_pdq','aic'])
    
        for param in pdq:
            for seasonal_param in seasonal_pdq:
                try:
                    model = SARIMAX(self.__train, order=param, seasonal_order=seasonal_param, enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit()
                    results_table = results_table.append({'pdq':param, 'seasonal_pdq':seasonal_param, 'aic':results.aic},ignore_index=True)
                except:
                    continue
                
        results_table = results_table.sort_values(by='aic')
        results_table = results_table.query('aic > 1600')      
        best_parameters = results_table[results_table['aic']==results_table.aic.min()]
        self.__order = best_parameters.pdq.values[0]
        self.__seasonal_order = best_parameters.seasonal_pdq.values[0]

        # End of process
        end_time = time.time() - start_time

        print(f'Time taken to calculate the best parameters: {end_time:.2f} seconds')

        # Creation of the SARIMA model
        model = SARIMAX(self.__train, order=self.__order, seasonal_order=self.__seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        self.__results = model.fit(disp=-1)
    
        # Model prediction
        self.__prediction_results = self.__results.get_prediction(start=self.__test.index[0], end=self.__test.index[-1], dynamic=False)
        self.__prediction = self.__prediction_results.predicted_mean.round()

        return best_parameters

    """
    Function that returns the summary of model results together with their diagnostic graphs.
    """     
    def summary(self):
    
        print(self.__results.summary())
        print(self.__results.plot_diagnostics())


    """
    Function that returns SARIMA model validation metrics.
    """     
    def accuracy(self):

        rmse = np.sqrt(mean_squared_error(self.__test, self.__prediction))
        nrmse = np.sqrt(mean_squared_error(self.__test, self.__prediction))/(max(self.__test)-min(self.__test))
        mape = np.mean(np.abs(self.__prediction - self.__test)/np.abs(self.__test))*100
    
        medidas = {            
            'Root Mean Square Error': rmse,
            'Normalized Root Mean Square Error': nrmse,
            'Mean Absolute Percent Error (%)': mape
            }
    
        return(pd.DataFrame(data=medidas, index=['model']))


    """
    Function that plots a graph comparing the test dataset with that predicted by the model.
    """     
    def prediction_chart(self):

        prediction_ci = self.__prediction_results.conf_int()
    
        data_plot = pd.concat([self.__test, self.__prediction], axis=1)
        data_plot.rename(columns={'Total': 'Real', 'predicted_mean': 'Predict'}, inplace=True)
    
        fig, ax = plt.subplots(figsize=(20,10), dpi= 100)
        sns.lineplot(data=data_plot, palette='gist_heat_r')
        ax.fill_between(prediction_ci.index,
                prediction_ci.iloc[:, 0],
                prediction_ci.iloc[:, 1], color='k', alpha=.2)
    
        # Personalization
        plt.title('Real VS Predict', fontsize=22)
        plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.grid(axis='both', alpha=.3)
        ax.legend(frameon=False, loc=9, ncol=len(data_plot.columns), fontsize='large')
        ax.set_xlabel('')
        ax.set_ylabel('')
            
        # Edge Removal
        plt.gca().spines["top"].set_alpha(0.0)    
        plt.gca().spines["bottom"].set_alpha(0.3)
        plt.gca().spines["right"].set_alpha(0.0)    
        plt.gca().spines["left"].set_alpha(0.3)
    
        return plt.show()


    """
    Function that plots the forecast graph for the next 12 months of the time series.
    """     
    def forecast_chart(self):

        results_forecast = self.__results.get_forecast(steps=len(self.__test)+12)
        forecast = results_forecast.predicted_mean.round()
        forecast_ci = results_forecast.conf_int()
    
        data_plot = pd.concat([self.__selection, forecast], axis=1)
        data_plot.rename(columns={'predicted_mean': 'Forecast'}, inplace=True)
    
        fig, ax = plt.subplots(figsize=(20,10), dpi= 100)
        sns.lineplot(data=data_plot, palette='gist_heat_r')
        ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='k', alpha=.2)
    
        # Personalization
        plt.title('Total Energy Forecast for the next 12 months', fontsize=22)
        plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
        plt.yticks(fontsize=12, alpha=.7)
        plt.grid(axis='both', alpha=.3)
        ax.legend(frameon=False, loc=9, ncol=len(data_plot.columns), fontsize='large')
        ax.set_xlabel('')
        ax.set_ylabel('')
            
        # Edge Removal
        plt.gca().spines["top"].set_alpha(0.0)    
        plt.gca().spines["bottom"].set_alpha(0.3)
        plt.gca().spines["right"].set_alpha(0.0)    
        plt.gca().spines["left"].set_alpha(0.3)
    
        return plt.show()




