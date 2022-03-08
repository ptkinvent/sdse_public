import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.tsa.arima_process as process
import statsmodels.tsa.arima.model as model
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.stattools import kpss, adfuller
import pandas as pd

""
def make_process(ar,ma):
    ar = np.array(ar)
    ma = np.array(ma)
    ar = np.r_[1, -ar]
    ma = np.r_[1, ma]
    return process.ArmaProcess(ar=ar,ma=ma)

""
def plot_arma_sample(ar,ma,nsamp=1000,figsize=(12,5)):
    process = make_process(ar=ar,ma=ma) 
    z = process.generate_sample(nsample=nsamp)
    plt.figure(figsize=figsize)
    plt.plot(z,linewidth=2,color='k',markersize=20)
    plt.xlim([0,nsamp])
    plt.grid()

# ######################################
# def plot_data_acf(x,lags=10):

""
def plot_process_acf(process,lags=20,figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.stem(process.acf(lags=lags))
    plt.xticks(ticks=range(0,lags,2))
    plt.ylabel('ACF',fontsize=14)

""
def plot_process_pacf(process,lags=20,figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.stem(process.pacf(lags=lags))
    plt.xticks(ticks=range(0,lags,2))
    plt.ylabel('PACF',fontsize=14)

""
def plot_process_acf_pcf(process,lags=10,figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.stem(process.acf(lags=lags))
    plt.ylabel('ACF')
    plt.subplot(212)
    plt.stem(process.pacf(lags=lags))
    plt.ylabel('PACF')

""
def plot_process_roots(process):
    ar_roots = np.roots(np.r_[1, -process.arcoefs])
    ma_roots = np.roots(np.r_[1, process.macoefs])

    figure, axes = plt.subplots() 
    plt.grid()
    axes.set_aspect( 1 )
    plt.scatter(ar_roots.real,ar_roots.imag,marker='x',color='red',label='AR')
    plt.scatter(ma_roots.real,ma_roots.imag,marker='o',color='green',label='MA')
    axes.add_artist( plt.Circle(( 0.0 , 0.0 ), 1.0,fill=False )) 
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.legend()
    plt.show()

""
def plot_ARMA(ar,ma,nsamp,lags):
    process = make_process(ar=ar,ma=ma) 
    z = process.generate_sample(nsample=nsamp)
    
    plt.figure(figsize=(10,10))

    
    plt.subplot(311)
    plt.plot(z)
    plt.xlim([0,nsamp])
    plt.grid()
    
    if process.isstationary:
        plt.subplot(323)
        plt.stem(process.acf(lags=lags))
        plt.ylabel('ACF')
    
        plt.subplot(325)
        plt.stem(process.pacf(lags=lags))
        plt.ylabel('PACF')
    
    ar_roots = np.roots(np.r_[1, -process.arcoefs])
    ma_roots = np.roots(np.r_[1, process.macoefs])

    axes = plt.subplot(324)
    plt.scatter(ar_roots.real,ar_roots.imag,marker='x',color='red',label='AR')
    plt.scatter(ma_roots.real,ma_roots.imag,marker='o',color='green',label='MA')
    axes.add_artist( plt.Circle(( 0.0 , 0.0 ), 1.0,fill=False )) 
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.legend()

""
def plot_ARMA2(ar,ma,nsamp,lags):
    process = make_process(ar=ar,ma=ma) 
    z = process.generate_sample(nsample=nsamp)
    
    plt.figure(figsize=(12,3))
    plt.plot(z)
    plt.xlim([0,nsamp])
    plt.grid()
    plt.show()
    
    if process.isstationary:

        plt.figure(figsize=(6,3))
        plt.stem(process.acf(lags=lags))
        plt.ylabel('ACF')
    
    
    ar_roots = np.roots(np.r_[1, -process.arcoefs])
    ma_roots = np.roots(np.r_[1, process.macoefs])

    plt.figure(figsize=(5,5))
    axes = plt.subplot()
    plt.scatter(ar_roots.real,ar_roots.imag,marker='x',color='red',label='AR')
    plt.scatter(ma_roots.real,ma_roots.imag,marker='o',color='green',label='MA')
    axes.add_artist( plt.Circle(( 0.0 , 0.0 ), 1.0,fill=False )) 
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.legend()

""
def kpss_test(x):

    result = kpss(x)

    output = pd.Series({
                     'Test statistic' : result[0],
                     'p-value' : result[1],
                     '#Lags' : result[2],
                     'Crit val. 1%': result[3]['1%'],
                     'Crit val. 5%': result[3]['5%'],
                     'Crit val. 10%': result[3]['10%'] })

    if output['Test statistic'] > output['Crit val. 1%']:
        print('The null hypothesis can be rejected with alpha=1%')
        print('The series is NOT stationary with 99% confidence.')

    elif output['Test statistic'] > output['Crit val. 5%']:
        print('The null hypothesis can be rejected with alpha=5%')
        print('The series is NOT stationary with 95% confidence.')

    elif output['Test statistic'] > output['Crit val. 10%']:
        print('The null hypothesis can be rejected with alpha=10%')
        print('The series is NOT stationary with 90% confidence.')

    else:
        print('The null hypothesis cannot be rejected.')
        print('The series is stationary.')

""
def adf_test(x):

    result = adfuller(x)

    output = pd.Series({
                     'Test statistic' : result[0],
                     'p-value' : result[1],
                     '#Lags' : result[2],
                     'Crit val. 1%': result[4]['1%'],
                     'Crit val. 5%': result[4]['5%'],
                     'Crit val. 10%': result[4]['10%'] })

    if output['Test statistic'] < output['Crit val. 1%']:
        print('The null hypothesis can be rejected with alpha=1%')
        print('The series is stationary with 99% confidence.')

    elif output['Test statistic'] < output['Crit val. 5%']:
        print('The null hypothesis can be rejected with alpha=5%')
        print('The series is stationary with 95% confidence.')

    elif output['Test statistic'] < output['Crit val. 10%']:
        print('The null hypothesis can be rejected with alpha=10%')
        print('The series is stationary with 90% confidence.')

    else:
        print('The null hypothesis cannot be rejected.')
        print('The series is not stationary.') 

""
def extrapolate_trend(trend, npoints):
    """
    Replace nan values on trend's end-points with least-squares extrapolated
    values with regression considering npoints closest defined points.
    """
    front = next(i for i, vals in enumerate(trend)
                 if not np.any(np.isnan(vals)))
    back = trend.shape[0] - 1 - next(i for i, vals in enumerate(trend[::-1])
                                     if not np.any(np.isnan(vals)))
    front_last = min(front + npoints, back)
    back_first = max(front, back - npoints)

    k, n = np.linalg.lstsq(
        np.c_[np.arange(front, front_last), np.ones(front_last - front)],
        trend[front:front_last], rcond=-1)[0]
    extra = (np.arange(0, front) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[:front] = extra

    k, n = np.linalg.lstsq(
        np.c_[np.arange(back_first, back), np.ones(back - back_first)],
        trend[back_first:back], rcond=-1)[0]
    extra = (np.arange(back + 1, trend.shape[0]) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[back + 1:] = extra

    return trend
