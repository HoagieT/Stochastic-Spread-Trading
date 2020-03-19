# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:21:25 2020

@author: hogan
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import tkinter.filedialog
from datetime import timedelta
import math 
import time
import calendar
from statsmodels.tsa.api import VAR, DynamicVAR
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from datetime import datetime,timedelta
from scipy.optimize import minimize, rosen, rosen_der
from statsmodels.tsa.ar_model import AR

def initial_estimate(x):
    # x_k = A + B*x_{k-1} +C*e_k
    # y_k = x_k + D*w_k
    # x must be a dataframe with timestamp index
    state_equation = AR(x).fit(1)
    A,B = state_equation.params
    C = state_equation.resid.std()
    D = (x-A/(1-B)).std()[0]
    
    return A, B, C, D

class kalman_filter_result():
    def __init__(self, y, x, x_minus, R, R_minus, K, A, B, C, D, index, name):
        self.y = y
        self.x = x
        self.x_minus = x_minus
        self.R = R
        self.R_minus = R_minus
        self.K = K
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.index = index
        self.name = name

def kalman_filter(y,A,B,C,D, initial_state, initial_error):
    # x_k = A + B*x_{k-1} +C*e_k
    # y_k = x_k + D*w_k
    # y must be a dataframe with timestamp as index
    col = y.columns[0]
    x_minus = np.zeros(shape=(len(y.index),1))
    x = np.zeros(shape=(len(y.index),1))
    R_minus = np.zeros(shape=(len(y.index),1))
    R = np.zeros(shape=(len(y.index),1))
    K = np.zeros(shape=(len(y.index),1))
    x[0]=initial_state
    R[0]=initial_error
    x_minus[0] = np.NaN
    
    for i in range(1,len(x)):
        
        # predict step
        x_minus[i] = A + B*x[i-1]
        R_minus[i] = B**2*R[i-1] +C**2
        
        # update
        K[i] = R_minus[i]/(R_minus[i]+D**2)
        x[i] = x_minus[i] + K[i]*(y.iloc[i,0]-x_minus[i])
        R[i] = R_minus[i] - K[i]*R_minus[i]
    
    x = pd.DataFrame(data=x, index=y.index, columns=[col+'.post'])
    x_minus = pd.DataFrame(data=x_minus, index=y.index, columns=[col+'.prior'])
    R = pd.DataFrame(data=R, index=y.index, columns=['R.post'])
    R_minus = pd.DataFrame(data=R_minus, index=y.index, columns=['R.prior'])
    K = pd.DataFrame(data=K, index=y.index, columns=['Kalman gain'])
        
    return kalman_filter_result(y=y, x=x, x_minus=x_minus, R=R, R_minus=R_minus, K=K, A=A, B=B, C=C, D=D, index=y.index, name=y.columns[0])


class Shumway_Stoffer_smoother_result():
    def __init__(self, x_sm, R_sm, R_sm_minus, A, B, C, D):
        self.x_sm = x_sm #smoothed kalman fiter result
        self.R_sm = R_sm
        self.R_sm_minus = R_sm_minus
        self.A = A
        self.B = B
        self.C = C
        self.D = D

def Shumway_Stoffer_smoother(KF_result):
    N=len(KF_result.x)
    A, B, C, D = KF_result.A, KF_result.B, KF_result.C, KF_result.D
    
    x_sm = np.zeros(shape=(N,1))
    R_sm = np.zeros(shape=(N,1))
    R_sm_minus = np.zeros(shape=(N,1))
    J = np.zeros(shape=(N,1))
    
    x = np.mat(KF_result.x)
    x_minus = np.mat(KF_result.x_minus)
    R = np.mat(KF_result.R)
    R_minus = np.mat(KF_result.R_minus)
    K = np.mat(KF_result.K)
    y = np.mat(KF_result.y)
    
    x_sm[N-1] = x[N-1]
    R_sm[N-1] = R[N-1]
    R_sm_minus[N-2] = B*(1-K[N-1])*R[N-2]
    
    # Smoothing step
    for i in range(N-2,-1,-1):
        J[i] = B*R[i]/R_minus[i+1]
        x_sm[i] = x[i] + J[i]*(x_sm[i+1]-(A+B*x[i]))
        R_sm[i] = R[i] + J[i]**2*(R_sm[i+1]-R_minus[i+1])
        
    for i in range(N-3,-1,-1):
        R_sm_minus[i] = J[i]*R[i+1] + J[i]*J[i+1]*(R_sm_minus[i+1]-B*R[i+1])
        
    # Estimation of parameters
    alpha=0.0
    beta=0.0
    gamma=0.0
    delta=0.0
    
    for i in range(1,N):
        alpha = alpha + R_sm[i-1] + x_sm[i-1]**2
        beta = beta + R_sm_minus[i-1] + x_sm[i-1]*x_sm[i]
        gamma = gamma + x_sm[i]
        delta = delta + x_sm[i-1]
    
    A_hat = (alpha*gamma-delta*beta)/(N*alpha-delta**2)
    B_hat = (N*beta-gamma*delta)/(N*alpha-delta**2)
    
    C_hat_sq = 0.0
    D_hat_sq = 0.0
    
    for i in range(1,N):
        C_hat_sq = C_hat_sq+ (R_sm[i] + x_sm[i]**2 + A_hat**2 + B_hat**2*R_sm[i-1] + B_hat**2*x_sm[i-1]**2 - 2*A_hat*x_sm[i] + 2*A_hat*B_hat*x_sm[i-1] -2*B_hat*R_sm_minus[i-1] - 2*B_hat*x_sm[i]*x_sm[i-1])/(N-1)
        D_hat_sq = D_hat_sq + (y[i]**2 - 2*y[i]*x_sm[i] + R_sm[i] + x_sm[i]**2)/N
    
    x_sm = pd.DataFrame(data=x_sm, index=KF_result.index, columns=[KF_result.name+'.sm'])
    R_sm = pd.DataFrame(data=R_sm, index=KF_result.index, columns=['R.sm'])
    R_sm_minus = pd.DataFrame(data=R_sm_minus, index=KF_result.index, columns=['R.sm_minus'])
    
    
    return Shumway_Stoffer_smoother_result(x_sm=x_sm, R_sm=R_sm, R_sm_minus=R_sm_minus, A=A_hat[0], B=B_hat[0], C=np.sqrt(C_hat_sq)[0], D=np.sqrt(D_hat_sq)[0,0])
        
       
def EM_Algorithm(y, n_iter, initial_estimation):
    # initial_estimate must be a vector of 4 in the form of [A,B,C,D]
    A,B,C,D = initial_estimation
    KF = kalman_filter(y,A,B,C,D,y.iloc[0,0],D**2)
    SS = Shumway_Stoffer_smoother(KF)
    for i in range(1,n_iter):
        KF = kalman_filter(y,SS.A,SS.B,SS.C,SS.D, SS.x_sm.iloc[0,0], SS.R_sm.iloc[0,0])
        SS = Shumway_Stoffer_smoother(KF)
    
    return EM_Algorithm_result(A=SS.A,B=SS.B,C=SS.C,D=SS.D,x_sm=SS.x_sm, y=y, x=KF.x, x_minus=KF.x_minus)

class EM_Algorithm_result():
    def __init__(self, A, B, C, D, x_sm, x, y, x_minus):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x_sm = x_sm #smoothed x
        self.y = y
        self.x = x
        self.x_minus = x_minus
        

def plot_spread(KF_result, multiple=1.0, figure_size=[12,8], line_width=2.0, font_size='xx-large'):
    plt.rcParams['figure.figsize'] = (figure_size[0], figure_size[1])
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['image.cmap'] = 'gray'
    
    index = KF_result.y.index
    forecast = KF_result.x_minus
    observation = KF_result.y
    stddev = KF_result.D
    plt.figure()
    plt.plot(index,forecast,label='forecast',color='r', linewidth=line_width)
    plt.plot(index,observation,label='observed',color='k', linewidth=line_width)
    plt.fill_between(index,(forecast-multiple*stddev).iloc[:,0], (forecast+multiple*stddev).iloc[:,0],color='b', alpha=0.2)
    plt.legend()
    plt.title(observation.columns[0], fontweight='bold', fontsize=font_size)
    plt.show()
    
    return plt

def StochasticSpread(spread, count, n_iter=2):
    A, B, C, D = initial_estimate(spread)
    res = kalman_filter(spread,A,B,C,D,spread.iloc[0,0], 0.1)
    smoother = Shumway_Stoffer_smoother(res)
    em = EM_Algorithm(spread, n_iter, [A,B,C,D])
    kf = kalman_filter(em.y.iloc[-count:],em.A,em.B,em.C,em.D, em.y.iloc[-count,0], 0.1)
    
    return kf