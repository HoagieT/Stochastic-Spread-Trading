# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:03:22 2020

@author: hogan
"""

from StochasticSpread import *
from WindPy import w
w.start(waitTime=60)
w.isconnected()

temp=w.edb("M0330936,M0096851,S0213048", "2018-08-17", "-0D","Fill=Previous")
market_data = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['TS','TF','T'])

model1=sm.OLS(market_data['TS'], sm.add_constant(market_data[['TF']])).fit()
termspread1 = (market_data['TS']-0.4*market_data['TF'])
termspread1 = pd.DataFrame(data=termspread1.values, index=termspread1.index, columns=['TS-0.4TF'])

A, B, C, D = initial_estimate(termspread1)
res = kalman_filter(termspread1,A,B,C,D,termspread1.iloc[0,0], 0.1)
smoother = Shumway_Stoffer_smoother(res)
em = EM_Algorithm(termspread1.iloc[0:], 1, [A,B,C,D])

plot_spread(em, 1.8, figure_size=[24,16])
print('ADF test p-value: '+str(np.round(ts.adfuller(termspread1.iloc[:,0].values)[1],4)))



model2=sm.OLS(market_data['TF'], sm.add_constant(market_data[['T']])).fit()
termspread2 = (market_data['TF']-0.6*market_data['T'])
termspread2 = pd.DataFrame(data=termspread2.values, index=termspread2.index, columns=['TF-0.6T'])

A, B, C, D = initial_estimate(termspread2)
res = kalman_filter(termspread2,A,B,C,D,termspread2.iloc[0,0], 0.1)
smoother = Shumway_Stoffer_smoother(res)
em = EM_Algorithm(termspread2.iloc[0:], 1, [A,B,C,D])

plot_spread(em, 1.5, figure_size=[24,16])
print('ADF test p-value: '+str(np.round(ts.adfuller(termspread2.iloc[:,0].values)[1],4)))


model3=sm.OLS(market_data['TS'], sm.add_constant(market_data[['T']])).fit()
model3=sm.OLS(market_data['TS'], market_data[['T']]).fit()
termspread3 = (market_data['TS']-0.25*market_data['T'])
termspread3 = pd.DataFrame(data=termspread3.values, index=termspread3.index, columns=['TS-0.25T'])

A, B, C, D = initial_estimate(termspread3)
res = kalman_filter(termspread3,A,B,C,D,termspread3.iloc[0,0], 0.1)
smoother = Shumway_Stoffer_smoother(res)
em = EM_Algorithm(termspread3.iloc[0:], 1, [A,B,C,D])

plot_spread(em, 1.8, figure_size=[24,16])
print('ADF test p-value: '+str(np.round(ts.adfuller(termspread3.iloc[:,0].values)[1],4)))
