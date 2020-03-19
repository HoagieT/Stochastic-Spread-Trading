# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:36 2020

@author: 029046
"""

from StochasticSpread import *
from WindPy import w
w.start(waitTime=60)
w.isconnected()

temp=w.edb("S0186244,S0181380,S0181383,S0181373,S0213046,S0213047,S0181384,S0181381", "2018-08-01", "-0D","Fill=Previous")
market_data = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Iron','Coke','Rebar','HS300','IC500','SSE50','Cotton','PTA'])

model1=sm.OLS(market_data['Cotton'], sm.add_constant(market_data[['PTA']])).fit()
#model1=sm.OLS(market_data['Cotton'], market_data[['PTA']]).fit()
model1.summary()
commodityspread1 = (market_data['Cotton']-1.3*market_data['PTA'])
commodityspread1 = pd.DataFrame(data=commodityspread1.values, index=commodityspread1.index, columns=['Cotton-1.3PTA'])

A, B, C, D = initial_estimate(commodityspread1)
res = kalman_filter(commodityspread1,A,B,C,D,commodityspread1.iloc[0,0], 0.1)
smoother = Shumway_Stoffer_smoother(res)
em = EM_Algorithm(commodityspread1.iloc[0:], 1, [A,B,C,D])

plot_spread(em, 1.8, figure_size=[9,6])
print('ADF test p-value: '+str(np.round(ts.adfuller(commodityspread1.iloc[:,0].values)[1],4)))


model2=sm.OLS(market_data['Rebar'], sm.add_constant(market_data[['Iron','Coke']])).fit()
#model2=sm.OLS(market_data['Rebar'], market_data[['Iron','Coke']]).fit()
model2.summary()
commodityspread2 = (market_data['Rebar']-0.6*market_data['Iron']-1.2*market_data['Coke'])
commodityspread2 = pd.DataFrame(data=commodityspread2.values, index=commodityspread2.index, columns=['Rebar-0.6Iron-1.2Coke'])

A, B, C, D = initial_estimate(commodityspread2)
res = kalman_filter(commodityspread2,A,B,C,D,commodityspread2.iloc[0,0], 0.1)
smoother = Shumway_Stoffer_smoother(res)
em = EM_Algorithm(commodityspread2.iloc[0:], 1, [A,B,C,D])

plot_spread(em, 1.6, figure_size=[9,6])
print('ADF test p-value: '+str(np.round(ts.adfuller(commodityspread2.iloc[:,0].values)[1],4)))