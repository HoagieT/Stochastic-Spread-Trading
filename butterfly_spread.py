# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:46:30 2020

@author: hogan
"""
from StochasticSpread import *
from WindPy import w # import Wind database, please install Wind on your device and repair the WindPy API
from Functions import *
w.start(waitTime=60)
w.isconnected()

temp=w.edb("M0330936,M0096851,S0213048", "2018-08-17", "-0D","Fill=Previous")
market_data = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['TS','TF','T'])

#alternatively, use the following command to import market data
#market_data=import_data('rates_future.xlsx', 'close_price', start=0, interpolation=False, encoding='gb18030')


#use OLS regression to construct linear combination of futures
#model1=sm.OLS(market_data['TS'], sm.add_constant(market_data[['TF','T']])).fit()
#model2=sm.OLS(market_data['TS'], market_data[['TF','T']]).fit()

butterfly = (market_data['TS']-1.7732*market_data['TF']+0.7801*market_data['T'])
butterfly = pd.DataFrame(data=butterfly.values, index=butterfly.index, columns=['TS-1.77TF+0.78T'])


A, B, C, D = initial_estimate(butterfly)
res = kalman_filter(butterfly,A,B,C,D,butterfly.iloc[0,0], 0.1)
smoother = Shumway_Stoffer_smoother(res)
em = EM_Algorithm(butterfly.iloc[-100:], 1, [A,B,C,D])

plot_spread(em, 1.5, figure_size=[9,6])
print('ADF test p-value: '+str(np.round(ts.adfuller(butterfly.iloc[:,0].values)[1],4)))



