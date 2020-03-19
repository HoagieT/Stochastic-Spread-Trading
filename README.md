# Stochastic Spread Trading

This project implements the framework described in "[Pairs Trading](http://stat.wharton.upenn.edu/~steele/Courses/434/434Context/PairsTrading/PairsTradingQFin05.pdf)" by Elliott et al.
The model is a statistical arbitrage algorithm based on mean reversion stochastic process and Kalman Filter. When implemented in China's interest rate futures and commodity futures, the model generates stable and significant returns. 


A chart for trading indicator is also designed in this project.

**Chart Notes:** 
1. Red line represents the priori Kalman Filter prediction, i.e. the fair value of spreads.
2. Black line represents the observed spreads calculated with close price each day.
3. Blue area represents trading threshold, trade is triggered when black line breaks out of blue area.


## Download instructions

Download the code as a ZIP file by clicking the green 'Clone or download' button and selecting 'Download ZIP'.

## File and folder description

* `rates_future.xlsx` : example China interest rate future market data
* `Functions` : functions for importing data, estimating model, and a few miscellaneous funtions that may or may not be used in the model
* `StochasticSpread.py` : the main script of stochastic spread trading tool kit, including AR(1) model, Kalman Filter, EM algorith, charts, etc.
* `butterfly_spread.py` : the main example script to implement stochastic spread model on China's interest rate future
* `term_spread.py` : another example
* `commodities.py` : another example with significant higher expected returns as well as volitilities

**Notes:**Wind database and WindPy API are strongly recommended to extract market data in China's futures market.

## Required software and versioning

Python 3 is required to run the code. The code was tested in Python 3.6 and later versions. Functionality with earlier versions of Python is not guaranteed.
