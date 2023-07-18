# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:47:22 2023

@author: sigma
"""

# Import base libraries
from numpy import *
from numpy.linalg import multi_dot
import pandas as pd
import yfinance as yf

# Import optimization module from scipy
import scipy.optimize as sco

# Import cufflinks
import cufflinks as cf
cf.set_config_file(offline=True, dimensions=((1000,600)))

# Import plotly express for EF plot
import plotly.express as px
px.defaults.template, px.defaults.width, px.defaults.height = "plotly_white", 1000, 600


# Nasdaq-listed stocklist
symbols = ['AAPL', 'AMZN', 'PG', 'WMT', 'BA', 'JPM']

# Number of assets
numofasset = len(symbols)

# Number of portfolio for optimization
numofportfolio = 5000

# Fetch data from yahoo finance for last six years
nasdaqstocks = yf.download(symbols, start='2013-02-22', end='2023-03-22', progress=False)['Adj Close']
nasdaqstocks.to_csv(r'D:\Python files\CQF Python 2023 march\03-Portfolio-Optimization\nasdaqstocks.csv')

# Load locally stored data
df= pd.read_csv(r'D:\Python files\CQF Python 2023 march\03-Portfolio-Optimization\nasdaqstocks.csv', index_col=0, parse_dates=True)

# Verify the output
df

# Plot normalize price history
df['2023':].normalize().iplot(kind='line')


# Calculate returns 
returns = df.pct_change().fillna(0)
returns


# Plot annualized return and volatility
pd.DataFrame({
    'Annualized Return': round(returns.mean()*252*100,2),
    'Annualized Volatility': round(returns.std()*sqrt(252)*100,2)
}).iplot(kind='bar', shared_xaxes=True, subplots=True)


# Define Weights for Equal weighted portfolio
wts = array(numofasset * [1./numofasset])[:, newaxis]
wts

# Derive portfolio returns, ret
ret = array(returns.mean()*252)[:, newaxis]
ret

# Portfolio returns using @
wts.T @ ret

# Portfolio Variance & Volatility
cov = returns.cov()*252
var = multi_dot([wts.T, cov, wts])
sqrt(var)


#Portfolio simulation
def portfolio_simulation(returns):

    # Initialize the lists
    rets = []; vols = []; wts = []

    # Simulate 5,000 portfolios
    for i in range(numofportfolio):

        # Generate random weights
        weights = random.random(numofasset)[:, newaxis]

        # Set weights such that sum of weights equals 1
        weights /= sum(weights)

        # Portfolio statistics
        rets.append(weights.T @ array(returns.mean()*252)[:, newaxis])
        vols.append(sqrt(multi_dot([weights.T, returns.cov()*252, weights])))
        wts.append(weights.flatten())

    # Create a dataframe for analysis
    portdf = 100*pd.DataFrame({
        'port_rets': array(rets).flatten(),
        'port_vols': array(vols).flatten(),
        'weights': list(array(wts))
    })
    
    portdf['sharpe_ratio'] = portdf['port_rets'] / portdf['port_vols']

    return round(portdf,2)

# Create a dataframe for analysis
temp = portfolio_simulation(returns)
temp


#Portfolio statsitics function
# Define portfolio stats function
def portfolio_stats(weights):
    
    weights = array(weights)[:,newaxis]
    port_rets = weights.T @ array(returns.mean() * 252)[:,newaxis]    
    port_vols = sqrt(multi_dot([weights.T, returns.cov() * 252, weights])) 
    
    return array([port_rets, port_vols, port_rets/port_vols]).flatten()

# Maximizing sharpe ratio
def min_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]




# Define initial weights
initial_wts = numofasset * [1./numofasset]
initial_wts

# Each asset boundary ranges from 0 to 1 bounds
bnds = tuple((0,1) for x in range(numofasset))
bnds

# Specify constraints
cons = ({'type': 'eq', 'fun': lambda x: sum(x)-1})
cons

# Optimizing for maximum sharpe ratio
opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
opt_sharpe


#Sequential Least Squares Programming optimizer (SLSQP)
#SLSQP minimizes a function of several variables with any combination of 
#bounds, equality and inequality constraints. 
#The method wraps the SLSQP Optimization subroutine 
#originally implemented by Dieter Kraft.

# Portfolio weights (Max Sharpe)
list(zip(symbols, opt_sharpe['x']))


# Portfolio stats (Max Sharpe)
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats, around(portfolio_stats(opt_sharpe['x']),4)))

# Minimize the variance
def min_variance(weights):
    return portfolio_stats(weights)[1]**2


# Optimizing for minimum variance
opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
opt_var



# Portfolio weights (Min Variance)
list(zip(symbols, around(opt_var['x']*100,2)))



# Portfolio stats (Min Variance)
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats, around(portfolio_stats(opt_var['x']),4)))


#Efficient frontier
def min_volatility(weights):
    return portfolio_stats(weights)[1]

# Efficient frontier params
targetrets = linspace(0.30,0.60,100)
tvols = []

for tr in targetrets:
    
    ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
               {'type': 'eq', 'fun': lambda x: sum(x) - 1})
    
    opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)
    
    tvols.append(opt_ef['fun'])

targetvols = array(tvols)


# Dataframe for EF
efport = pd.DataFrame({
    'targetrets' : around(100*targetrets[14:],2),
    'targetvols': around(100*targetvols[14:],2),
    'targetsharpe': around(targetrets[14:]/targetvols[14:],2)
})

efport.head(5)

# Plot efficient frontier portfolio
fig = px.scatter(
    efport, x='targetvols', y='targetrets',  color='targetsharpe',
    labels={'targetrets': 'Expected Return', 'targetvols': 'Expected Volatility','targetsharpe': 'Sharpe Ratio'},
    title="Efficient Frontier Portfolio"
     ).update_traces(mode='markers', marker=dict(symbol='cross'))


# Plot maximum sharpe portfolio
fig.add_scatter(
    mode='markers',
    x=[100*portfolio_stats(opt_sharpe['x'])[1]], 
    y=[100*portfolio_stats(opt_sharpe['x'])[0]],
    marker=dict(color='red', size=20, symbol='star'),
    name = 'Max Sharpe'
).update(layout_showlegend=False)

# Plot minimum variance portfolio
fig.add_scatter(
    mode='markers',
    x=[100*portfolio_stats(opt_var['x'])[1]], 
    y=[100*portfolio_stats(opt_var['x'])[0]],
    marker=dict(color='green', size=20, symbol='star'),
    name = 'Min Variance'
).update(layout_showlegend=False)

# Show spikes
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()