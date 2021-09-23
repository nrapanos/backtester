from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

geometric_aliases = ['pct', 'percent', 'percentage', 'geometric']
arithmetic_aliases = ['diff', 'chg', 'arithmetic']


def perf(levels=None, returns=None, ann=252, rf=0, compounding='pct', mul=1):
    if levels is not None and returns is None:
        returns = get_returns(levels, compounding=compounding)

    if returns is not None and levels is None:
        levels = get_levels(returns, compounding=compounding)

    if type(returns) == type(pd.Series()):
        returns = returns.to_frame()

    if type(levels) == type(pd.Series()):
        levels = levels.to_frame()

    res = OrderedDict()
    res['mean'] = returns.mean() * ann * mul
    res['vol'] = returns.std() * np.sqrt(ann) * mul
    res['sharpe'] = sharpe(returns, rf=rf, ann=ann)
    res['maxDD'] = max_drawdown(levels) * mul
    res['skew'] = returns.skew()
    res['kurt'] = returns.kurt()

    res = pd.DataFrame(res).T

    return res


def samples_per_day(df, stat='median'):
    df = df.copy()
    if type(df) == type(pd.Series()):
        df = df.to_frame()
    df['date'] = df.index.date
    n_samples = df.groupby('date').count().apply(stat)
    if len(n_samples) == 1:
        n_samples = n_samples.iloc[0]
    return n_samples


def samples_per_year(df, stat='median'):
    df = df.copy()
    if type(df) == type(pd.Series()):
        df = df.to_frame()
    df['year'] = df.index.year
    n_samples = df.groupby('year').count().apply(stat)
    if len(n_samples) == 1:
        n_samples = n_samples.iloc[0]
    return n_samples


def get_levels(returns, compounding='pct'):
    sDate = returns.index[0] - BDay(1)
    if compounding in geometric_aliases:
        levels = (returns + 1).cumprod()
        levels.loc[sDate] = 1
    elif compounding in arithmetic_aliases:
        levels = returns.cumsum()
        levels.loc[sDate] = 0
    else:
        raise ValueError(f'Returns type {compounding} is not supported.')
    levels = levels.sort_index()
    return levels


def get_returns(levels, compounding='pct'):
    if compounding in geometric_aliases:
        rets = levels.pct_change(fill_method=None).dropna()
    elif compounding in arithmetic_aliases:
        rets = levels.diff().dropna()
    else:
        raise ValueError(f'Returns type {compounding} is not supported.')

    return rets


# TODO: Allow for arithmetic returns.
def get_daily_returns(returns, compounding='pct'):
    df = returns.to_frame(name='pct_return')

    # make sure you don't lose first observation
    new_index = [df.index[0] - BDay(1)] + list(df.index)
    df = df.reindex(new_index)
    df.iloc[0, :] = 0

    # create date column
    df['date'] = df.index.date

    # convert to prices and then back to returns
    daily_ret = df.groupby('date')['pct_return'].apply(lambda x: (x + 1).cumprod() - 1)
    return daily_ret


def sharpe(returns, rf=0.0, ann=252):
    er = returns - rf
    sharpe = er.mean() / er.std()
    sharpe = sharpe * np.sqrt(ann)
    if type(sharpe) == type(pd.DataFrame):
        sharpe.replace(np.inf, np.nan)
    return sharpe


def rolling_sharpe(returns, window, ann=252):
    mu = rolling_return(returns, window, ann)
    sigma = rolling_vol(returns, window, ann)
    sharpe = mu / sigma
    return sharpe


def rolling_return(returns, window, ann=252):
    mu = returns.rolling(window).mean()
    mu = mu * ann
    return mu


def rolling_vol(returns, window=90, ann=252):
    vol = returns.rolling(window).std()
    vol = vol * np.sqrt(ann)
    return vol


def max_drawdown(levels):
    dd = drawdown(levels)
    maxdd = dd.min()
    return maxdd


def seasonality(returns, agg='mean', mul=100, ann=256):
    returns = returns.squeeze()
    name = returns.name
    returns = returns.to_frame()

    returns['year'] = returns.index.year
    returns['month'] = returns.index.month
    returns['date'] = returns.index.date
    returns['day'] = returns.index.dayofweek

    table = returns.groupby(['month', 'day'])[[name]].agg(agg)
    table = table.reset_index().pivot_table(index='month', columns='day', values=name)
    table = (table * mul * ann)

    return table


def drawdown(series):
    '''
        Calculates drawdown given a time-series of price levels.

        When the price is at all time highs, the drawdown is 0.
        When prices are below high water marks (HWM), the drawdown is calculated as

             DD_t = S_t - HWM_t

        Missing values are ignored.
    :param series: pd.Series
        Series of price levels.
    :return: pd.Series
        Drawdown series.
    '''
    # make a copy so that we don't modify original data
    drawdown = series

    # Fill NaN's with previous values
    drawdown = drawdown.fillna(method='ffill')

    # Ignore problems with NaN's in the beginning
    drawdown[np.isnan(drawdown)] = -np.Inf

    # Rolling maximum
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown - roll_max
    return drawdown


def _calc_exposures(holdings, prices):
    exposures = holdings * prices
    exposures.dropna(how='all', inplace=True)
    exposures.dropna(axis=1, how='all', inplace=True)
    return exposures


def _turnover_dollars(holdings, prices, by_asset=False):
    turnover = np.abs(holdings.diff()) * prices
    turnover = turnover.dropna(axis=1, how='all')
    if by_asset == False:
        turnover = turnover.sum(axis=1)
        turnover.dropna(inplace=True)
    return turnover


def turnover(holdings, prices):
    exposures = _calc_exposures(holdings, prices)
    val = np.abs(exposures).sum(axis=1).shift()
    churn = _turnover_dollars(holdings, prices)
    turnover = churn / val

    # treat missing values
    turnover = turnover.replace(np.inf, np.nan)
    turnover = turnover.replace(0, np.nan)
    turnover = turnover.dropna()
    turnover.name = 'turnover'

    return turnover
