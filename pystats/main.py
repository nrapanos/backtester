import matplotlib.pyplot as plt
# mpl.use("pgf")
from matplotlib.backends.backend_pdf import PdfPages
from utilities.plotting import *

from backtester.pystats.functions import *

A4 = (8.27, 11.69)


class PerformanceStats(object):
    def __init__(
            self,
            levels,
            positions=None,
            market_data=None,
    ):

        self._levels = levels
        self._positions = positions
        self._market_data = market_data

    def levels(self):
        return self._levels

    def positions(self):
        return self._positions

    def market_data(self):
        return self._market_data

    def stats(self, compounding='diff', ann=252, mul=1):
        table = perf(levels=self.levels(), compounding=compounding, ann=ann, mul=mul)
        return table

    def returns(self, compounding='diff'):
        r = get_returns(self.levels(), compounding=compounding)
        return r

    def pnl(self):
        pnl = self.levels().diff()
        return pnl

    def drawdown(self, compounding='diff'):
        dd = drawdown(self.levels(), compounding=compounding)
        return dd

    def rolling_volatility(self, compounding='diff', window=252, ann=252):
        rv = rolling_vol(self.returns(compounding), window=window, ann=ann)
        return rv

    def rolling_sharpe(self, compounding='diff', window=252, ann=252):
        sharpe = rolling_sharpe(self.returns(compounding), window=window, ann=ann)
        return sharpe

    def turnover(self):
        to = turnover(self.positions(), self.market_data())
        return to

    def exposures(self):
        holdings = self.positions()
        prices = self.market_data()
        exposures = holdings * prices
        exposures.dropna(how='all', inplace=True)
        exposures.dropna(axis=1, how='all', inplace=True)
        return exposures

    def gross_weights(self):
        '''
            Return asset weights relative to the gross portfolio exposure.
        '''
        exposures = self.exposures()
        weight_gross = abs(exposures).div(abs(exposures).sum(axis=1), axis=0)
        return weight_gross

    def net_weights(self):
        '''
            Returns signed asset weights relative to the gross portfolio exposure.
        '''
        exposures = self.exposures()
        signs = np.sign(exposures)
        gross_weights = self.gross_weights()
        net_weights = signs * gross_weights
        return net_weights

    def weights(self):
        '''
            Returns signed weights relative to the net portfolio exposure.
        '''
        weights = self.exposures().div(self.exposures().sum(axis=1), axis=0)
        return weights

    def report(self, folder=r'C:\Users\nrapanos\Desktop\caxgit\backtester\research\Reports', name=None):

        if name is None:
            if self.name is not None:
                name = self.name
            else:
                name = f'Strategy'
        filepath = fr'{folder}\{name}.pdf'

        pdf = PdfPages(filepath)

        sr = sharpe(self.returns(compounding='diff'))

        plt.ioff()
        fig1, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True, figsize=A4)
        legend = []
        legend.append(f'{name} (SR: {sr.round(2)})')
        ax1 = plot_series(self.levels(), ax=ax1, ylabel='Cumulative PnL', title=f'{name}: Performance')

        if hasattr(self, "levels_gross"):
            sr_gross = sharpe(get_returns(self.levels_gross(), compounding='diff'))
            legend.append(f'{name} (Gross, SR: {sr_gross.round(2)})')
            ax1 = plot_series(self.levels_gross(), ax=ax1, color='gray')

        ax1.legend(legend, loc='upper left')

        plot_series(self.pnl(), ax=ax2, ylabel='Daily PnL')
        plot_series(self.rolling_volatility(), ax=ax3, ylabel='Rolling Volatility')
        ax3.set_ylim([0, 2 * self.rolling_volatility().max()])
        plot_series(self.drawdown(), ax=ax4, ylabel='Drawdown')
        fig1.tight_layout()
        pdf.savefig(fig1, papertype='a4')

        if self._positions is not None and self._market_data is not None:
            fig2, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, figsize=A4)
            plot_series(self.positions(), ax=ax1, ylabel='Holdings', legend=True, title=f'{name}: Holdings')
            plot_series(self.turnover(), ax=ax2, ylabel='Turnover', decimals=True)
            plot_stacked(self.weights(), ax=ax3, loc='upper left', ylabel='Portfolio Weights')
            fig2.tight_layout()
            # export to pdf
            pdf.savefig(fig2, papertype='a4')

        pdf.close()
        plt.ion()
        return
