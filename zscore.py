class Backtester:
    def __init__(self, df, transaction_cost=0.003, slippage=0.001):
        self.df = df.copy()
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    #Function to calculate zscores
    def zscores(self, window):
        
        col_names = {
        "mean": f"mean_{window}",
        "std": f"std_{window}",
        "z": f"z_{window}"
        }

        self.df[col_names["mean"]] = self.df['first_diff'].rolling(window).mean().shift(1)
        self.df[col_names["std"]] = self.df['first_diff'].rolling(window).std().shift(1)
        self.df[col_names["z"]] = (
            (self.df['first_diff'] - self.df[col_names["mean"]]) / self.df[col_names["std"]]
        )
        
        return self.df

    # A run strategy function that directly makes use of the spread to backtest instead of maintaining separate positions
    def run_strategy(self, window, entry_z, exit_z):

        df = self.zscores(window).dropna()

        banknifty_pos, nifty_pos = 0, 0
        entry_spread, entry_tte = 0, 0
        realized_pnl = 0
        pnl_curve = []
        trade_pnls = []

        for i, row in df.iterrows():
            z_score = row[f'z_{window}']
            
            # This enters a short spread position
            if banknifty_pos == 0 and z_score > entry_z:
                banknifty_pos, nifty_pos = -1, 1
                entry_spread, entry_tte = row['spread'], row['tte']
                realized_pnl -= 2 * (self.transaction_cost + self.slippage)

            # This enters a short spread position
            elif banknifty_pos == 0 and z_score < -entry_z:
                banknifty_pos, nifty_pos = 1, -1
                entry_spread, entry_tte = row['spread'], row['tte']
                realized_pnl -= 2 * (self.transaction_cost + self.slippage)

            # Exit a position and calculate profit based on specified formula
            elif banknifty_pos != 0 and abs(z_score) < exit_z:
                exit_value = (row['spread'] * (row['tte'] ** 0.7))
                entry_value = (entry_spread * (entry_tte ** 0.7))
                trade_pnl = exit_value - entry_value

                if banknifty_pos == 1: 
                    realized_pnl += trade_pnl
                else:                 
                    realized_pnl -= trade_pnl

                trade_pnls.append(trade_pnl)
                banknifty_pos = nifty_pos = 0
                entry_spread, entry_tte = 0, 0
            
            # Unrealised pnl is accurate analysis of the equity curve
            unrealized_pnl = 0
            if banknifty_pos != 0:
                current_value = (row['spread'] * (row['tte'] ** 0.7))
                entry_value = (entry_spread * (entry_tte ** 0.7))
                unrealized_pnl = current_value - entry_value
                if banknifty_pos == -1: 
                    unrealized_pnl = -unrealized_pnl

            equity_value = realized_pnl + unrealized_pnl
            pnl_curve.append(equity_value)

        return pd.Series(pnl_curve, index=df.index), realized_pnl, len(trade_pnls), trade_pnls

    def evaluate(self, pnl_series, trade_pnls=None):
        
        #resample to get a better/more comprehensive way to calculate sharpe 
        daily_returns = pnl_series.resample("1D").last().diff().dropna()
        sharpe_daily = 0
        if daily_returns.std() != 0:
            sharpe_daily = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        return {
            "sharpe_per_bar": sharpe_daily,
            "final_pnl": pnl_series.iloc[-1],
            "max_drawdown": self.max_drawdown(pnl_series),
            "num_trades": len(trade_pnls) if trade_pnls else 0
        }

    @staticmethod
    def max_drawdown(pnl_series):
        cummax = pnl_series.cummax()
        drawdown = pnl_series - cummax
        return -drawdown.min()
    

backtester = Backtester(df_trading)
pnl_curve, cum_pnl, trade_count, trade_pnls = backtester.run_strategy(window=60, entry_z=2, exit_z=0.5)
metrics = backtester.evaluate(pnl_curve, trade_pnls)
print(metrics)

class Backtester:
    def __init__(self, df, transaction_cost=0.003, slippage=0.001):
        self.df = df.copy()
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    #Function to calculate zscores
    def zscores(self, window):
        
        col_names = {
        "mean": f"mean_{window}",
        "std": f"std_{window}",
        "z": f"z_{window}"
        }

        self.df[col_names["mean"]] = self.df['first_diff'].rolling(window).mean().shift(1)
        self.df[col_names["std"]] = self.df['first_diff'].rolling(window).std().shift(1)
        self.df[col_names["z"]] = (
            (self.df['first_diff'] - self.df[col_names["mean"]]) / self.df[col_names["std"]]
        )
        
        return self.df

    def run_strategy(self, window, entry_z, exit_z):

        df = self.zscores(window).dropna()

        banknifty_pos, nifty_pos = 0, 0
        entry_spread, entry_tte = 0, 0
        realized_pnl = 0
        pnl_curve = []
        trade_pnls = []

        for i, row in df.iterrows():
            z_score = row[f'z_{window}']

            if banknifty_pos == 0 and z_score > entry_z:
                banknifty_pos, nifty_pos = -1, 1
                entry_spread, entry_tte = row['spread'], row['tte']
                realized_pnl -= 2 * (self.transaction_cost + self.slippage)

            elif banknifty_pos == 0 and z_score < -entry_z:
                banknifty_pos, nifty_pos = 1, -1
                entry_spread, entry_tte = row['spread'], row['tte']
                realized_pnl -= 2 * (self.transaction_cost + self.slippage)

            elif banknifty_pos != 0 and abs(z_score) < exit_z:
                exit_value = (row['spread'] * (row['tte'] ** 0.7))
                entry_value = (entry_spread * (entry_tte ** 0.7))
                trade_pnl = exit_value - entry_value

                if banknifty_pos == 1: 
                    realized_pnl += trade_pnl
                else:                 
                    realized_pnl -= trade_pnl

                trade_pnls.append(trade_pnl)
                banknifty_pos = nifty_pos = 0
                entry_spread, entry_tte = 0, 0

            unrealized_pnl = 0
            if banknifty_pos != 0:
                current_value = (row['spread'] * (row['tte'] ** 0.7))
                entry_value = (entry_spread * (entry_tte ** 0.7))
                unrealized_pnl = current_value - entry_value
                if banknifty_pos == -1: 
                    unrealized_pnl = -unrealized_pnl

            equity_value = realized_pnl + unrealized_pnl
            pnl_curve.append(equity_value)

        return pd.Series(pnl_curve, index=df.index), realized_pnl, len(trade_pnls), trade_pnls

    def evaluate(self, pnl_series, trade_pnls=None):
        daily_returns = pnl_series.resample("1D").last().diff().dropna()
        sharpe_daily = 0
        if daily_returns.std() != 0:
            sharpe_daily = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        return {
            "sharpe_per_bar": sharpe_daily,
            "final_pnl": pnl_series.iloc[-1],
            "max_drawdown": self.max_drawdown(pnl_series),
            "num_trades": len(trade_pnls) if trade_pnls else 0
        }

    @staticmethod
    def max_drawdown(pnl_series):
        cummax = pnl_series.cummax()
        drawdown = pnl_series - cummax
        return -drawdown.min()
    
    #This function is used for grid-searching different paramenter(window, entry and exit)
    def grid_search(self, window_range, entry_z_range, exit_z_range):
        
        results = []

        for window in window_range:
            for entry_z in entry_z_range:
                for exit_z in exit_z_range:
                    pnl_series, cum_pnl, num_trades, trade_pnls = self.run_strategy(window, entry_z, exit_z)
                    metrics = self.evaluate(pnl_series, trade_pnls)
                    results.append({
                            "window": window,
                            "entry_z": entry_z,
                            "exit_z": exit_z,
                            **metrics
                    })

        results_df = pd.DataFrame(results)
        return results_df

cutoff_date = "2022-01-01" 
train_df = df_trading.loc[:cutoff_date]
test_df = df_trading.loc[cutoff_date:]

bt = Backtester(train_df)
results = bt.grid_search(window_range=[30, 60, 200, 600, 850, 1000, 1400, 1875], entry_z_range=[1, 1.5, 2, 2.5], exit_z_range=[0.5, 0.7, 1, 1.5])
results.sort_values("sharpe_per_bar", ascending=False).head()