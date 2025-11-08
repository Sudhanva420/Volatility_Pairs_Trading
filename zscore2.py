class Backtester:
    def __init__(self, df, transaction_cost=0.003, slippage=0.001):
        self.df = df.copy()
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    # Function to calculate z-scores
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

    def run_strategy(self, window, entry_z, exit_z, trend_window):

        df = self.zscores(window).dropna()
        
        #Making use of a long-term moving average to act as a trend filter.
        df['trend_ma'] = df['first_diff'].rolling(trend_window).mean().shift(1)
        
        # A rolling standard deviation of the z-scores to make entry/exit thresholds dynamic and adaptive
        df['rolling_z_std'] = df[f'z_{window}'].rolling(window).std().shift(1)
        
        df = df.dropna()

        banknifty_pos, nifty_pos = 0, 0
        entry_spread, entry_tte = 0, 0
        realized_pnl = 0
        pnl_curve = []
        trade_pnls = []

        for i, row in df.iterrows():
            z_score = row[f'z_{window}']
            
            #The dynamic thresholds
            dynamic_entry_z = entry_z * row['rolling_z_std']
            dynamic_exit_z = exit_z * row['rolling_z_std']

            # Checking if the current spread is far from its long-term average
            is_trending = abs(row['first_diff'] - row['trend_ma']) > 0.05

            # We only enter a short spread position if it's not trending
            if banknifty_pos == 0 and z_score > dynamic_entry_z and not is_trending:
                banknifty_pos, nifty_pos = -1, 1
                entry_spread, entry_tte = row['spread'], row['tte']
                realized_pnl -= 2 * (self.transaction_cost + self.slippage)

            # We only enter a long spread position only if it's not trending
            elif banknifty_pos == 0 and z_score < -dynamic_entry_z and not is_trending:
                banknifty_pos, nifty_pos = 1, -1
                entry_spread, entry_tte = row['spread'], row['tte']
                realized_pnl -= 2 * (self.transaction_cost + self.slippage)

            elif banknifty_pos != 0 and abs(z_score) < dynamic_exit_z:
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
    
    def grid_search(self, window_range, entry_z_range, exit_z_range, trend_window_range):

        results = []

        for window in window_range:
            for entry_z in entry_z_range:
                for exit_z in exit_z_range:
                    for trend_window in trend_window_range:
                        pnl_series, cum_pnl, num_trades, trade_pnls = self.run_strategy(window, entry_z, exit_z, trend_window)
                        metrics = self.evaluate(pnl_series, trade_pnls)
                        results.append({
                            "window": window,
                            "entry_z": entry_z,
                            "exit_z": exit_z,
                            "trend_window": trend_window,
                            **metrics
                        })

        results_df = pd.DataFrame(results)
        return results_df

bt = Backtester(train_df)
results = bt.grid_search(window_range=[30, 60, 100, 200, 320, 600, 850, 1000, 1200, 1400, 1875], entry_z_range=[1, 1.5, 2, 2.5, 3], exit_z_range=[0.5, 0.7, 1, 1.3, 1.6, 2], trend_window_range=[200, 300, 400, 650, 800, 920, 1200])
results.sort_values("sharpe_per_bar", ascending=False).head()