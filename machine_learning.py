class BacktesterML:

    def __init__(self, df, transaction_cost=0.003, slippage=0.001):
        self.df = df.copy()
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.features = []
        self.target = 'target'
        self.model = None

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

    #This functionc creates a target column for ML model training
    def _add_target(self, z_window, revert_window, revert_std):

        z_col = f'z_{z_window}'
        self.df['future_z'] = self.df[z_col].rolling(revert_window).apply(
            lambda x: (x.iloc[-1] - x.min()) if x.iloc[-1] > 0 else (x.max() - x.iloc[-1]), raw=False
        ).shift(-revert_window)

        self.df[self.target] = 0
        self.df.loc[self.df['future_z'] > revert_std, self.target] = 1

    #Set up the features for the model
    def set_features(self, features):
        self.features = features
    
    def train_model(self, train_df, z_window, revert_window, revert_std):
        self.df = train_df.copy()
        self.zscores(window=z_window)
        self._add_target(z_window, revert_window, revert_std)
        self.df = self.df.dropna()

        X = self.df[self.features]
        y = self.df[self.target]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        self.model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=100)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10)])

    def run_strategy_ml(self, z_window, entry_z, probability_threshold, exit_z):
        df = self.df.copy()
        df = self.zscores(window=z_window)
        z_col = f'z_{z_window}'
        df[z_col] = df[z_col].fillna(0)

        
        df['signal_prob'] = self.model.predict_proba(df[self.features])[:, 1]
        
        banknifty_pos = 0
        realized_pnl = 0
        pnl_curve = []
        trade_pnls = []
        entry_spread, entry_tte = 0, 0
        
        for i, row in df.iterrows():
            z_score = row[z_col]
            signal_prob = row['signal_prob']
            
            if banknifty_pos == 0 and abs(z_score) > entry_z and signal_prob > probability_threshold:
                if z_score > 0:
                    banknifty_pos = -1
                else:
                    banknifty_pos = 1
                
                entry_spread, entry_tte = row['spread'], row['tte']
                realized_pnl -= 2 * (self.transaction_cost + self.slippage)
            
            elif banknifty_pos != 0 and abs(z_score) < exit_z:
                exit_value = (row['spread'] * (row['tte'] ** 0.7))
                entry_value = (entry_spread * (entry_tte ** 0.7))
                trade_pnl = exit_value - entry_value
                if banknifty_pos == -1:
                    realized_pnl -= trade_pnl
                else:
                    realized_pnl += trade_pnl
                trade_pnls.append(trade_pnl)
                banknifty_pos = 0
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

    def grid_search_ml(self, test_df, z_window, entry_z_range, probability_range, exit_z_range):

        results = []
        for entry_z in entry_z_range:
            for exit_z in exit_z_range:
                for prob_threshold in probability_range:
                    
                    self.df = test_df.copy()
                    pnl_series, cum_pnl, num_trades, trade_pnls = self.run_strategy_ml(z_window, entry_z, prob_threshold, exit_z)
                    metrics = self.evaluate(pnl_series, trade_pnls)
                    results.append({
                        "entry_z": entry_z,
                        "probability_threshold": prob_threshold,
                        "exit_z": exit_z,
                        **metrics
                    })
        return pd.DataFrame(results)

    def full_grid_search_ml(self, train_df, test_df, z_window_range, revert_window_range, revert_std_range, entry_z_range, probability_range, exit_z_range):

        all_results = []
        for z_window in z_window_range:
            for revert_window in revert_window_range:
                for revert_std in revert_std_range:
                    #print(f"--- Training model with z_window={z_window}, revert_window={revert_window}, revert_std={revert_std} ---")

                    self.train_model(train_df, z_window, revert_window, revert_std)

                    trading_results = self.grid_search_ml(
                        test_df=test_df,
                        z_window=z_window,
                        entry_z_range=entry_z_range,
                        probability_range=probability_range,
                        exit_z_range=exit_z_range
                    )

                    trading_results['z_window'] = z_window
                    trading_results['revert_window'] = revert_window
                    trading_results['revert_std'] = revert_std
                    all_results.append(trading_results)
                    
        return pd.concat(all_results, ignore_index=True)

features_list = ['banknifty', 'nifty', 'tte', 'spread', 'first_diff', 'minute_day',
       'mean_30', 'std_30', 'z_30', 'mean_200', 'std_200', 'z_200', 'mean_400',
       'std_400', 'z_400', 'mean_800', 'std_800', 'z_800', 'mean_1200',
       'std_1200', 'z_1200', 'mean_1875', 'std_1875', 'z_1875', '15_momentum',
       '30_momentum', '180_momentum', '1d_momentum', '2d_momentum',
       '5d_momentum', 'minute', 'hour', 'minute_of_day']

ml_backtester = BacktesterML(df=train_df)
ml_backtester.set_features(features_list)

z_window_range = [30, 60, 200, 600, 850, 1000, 1400, 1875]
revert_window_range = [20, 50, 100]
revert_std_range = [0.5, 1.0, 1.5]
entry_z_range = [1, 1.5, 2.0, 2.5]
probability_range = [0.5, 0.6, 0.7]
exit_z_range = [0.5, 1.0, 1.5]

results_df = ml_backtester.full_grid_search_ml(
    train_df,
    test_df,
    z_window_range,
    revert_window_range,
    revert_std_range,
    entry_z_range,
    probability_range,
    exit_z_range
)