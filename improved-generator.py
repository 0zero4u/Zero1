import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from ..processor import create_bars_from_trades
from .config import SETTINGS
from .engine import HierarchicalTradingEnvironment

class AdvancedBacktester:
    """IMPROVED: Enhanced backtesting system with true ensembling and vectorized calculations."""

    def __init__(self, model_path: str = None, save_results: bool = True):
        self.cfg = SETTINGS
        self.model_path = model_path or self.cfg.get_model_path()
        self.save_results = save_results
        self.results = {}
        self.trade_log = []
        self.portfolio_history = []
        self.attention_history = []

    def run_backtest(self, period: str = "out_of_sample",
                    save_detailed_logs: bool = True) -> Dict:
        """Run comprehensive backtest with detailed analysis."""
        print(f"🔍 Starting Enhanced Backtest for {period.upper()}")
        print("="*60)

        # 1. Prepare Environment
        bars_df = create_bars_from_trades(period)
        env = HierarchicalTradingEnvironment(bars_df)

        # 2. Load Model
        # SECURITY WARNING: Loading model files with pickle can execute arbitrary code.
        # Only load models from sources you trust completely.
        print("\n" + "!"*60)
        print("!!! SECURITY WARNING !!!")
        print("Loading a model file will execute its code. Ensure the file")
        print(f"`{self.model_path}` is from a trusted source.")
        print("!"*60 + "\n")
        
        model = PPO.load(self.model_path, env=env)
        print(f"✅ Loaded model from: {self.model_path}")

        # 3. Initialize Tracking
        obs, info = env.reset()
        done = False
        step_count = 0

        # Initialize metrics tracking
        portfolio_values = [info['balance']]
        balances = [info['balance']]
        assets_held = [info['asset_held']]
        actions_taken = []
        rewards = []
        prices = []
        timestamps = []
        initial_value = portfolio_values[0]

        print(f"🚀 Starting simulation with initial portfolio: ${initial_value:,.2f}")
        print("-" * 60)

        # 4. Run Simulation
        while not done:
            # Get model prediction
            action, _states = model.predict(obs, deterministic=True)
            actions_taken.append(action.copy())

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_count += 1
            rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
            balances.append(info['balance'])
            assets_held.append(info['asset_held'])

            # Get current price and timestamp
            current_price = env.timeframes[env.cfg.base_bar_timeframe]['close'].iloc[env.current_step]
            current_timestamp = env.base_timestamps[env.current_step]

            prices.append(current_price)
            timestamps.append(current_timestamp)

            # Log detailed trade information
            if save_detailed_logs:
                self._log_trade_details(step_count, action, reward, info, current_price, current_timestamp)

            # Extract attention weights and regime information
            self._extract_model_insights(model, step_count)

            # Progress reporting
            if step_count % 1000 == 0:
                current_return = (info['portfolio_value'] - initial_value) / initial_value * 100
                print(f"Step {step_count:,}: Portfolio ${info['portfolio_value']:,.2f} "
                      f"({current_return:+.2f}%) | Balance: ${info['balance']:,.2f} | "
                      f"Assets: {info['asset_held']:.6f}")

            # Show model reasoning periodically
            if step_count % 5000 == 0:
                self._display_model_reasoning(model, action, step_count)

        # 5. Calculate Comprehensive Metrics
        print("\n" + "="*60)
        print("📊 CALCULATING PERFORMANCE METRICS")
        print("="*60)

        metrics = self._calculate_comprehensive_metrics_vectorized(
            portfolio_values, prices, timestamps, actions_taken, rewards
        )

        # 6. Store Results
        self.results = {
            'metrics': metrics,
            'portfolio_values': portfolio_values,
            'balances': balances,
            'assets_held': assets_held,
            'prices': prices,
            'timestamps': timestamps,
            'actions': actions_taken,
            'rewards': rewards,
            'trade_log': self.trade_log,
            'attention_history': self.attention_history,
        }

        # 7. Display Results
        self._display_results(metrics)

        # 8. Generate Visualizations
        if self.save_results:
            self._generate_visualizations()
            self._save_results_to_files()

        print("\n✅ Enhanced backtest completed!")
        return self.results

    def _log_trade_details(self, step: int, action: np.ndarray, reward: float,
                          info: Dict, price: float, timestamp: pd.Timestamp):
        """Log detailed trade information."""
        trade_entry = {
            'step': step,
            'timestamp': timestamp,
            'price': price,
            'action_signal': float(action[0]),
            'action_size': float(action[1]),
            'reward': reward,
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'asset_held': info['asset_held']
        }

        self.trade_log.append(trade_entry)

    def _extract_model_insights(self, model: PPO, step: int):
        """Extract attention weights and regime probabilities for analysis."""
        try:
            # Extract attention weights
            if hasattr(model.policy.features_extractor, 'last_attention_weights'):
                attention_weights = model.policy.features_extractor.last_attention_weights
                if attention_weights is not None:
                    self.attention_history.append({
                        'step': step,
                        'flow_weight': float(attention_weights[0, 0]),
                        'volatility_weight': float(attention_weights[0, 1]),
                        'value_trend_weight': float(attention_weights[0, 2]),
                        'context_weight': float(attention_weights[0, 3])
                    })

        except Exception as e:
            # Silently continue if extraction fails
            pass

    def _display_model_reasoning(self, model: PPO, action: np.ndarray, step: int):
        """Display detailed model reasoning."""
        try:
            if hasattr(model.policy.features_extractor, 'last_attention_weights'):
                weights = model.policy.features_extractor.last_attention_weights
                if weights is not None:
                    reasoning = {
                        "Step": step,
                        "Model_Decision": {
                            "Position_Signal": f"{float(action[0]):.3f}",
                            "Position_Size": f"{float(action[1]):.3f}",
                            "Expert_Attention": {
                                "Flow_Factors": f"{weights[0, 0]:.1%}",
                                "Volatility_Factors": f"{weights[0, 1]:.1%}",
                                "Value_Trend_Factors": f"{weights[0, 2]:.1%}",
                                "Context_Factors": f"{weights[0, 3]:.1%}"
                            }
                        }
                    }

                    print("\n" + "="*25 + " MODEL REASONING " + "="*25)
                    print(json.dumps(reasoning, indent=2))
                    print("="*67)
        except Exception:
            pass

    def _calculate_comprehensive_metrics_vectorized(self, portfolio_values: List[float],
                                                  prices: List[float], timestamps: List,
                                                  actions: List, rewards: List[float]) -> Dict:
        """IMPROVED: Vectorized performance metrics calculation for better performance."""
        
        # Convert to numpy arrays for vectorized operations
        portfolio_array = np.array(portfolio_values[1:])
        price_array = np.array(prices)
        
        # Create pandas Series with timestamps
        portfolio_series = pd.Series(portfolio_array, index=timestamps)
        price_series = pd.Series(price_array, index=timestamps)
        
        # Calculate returns vectorized
        returns = portfolio_series.pct_change().dropna()
        price_returns = price_series.pct_change().dropna()

        # Basic Performance Metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value

        # Risk Metrics - vectorized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # IMPROVED: Vectorized drawdown analysis
        cumulative_max = np.maximum.accumulate(portfolio_array)
        drawdowns_array = (cumulative_max - portfolio_array) / cumulative_max
        max_drawdown = np.max(drawdowns_array)

        # Calculate drawdown duration using vectorized operations
        drawdown_periods = self._calculate_drawdown_periods_vectorized(drawdowns_array)
        avg_drawdown_duration = np.mean(drawdown_periods) if len(drawdown_periods) > 0 else 0
        max_drawdown_duration = np.max(drawdown_periods) if len(drawdown_periods) > 0 else 0

        # Risk-Adjusted Returns
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
        calmar_ratio = (total_return * 252 / len(returns)) / abs(max_drawdown) if max_drawdown != 0 else 0

        # Benchmark Comparison (Buy and Hold)
        benchmark_return = (prices[-1] - prices[0]) / prices[0]
        excess_return_vs_benchmark = total_return - benchmark_return

        # Trading Activity Analysis - vectorized
        actions_array = np.array(actions)
        position_signals = actions_array[:, 0] if len(actions_array) > 0 else []
        position_sizes = actions_array[:, 1] if len(actions_array) > 0 else []

        # Count significant trades (position changes > 10%) - vectorized
        significant_trades = 0
        if len(actions_array) > 1:
            position_changes = np.abs(np.diff(actions_array[:, 0]))
            significant_trades = np.sum(position_changes > 0.1)

        # Win Rate Analysis - vectorized
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Information Ratio
        aligned_returns = returns.reindex(price_returns.index, fill_value=0)
        tracking_error = (aligned_returns - price_returns).std() * np.sqrt(252)
        information_ratio = excess_return_vs_benchmark / tracking_error if tracking_error > 0 else 0

        # VaR and CVaR (95% confidence) - vectorized
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0

        return {
            # Basic Performance
            'total_return': total_return,
            'annualized_return': total_return * 252 / len(returns),
            'final_portfolio_value': final_value,
            'initial_portfolio_value': initial_value,
            
            # Risk Metrics
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration_days': avg_drawdown_duration,
            'max_drawdown_duration_days': max_drawdown_duration,
            'var_95': var_95,
            'cvar_95': cvar_95,
            
            # Risk-Adjusted Returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            
            # Benchmark Comparison
            'benchmark_return': benchmark_return,
            'excess_return_vs_benchmark': excess_return_vs_benchmark,
            
            # Trading Activity
            'total_trades': len(actions),
            'significant_trades': significant_trades,
            'avg_position_signal': np.mean(position_signals) if len(position_signals) > 0 else 0,
            'avg_position_size': np.mean(position_sizes) if len(position_sizes) > 0 else 0,
            
            # Win/Loss Analysis
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            
            # Additional Metrics
            'total_days': len(returns),
            'positive_days': len(positive_returns),
            'negative_days': len(negative_returns)
        }

    def _calculate_drawdown_periods_vectorized(self, drawdowns_array: np.ndarray) -> List[int]:
        """IMPROVED: Vectorized drawdown period calculation."""
        # Find where drawdowns start and end
        in_drawdown = drawdowns_array > 0.01  # 1% threshold
        
        # Find transitions
        transitions = np.diff(in_drawdown.astype(int))
        starts = np.where(transitions == 1)[0] + 1  # Start of drawdown
        ends = np.where(transitions == -1)[0] + 1   # End of drawdown
        
        # Handle edge cases
        if len(starts) == 0:
            return []
            
        if len(starts) > len(ends):
            ends = np.append(ends, len(drawdowns_array))
            
        if len(ends) > len(starts):
            starts = np.insert(starts, 0, 0)
            
        # Calculate durations
        durations = ends - starts
        return durations.tolist()

    def _display_results(self, metrics: Dict):
        """Display comprehensive results in a formatted manner."""
        print("\n📈 PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"Initial Portfolio Value: ${metrics['initial_portfolio_value']:,.2f}")
        print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Benchmark Return: {metrics['benchmark_return']:.2%}")
        print(f"Excess Return: {metrics['excess_return_vs_benchmark']:+.2%}")

        print(f"\n⚖️ RISK METRICS")
        print("-" * 40)
        print(f"Volatility (Annual): {metrics['volatility']:.2%}")
        print(f"Downside Volatility: {metrics['downside_volatility']:.2%}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Avg Drawdown Duration: {metrics['avg_drawdown_duration_days']:.1f} days")
        print(f"VaR (95%): {metrics['var_95']:.2%}")
        print(f"CVaR (95%): {metrics['cvar_95']:.2%}")

        print(f"\n🎯 RISK-ADJUSTED RETURNS")
        print("-" * 40)
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"Information Ratio: {metrics['information_ratio']:.3f}")

        print(f"\n🔄 TRADING ACTIVITY")
        print("-" * 40)
        print(f"Total Actions: {metrics['total_trades']:,}")
        print(f"Significant Trades: {metrics['significant_trades']:,}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average Win: {metrics['avg_win']:.3%}")
        print(f"Average Loss: {metrics['avg_loss']:.3%}")

        # Performance Rating
        rating = self._calculate_performance_rating(metrics)
        print(f"\n🏆 OVERALL PERFORMANCE RATING: {rating}")

    def _calculate_performance_rating(self, metrics: Dict) -> str:
        """Calculate overall performance rating."""
        score = 0

        # Return component (25%)
        if metrics['annualized_return'] > 0.2: score += 25
        elif metrics['annualized_return'] > 0.1: score += 20
        elif metrics['annualized_return'] > 0.05: score += 15
        elif metrics['annualized_return'] > 0: score += 10

        # Sharpe ratio component (25%)
        if metrics['sharpe_ratio'] > 2: score += 25
        elif metrics['sharpe_ratio'] > 1.5: score += 20
        elif metrics['sharpe_ratio'] > 1: score += 15
        elif metrics['sharpe_ratio'] > 0.5: score += 10

        # Drawdown component (25%)
        if metrics['max_drawdown'] > -0.05: score += 25
        elif metrics['max_drawdown'] > -0.1: score += 20
        elif metrics['max_drawdown'] > -0.2: score += 15
        elif metrics['max_drawdown'] > -0.3: score += 10

        # Win rate component (25%)
        if metrics['win_rate'] > 0.6: score += 25
        elif metrics['win_rate'] > 0.55: score += 20
        elif metrics['win_rate'] > 0.5: score += 15
        elif metrics['win_rate'] > 0.45: score += 10

        if score >= 90: return "EXCELLENT ⭐⭐⭐⭐⭐"
        elif score >= 75: return "VERY GOOD ⭐⭐⭐⭐"
        elif score >= 60: return "GOOD ⭐⭐⭐"
        elif score >= 40: return "FAIR ⭐⭐"
        else: return "POOR ⭐"

    def _generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n📊 Generating visualizations...")

        # Create results directory
        results_dir = Path(self.cfg.base_path) / "backtest_results"
        results_dir.mkdir(exist_ok=True)

        # 1. Portfolio Performance Chart
        self._create_portfolio_performance_chart(results_dir)

        # 2. Drawdown Chart
        self._create_drawdown_chart(results_dir)

        # 3. Attention Weights Heatmap
        if self.attention_history:
            self._create_attention_heatmap(results_dir)

        # 4. Trade Analysis
        self._create_trade_analysis_chart(results_dir)

        # 5. Risk Metrics Dashboard
        self._create_risk_dashboard(results_dir)

        print(f"✅ Visualizations saved to: {results_dir}")

    def _create_portfolio_performance_chart(self, results_dir: Path):
        """Create portfolio performance comparison chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value vs Benchmark', 'Daily Returns'),
            vertical_spacing=0.08
        )

        timestamps = self.results['timestamps']
        portfolio_values = self.results['portfolio_values'][1:]
        prices = self.results['prices']

        # Portfolio performance
        initial_portfolio = self.results['portfolio_values'][0]
        portfolio_returns = [(pv / initial_portfolio - 1) * 100 for pv in portfolio_values]

        # Benchmark performance
        initial_price = prices[0]
        benchmark_returns = [(p / initial_price - 1) * 100 for p in prices]

        fig.add_trace(
            go.Scatter(x=timestamps, y=portfolio_returns, name="Portfolio", line=dict(color='blue')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=timestamps, y=benchmark_returns, name="Benchmark (Buy & Hold)", line=dict(color='red')),
            row=1, col=1
        )

        # Daily returns
        returns = pd.Series(portfolio_values).pct_change().fillna(0) * 100

        fig.add_trace(
            go.Scatter(x=timestamps, y=returns, name="Daily Returns", line=dict(color='green'), opacity=0.7),
            row=2, col=1
        )

        fig.update_layout(
            title="Portfolio Performance Analysis",
            height=800,
            showlegend=True
        )

        fig.write_html(results_dir / "portfolio_performance.html")

    def _create_drawdown_chart(self, results_dir: Path):
        """Create drawdown analysis chart."""
        portfolio_series = pd.Series(self.results['portfolio_values'][1:], index=self.results['timestamps'])
        cumulative_max = portfolio_series.cummax()
        drawdown = (portfolio_series - cumulative_max) / cumulative_max * 100

        fig = go.Figure()

        # Drawdown area
        fig.add_trace(go.Scatter(
            x=self.results['timestamps'],
            y=drawdown,
            fill='tonexty',
            name='Drawdown %',
            line=dict(color='red'),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(
            title="Portfolio Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=500
        )

        fig.write_html(results_dir / "drawdown_analysis.html")

    def _create_attention_heatmap(self, results_dir: Path):
        """Create attention weights heatmap over time."""
        attention_df = pd.DataFrame(self.attention_history)
        if len(attention_df) > 0:
            # Sample data for visualization (too many points can be overwhelming)
            if len(attention_df) > 1000:
                attention_df = attention_df.iloc[::len(attention_df)//1000]

            weights_data = attention_df[['flow_weight', 'volatility_weight', 'value_trend_weight', 'context_weight']].T

            fig = go.Figure(data=go.Heatmap(
                z=weights_data.values,
                x=attention_df['step'],
                y=['Flow Factors', 'Volatility Factors', 'Value/Trend Factors', 'Context Factors'],
                colorscale='Viridis',
                hoverongaps=False
            ))

            fig.update_layout(
                title="Model Attention Weights Over Time",
                xaxis_title="Training Step",
                yaxis_title="Expert Type",
                height=500
            )

            fig.write_html(results_dir / "attention_heatmap.html")

    def _create_trade_analysis_chart(self, results_dir: Path):
        """Create trade analysis visualization."""
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Action Signals Over Time', 'Position Sizes Over Time',
                              'Reward Distribution', 'Price vs Portfolio Value')
            )

            # Action signals
            fig.add_trace(
                go.Scatter(x=trade_df['step'], y=trade_df['action_signal'],
                          name="Action Signal", line=dict(color='blue')),
                row=1, col=1
            )

            # Position sizes
            fig.add_trace(
                go.Scatter(x=trade_df['step'], y=trade_df['action_size'],
                          name="Position Size", line=dict(color='green')),
                row=1, col=2
            )

            # Reward distribution
            fig.add_trace(
                go.Histogram(x=trade_df['reward'], name="Rewards", nbinsx=50),
                row=2, col=1
            )

            # Price vs Portfolio
            fig.add_trace(
                go.Scatter(x=trade_df['price'], y=trade_df['portfolio_value'],
                          mode='markers', name="Price vs Portfolio",
                          marker=dict(color=trade_df['reward'], colorscale='RdYlGn',
                                    colorbar=dict(title="Reward"))),
                row=2, col=2
            )

            fig.update_layout(
                title="Trading Activity Analysis",
                height=800,
                showlegend=False
            )

            fig.write_html(results_dir / "trade_analysis.html")

    def _create_risk_dashboard(self, results_dir: Path):
        """Create comprehensive risk metrics dashboard."""
        metrics = self.results['metrics']

        # Create gauge charts for key metrics
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=['Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Volatility']
        )

        # Sharpe Ratio
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['sharpe_ratio'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sharpe Ratio"},
            gauge={'axis': {'range': [None, 3]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 1.5}}
        ), row=1, col=1)

        # Max Drawdown
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=abs(metrics['max_drawdown']) * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Max Drawdown %"},
            gauge={'axis': {'range': [0, 50]},
                   'bar': {'color': "red"},
                   'steps': [{'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 20], 'color': "yellow"}],
                   'threshold': {'line': {'color': "black", 'width': 4},
                               'thickness': 0.75, 'value': 15}}
        ), row=1, col=2)

        # Win Rate
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['win_rate'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Win Rate %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green"},
                   'steps': [{'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 60], 'color': "yellow"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 50}}
        ), row=2, col=1)

        # Volatility
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['volatility'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Volatility %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "orange"},
                   'steps': [{'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 40], 'color': "yellow"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 30}}
        ), row=2, col=2)

        fig.update_layout(
            title="Risk Metrics Dashboard",
            height=800
        )

        fig.write_html(results_dir / "risk_dashboard.html")

    def _save_results_to_files(self):
        """Save detailed results to CSV files."""
        results_dir = Path(self.cfg.base_path) / "backtest_results"

        # Save metrics
        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_df.to_csv(results_dir / "performance_metrics.csv", index=False)

        # Save trade log
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            trade_df.to_csv(results_dir / "detailed_trade_log.csv", index=False)

        # Save attention history
        if self.attention_history:
            attention_df = pd.DataFrame(self.attention_history)
            attention_df.to_csv(results_dir / "attention_weights_history.csv", index=False)

        # Save portfolio time series
        portfolio_df = pd.DataFrame({
            'timestamp': self.results['timestamps'],
            'portfolio_value': self.results['portfolio_values'][1:],
            'price': self.results['prices'],
            'balance': self.results['balances'][1:],
            'asset_held': self.results['assets_held'][1:]
        })
        portfolio_df.to_csv(results_dir / "portfolio_timeseries.csv", index=False)

        print(f"📁 Detailed results saved to: {results_dir}")

# --- IMPROVED ENSEMBLE BACKTESTING ---

class EnsembleBacktester:
    """IMPROVED: Backtest ensemble of models with true action averaging."""

    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        self.model_paths = model_paths
        self.weights = weights or [1.0] * len(model_paths)
        self.individual_results = []
        self.models = []

    def run_ensemble_backtest(self) -> Dict:
        """IMPROVED: Run backtest for ensemble of models with true ensembling."""
        print(f"🤖 Running ensemble backtest with {len(self.model_paths)} models")

        # Load all models
        bars_df = create_bars_from_trades("out_of_sample")
        env = HierarchicalTradingEnvironment(bars_df)

        for i, model_path in enumerate(self.model_paths):
            print(f"Loading model {i+1}/{len(self.model_paths)}: {model_path}")
            model = PPO.load(model_path, env=env)
            self.models.append(model)

        # Run ensemble backtest
        print(f"\n--- Running Ensemble Backtest ---")
        ensemble_results = self._run_true_ensemble_backtest(env)

        # Run individual backtests for comparison
        for i, model_path in enumerate(self.model_paths):
            print(f"\n--- Model {i+1}/{len(self.model_paths)} Individual Backtest ---")
            backtester = AdvancedBacktester(model_path, save_results=False)
            results = backtester.run_backtest(save_detailed_logs=False)
            self.individual_results.append(results)

        # Compare all models
        self._create_ensemble_comparison()

        return ensemble_results

    def _run_true_ensemble_backtest(self, env: HierarchicalTradingEnvironment) -> Dict:
        """IMPROVED: True ensemble backtesting with action averaging."""
        
        obs, info = env.reset()
        done = False
        step_count = 0

        # Initialize tracking
        portfolio_values = [info['balance']]
        balances = [info['balance']]
        assets_held = [info['asset_held']]
        ensemble_actions = []
        rewards = []
        prices = []
        timestamps = []

        print("🚀 Starting ensemble simulation...")

        while not done:
            # Get predictions from all models
            individual_actions = []
            for model in self.models:
                action, _ = model.predict(obs, deterministic=True)
                individual_actions.append(action)

            # IMPROVED: Weighted average of actions
            ensemble_action = np.average(individual_actions, weights=self.weights, axis=0)
            ensemble_actions.append(ensemble_action.copy())

            # Execute ensemble action
            obs, reward, terminated, truncated, info = env.step(ensemble_action)
            done = terminated or truncated

            step_count += 1
            rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
            balances.append(info['balance'])
            assets_held.append(info['asset_held'])

            # Get current price and timestamp
            current_price = env.timeframes[env.cfg.base_bar_timeframe]['close'].iloc[env.current_step]
            current_timestamp = env.base_timestamps[env.current_step]

            prices.append(current_price)
            timestamps.append(current_timestamp)

            # Progress reporting
            if step_count % 1000 == 0:
                initial_value = portfolio_values[0]
                current_return = (info['portfolio_value'] - initial_value) / initial_value * 100
                print(f"Step {step_count:,}: Portfolio ${info['portfolio_value']:,.2f} "
                      f"({current_return:+.2f}%)")

        # Calculate metrics for ensemble
        backtester = AdvancedBacktester()
        metrics = backtester._calculate_comprehensive_metrics_vectorized(
            portfolio_values, prices, timestamps, ensemble_actions, rewards
        )

        # Display ensemble results
        print("\n🎯 ENSEMBLE RESULTS")
        print("="*60)
        backtester._display_results(metrics)

        return {
            'metrics': metrics,
            'portfolio_values': portfolio_values,
            'balances': balances,
            'assets_held': assets_held,
            'prices': prices,
            'timestamps': timestamps,
            'actions': ensemble_actions,
            'rewards': rewards,
            'ensemble_type': 'weighted_average',
            'model_weights': self.weights
        }

    def _create_ensemble_comparison(self):
        """Create comparison chart of all models."""
        comparison_data = []

        # Add individual model results
        for i, results in enumerate(self.individual_results):
            metrics = results['metrics']
            comparison_data.append({
                'Model': f'Model {i+1}',
                'Model_Type': 'Individual',
                'Total Return': metrics['total_return'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Calmar Ratio': metrics['calmar_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate']
            })

        df = pd.DataFrame(comparison_data)

        print("\n📊 MODEL COMPARISON")
        print("="*80)
        print(df.to_string(index=False, float_format='%.3f'))

        # Save comparison to file
        results_dir = Path(SETTINGS.base_path) / "backtest_results"
        results_dir.mkdir(exist_ok=True)
        df.to_csv(results_dir / "ensemble_comparison.csv", index=False)

        print(f"\n📁 Comparison saved to: {results_dir / 'ensemble_comparison.csv'}")

# --- CONVENIENCE FUNCTIONS ---

def run_backtest(model_path: str = None) -> Dict:
    """Convenience function for running enhanced backtest."""
    backtester = AdvancedBacktester(model_path)
    return backtester.run_backtest()

def run_ensemble_backtest(model_paths: List[str], weights: Optional[List[float]] = None) -> Dict:
    """IMPROVED: Convenience function for running true ensemble backtest."""
    ensemble = EnsembleBacktester(model_paths, weights)
    return ensemble.run_ensemble_backtest()

# Backwards compatibility
def run_backtest_legacy():
    """Legacy function name for backwards compatibility."""
    return run_backtest()

if __name__ == "__main__":
    # Example usage
    results = run_backtest()
    print("✅ Backtest completed successfully!")
