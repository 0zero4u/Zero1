# enhanced_evaluator_verbose.py

# REFINED: evaluator.py with Fixed Transformer Backtesting and Corrected Import
# UPDATED: Now compatible with the high-performance, pre-computed engine.
# VERBOSE FIX: Now imports from the correct `enhanced_engine_verbose` module.

"""
REFINEMENT: Enhanced Backtesting System with Fixed Transformer Architecture Support

Key fix: Removed unnecessary LSTM state handling since Transformers don't have 
recurrent states that need to be passed between prediction steps.

CRITICAL IMPORT FIX: Updated to import the correctly named `FixedHierarchicalTradingEnvironment`
from the updated `enhanced_engine_verbose.py` to prevent ImportError during backtesting.

CRITICAL ENGINE COMPATIBILITY FIX: The system now generates and passes the required
`precomputed_features` DataFrame to the optimized environment constructor.
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
import logging

# Optional dependencies with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available, some visualizations disabled")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available, interactive visualizations disabled")

warnings.filterwarnings('ignore')

# Import from local modules - fixed import paths
from processor import create_bars_from_trades, generate_stateful_features_for_fitting
from config import SETTINGS
# --- START OF IMPORT FIX ---
# This now correctly imports from your enhanced, verbose engine file.
from enhanced_engine_verbose import FixedHierarchicalTradingEnvironment as EnhancedHierarchicalTradingEnvironment
# --- END OF IMPORT FIX ---
from normalizer import Normalizer

logger = logging.getLogger(__name__)

class AdvancedBacktester:
    """Enhanced backtesting system with fixed Transformer architecture support."""

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
        logger.info(f"🔍 Starting Enhanced Backtest for {period.upper()}")
        print("="*60)
        
        try:
            # 1. Prepare Environment Data
            bars_df = create_bars_from_trades(period)
            
            # --- START OF FIX for Optimized Engine ---
            # The optimized environment requires pre-calculated features.
            # We must generate them here before creating the environment.
            logger.info("Generating stateful (context) features for backtesting...")
            # This generates the context features (e.g., VWAP distance, S/R distances)
            context_features_df = generate_stateful_features_for_fitting(bars_df, self.cfg.strategy)

            # The `bars_df` already contains the simple precomputed features (e.g., typical_price, volatility).
            # We merge them to create a single comprehensive features DataFrame.
            all_features_df = pd.merge(bars_df, context_features_df, on='timestamp', how='left')
            # Fill any NaNs that might result from the merge or calculations
            all_features_df.fillna(method='ffill', inplace=True)
            all_features_df.fillna(0.0, inplace=True)
            # --- END OF FIX for Optimized Engine ---

            # Load the pre-fitted normalizer
            logger.info("Loading pre-fitted normalizer...")
            normalizer = Normalizer(self.cfg.strategy)
            normalizer_path = Path(self.cfg.get_normalizer_path())
            
            try:
                normalizer.load(normalizer_path)
                logger.info(f"✅ Normalizer loaded from {normalizer_path}")
            except FileNotFoundError:
                logger.error(f"FATAL: Normalizer file not found at {normalizer_path}.")
                logger.error("You must run the training process first to create the normalizer file.")
                raise
            
            # --- START OF FIX for Optimized Engine ---
            # Inject the normalizer AND the precomputed features into the environment
            logger.info("Initializing high-performance backtesting environment...")
            env = EnhancedHierarchicalTradingEnvironment(
                df_base_ohlc=bars_df, 
                normalizer=normalizer,
                precomputed_features=all_features_df,  # <-- Pass the MERGED features here
                worker_id=0 # Ensure backtesting runs in verbose mode if desired
            )
            # --- END OF FIX for Optimized Engine ---
            
            # 2. Load Model
            logger.warning("SECURITY WARNING: Loading model files can execute arbitrary code.")
            logger.warning(f"Ensure {self.model_path} is from a trusted source.")
            model = PPO.load(self.model_path, env=env)
            logger.info(f"✅ Loaded model from: {self.model_path}")
            
            # 3. Initialize Tracking
            obs, info = env.reset()
            done = False
            step_count = 0
            
            # REFINEMENT: Removed LSTM state handling since Transformers are stateless
            
            # Initialize metrics tracking
            portfolio_values = [info['balance']]
            balances = [info['balance']]
            assets_held = [info['asset_held']]
            actions_taken = []
            rewards = []
            prices = []
            timestamps = []
            initial_value = portfolio_values[0]
            
            logger.info(f"🚀 Starting simulation with initial portfolio: ${initial_value:,.2f}")
            print("-" * 60)
            
            # 4. Run Simulation
            while not done:
                try:
                    # REFINEMENT: Fixed model prediction for Transformer architecture
                    action = model.predict(obs, deterministic=True)
                    
                    if isinstance(action, tuple):
                        action = action[0]
                    
                    actions_taken.append(action.copy())
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    step_count += 1
                    rewards.append(reward)
                    portfolio_values.append(info['portfolio_value'])
                    balances.append(info['balance'])
                    assets_held.append(info['asset_held'])
                    
                    # OPTIMIZATION: Access price via direct NumPy indexing in the optimized env
                    current_price = env.timeframes_np[env.cfg.base_bar_timeframe.value]['close'][env.current_step]
                    current_timestamp = env.base_timestamps[env.current_step]
                    prices.append(current_price)
                    timestamps.append(current_timestamp)
                    
                    if save_detailed_logs:
                        self._log_trade_details(step_count, action, reward, info, current_price, current_timestamp)
                    
                    self._extract_model_insights(model, step_count)
                    
                    if step_count % 1000 == 0:
                        current_return = (info['portfolio_value'] - initial_value) / initial_value * 100
                        logger.info(f"Step {step_count:,}: Portfolio ${info['portfolio_value']:,.2f} "
                                  f"({current_return:+.2f}%) | Balance: ${info['balance']:,.2f} | "
                                  f"Assets: {info['asset_held']:.6f}")
                    
                    if step_count % 5000 == 0:
                        self._display_model_reasoning(model, action, step_count)
                        
                except Exception as e:
                    logger.error(f"Error in simulation step {step_count}: {e}")
                    break
            
            # 5. Calculate Comprehensive Metrics
            logger.info("📊 CALCULATING PERFORMANCE METRICS")
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
            
            logger.info("✅ Enhanced backtest completed!")
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

    def _log_trade_details(self, step: int, action: np.ndarray, reward: float,
                          info: Dict, price: float, timestamp: pd.Timestamp):
        """Log detailed trade information."""
        try:
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
        except Exception as e:
            logger.error(f"Error logging trade details: {e}")

    def _extract_model_insights(self, model: PPO, step: int):
        """Extract attention weights and regime probabilities for analysis."""
        try:
            if hasattr(model.policy.features_extractor, 'last_attention_weights'):
                attention_weights = model.policy.features_extractor.last_attention_weights
                if attention_weights is not None:
                    entry = {
                        'step': step,
                        'flow_weight': float(attention_weights[0, 0]),
                        'volatility_weight': float(attention_weights[0, 1]),
                        'value_trend_weight': float(attention_weights[0, 2]),
                        'context_weight': float(attention_weights[0, 3])
                    }
                    
                    if attention_weights.shape[1] > 4:
                        entry['precomputed_weight'] = float(attention_weights[0, 4])
                    
                    self.attention_history.append(entry)
        except Exception as e:
            pass

    def _display_model_reasoning(self, model: PPO, action: np.ndarray, step: int):
        """Display detailed model reasoning."""
        try:
            if hasattr(model.policy.features_extractor, 'last_attention_weights'):
                weights = model.policy.features_extractor.last_attention_weights
                if weights is not None:
                    attention_dict = {
                        "Flow_Factors": f"{weights[0, 0]:.1%}",
                        "Volatility_Factors": f"{weights[0, 1]:.1%}",
                        "Value_Trend_Factors": f"{weights[0, 2]:.1%}",
                        "Context_Factors": f"{weights[0, 3]:.1%}"
                    }
                    
                    if weights.shape[1] > 4:
                        attention_dict["Precomputed_Factors"] = f"{weights[0, 4]:.1%}"
                    
                    reasoning = {
                        "Step": step,
                        "Model_Decision": {
                            "Position_Signal": f"{float(action[0]):.3f}",
                            "Position_Size": f"{float(action[1]):.3f}",
                            "Expert_Attention": attention_dict
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
        """Vectorized performance metrics calculation for better performance."""
        try:
            portfolio_array = np.array(portfolio_values[1:])
            price_array = np.array(prices)
            
            portfolio_series = pd.Series(portfolio_array, index=timestamps)
            price_series = pd.Series(price_array, index=timestamps)
            
            if len(timestamps) < 2 or portfolio_series.empty:
                logger.warning("Not enough data points to calculate metrics.")
                return {}
            
            total_duration = timestamps[-1] - timestamps[0]
            total_days = max(1.0, total_duration.total_seconds() / (24 * 3600))
            
            base_bar_seconds = self.cfg.get_timeframe_seconds(self.cfg.base_bar_timeframe)
            bars_per_year = (365 * 24 * 3600) / base_bar_seconds
            
            returns = portfolio_series.pct_change().dropna()
            price_returns = price_series.pct_change().dropna()
            
            if returns.empty:
                logger.warning("No returns calculated, cannot compute performance metrics.")
                return {}
            
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value
            annualized_return = total_return * (365 / total_days)
            
            volatility = returns.std() * np.sqrt(bars_per_year)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(bars_per_year) if len(downside_returns) > 0 else 0
            
            cumulative_max = np.maximum.accumulate(portfolio_array)
            drawdowns_array = (cumulative_max - portfolio_array) / (cumulative_max + 1e-9)
            max_drawdown = np.max(drawdowns_array) if len(drawdowns_array) > 0 else 0
            
            drawdown_periods = self._calculate_drawdown_periods_vectorized(drawdowns_array)
            avg_drawdown_duration = np.mean(drawdown_periods) if len(drawdown_periods) > 0 else 0
            max_drawdown_duration = np.max(drawdown_periods) if len(drawdown_periods) > 0 else 0
            
            risk_free_rate = 0.02
            excess_returns = annualized_return - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 1e-8 else 0
            sortino_ratio = excess_returns / downside_volatility if downside_volatility > 1e-8 else 0
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown > 1e-8 else 0
            
            benchmark_return = (prices[-1] - prices[0]) / prices[0] if len(prices) > 0 else 0
            excess_return_vs_benchmark = total_return - benchmark_return
            
            actions_array = np.array(actions)
            position_signals = actions_array[:, 0] if len(actions_array) > 0 else []
            position_sizes = actions_array[:, 1] if len(actions_array) > 0 else []
            
            significant_trades = 0
            if len(actions_array) > 1:
                position_changes = np.abs(np.diff(actions_array[:, 0]))
                significant_trades = np.sum(position_changes > 0.1)
            
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
            avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            aligned_returns = returns.reindex(price_returns.index, fill_value=0)
            tracking_error = (aligned_returns - price_returns).std() * np.sqrt(bars_per_year)
            information_ratio = excess_return_vs_benchmark / tracking_error if tracking_error > 1e-8 else 0
            
            var_95 = np.percentile(returns, 5) if not returns.empty else 0
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            
            return {
                'total_return': total_return, 'annualized_return': annualized_return,
                'final_portfolio_value': final_value, 'initial_portfolio_value': initial_value,
                'volatility': volatility, 'downside_volatility': downside_volatility,
                'max_drawdown': max_drawdown, 'avg_drawdown_duration_days': avg_drawdown_duration,
                'max_drawdown_duration_days': max_drawdown_duration, 'var_95': var_95, 'cvar_95': cvar_95,
                'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio, 'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio, 'benchmark_return': benchmark_return,
                'excess_return_vs_benchmark': excess_return_vs_benchmark, 'total_trades': len(actions),
                'significant_trades': int(significant_trades), 'avg_position_signal': np.mean(position_signals) if len(position_signals) > 0 else 0,
                'avg_position_size': np.mean(position_sizes) if len(position_sizes) > 0 else 0, 'win_rate': win_rate,
                'profit_factor': profit_factor, 'avg_win': avg_win, 'avg_loss': avg_loss, 'total_days': total_days,
                'positive_days': len(positive_returns), 'negative_days': len(negative_returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def _calculate_drawdown_periods_vectorized(self, drawdowns_array: np.ndarray) -> List[int]:
        """Vectorized drawdown period calculation."""
        try:
            in_drawdown = drawdowns_array > 0.01
            
            transitions = np.diff(in_drawdown.astype(int))
            starts = np.where(transitions == 1)[0] + 1
            ends = np.where(transitions == -1)[0] + 1
            
            if len(starts) == 0: return []
            if len(starts) > len(ends): ends = np.append(ends, len(drawdowns_array))
            if len(ends) > len(starts): starts = np.insert(starts, 0, 0)
            
            durations = ends - starts
            return durations.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating drawdown periods: {e}")
            return []

    def _display_results(self, metrics: Dict):
        """Display comprehensive results in a formatted manner."""
        try:
            print("\n📈 PERFORMANCE SUMMARY")
            print("-" * 40)
            print(f"Initial Portfolio Value: ${metrics.get('initial_portfolio_value', 0):,.2f}")
            print(f"Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"Benchmark Return: {metrics.get('benchmark_return', 0):.2%}")
            print(f"Excess Return: {metrics.get('excess_return_vs_benchmark', 0):+.2%}")
            
            print(f"\n⚖️ RISK METRICS")
            print("-" * 40)
            print(f"Volatility (Annual): {metrics.get('volatility', 0):.2%}")
            print(f"Downside Volatility: {metrics.get('downside_volatility', 0):.2%}")
            print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"Avg Drawdown Duration: {metrics.get('avg_drawdown_duration_days', 0):.1f} days")
            print(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
            print(f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
            
            print(f"\n🎯 RISK-ADJUSTED RETURNS")
            print("-" * 40)
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
            print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
            print(f"Information Ratio: {metrics.get('information_ratio', 0):.3f}")
            
            print(f"\n🔄 TRADING ACTIVITY")
            print("-" * 40)
            print(f"Total Actions: {metrics.get('total_trades', 0):,}")
            print(f"Significant Trades: {metrics.get('significant_trades', 0):,}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Average Win: {metrics.get('avg_win', 0):.3%}")
            print(f"Average Loss: {metrics.get('avg_loss', 0):.3%}")
            
            rating = self._calculate_performance_rating(metrics)
            print(f"\n🏆 OVERALL PERFORMANCE RATING: {rating}")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")

    def _calculate_performance_rating(self, metrics: Dict) -> str:
        """Calculate overall performance rating."""
        try:
            score = 0
            annualized_return = metrics.get('annualized_return', 0)
            if annualized_return > 0.2: score += 25
            elif annualized_return > 0.1: score += 20
            elif annualized_return > 0.05: score += 15
            elif annualized_return > 0: score += 10
            
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 2: score += 25
            elif sharpe_ratio > 1.5: score += 20
            elif sharpe_ratio > 1: score += 15
            elif sharpe_ratio > 0.5: score += 10
            
            max_drawdown = metrics.get('max_drawdown', 0)
            if max_drawdown > -0.05: score += 25
            elif max_drawdown > -0.1: score += 20
            elif max_drawdown > -0.2: score += 15
            elif max_drawdown > -0.3: score += 10
            
            win_rate = metrics.get('win_rate', 0)
            if win_rate > 0.6: score += 25
            elif win_rate > 0.55: score += 20
            elif win_rate > 0.5: score += 15
            elif win_rate > 0.45: score += 10
            
            if score >= 90: return "EXCELLENT ⭐⭐⭐⭐⭐"
            elif score >= 75: return "VERY GOOD ⭐⭐⭐⭐"
            elif score >= 60: return "GOOD ⭐⭐⭐"
            elif score >= 40: return "FAIR ⭐⭐"
            else: return "POOR ⭐"
            
        except Exception as e:
            logger.error(f"Error calculating performance rating: {e}")
            return "UNKNOWN"

    def _generate_visualizations(self):
        """Generate comprehensive visualizations."""
        try:
            logger.info("📊 Generating visualizations...")
            results_dir = Path(self.cfg.base_path) / "backtest_results"
            results_dir.mkdir(exist_ok=True)
            
            if PLOTLY_AVAILABLE:
                self._create_portfolio_performance_chart(results_dir)
                self._create_drawdown_chart(results_dir)
                if self.attention_history:
                    self._create_attention_heatmap(results_dir)
                self._create_trade_analysis_chart(results_dir)
                self._create_risk_dashboard(results_dir)
                logger.info(f"✅ Visualizations saved to: {results_dir}")
            else:
                logger.warning("Plotly not available, skipping interactive visualizations")
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    def _create_portfolio_performance_chart(self, results_dir: Path):
        """Create portfolio performance comparison chart."""
        try:
            if not PLOTLY_AVAILABLE: return
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Portfolio Value vs Benchmark', 'Daily Returns'), vertical_spacing=0.08)
            timestamps, portfolio_values, prices = self.results['timestamps'], self.results['portfolio_values'][1:], self.results['prices']
            initial_portfolio = self.results['portfolio_values'][0]
            portfolio_returns = [(pv / initial_portfolio - 1) * 100 for pv in portfolio_values]
            initial_price = prices[0]
            benchmark_returns = [(p / initial_price - 1) * 100 for p in prices]
            fig.add_trace(go.Scatter(x=timestamps, y=portfolio_returns, name="Portfolio", line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=timestamps, y=benchmark_returns, name="Benchmark (Buy & Hold)", line=dict(color='red')), row=1, col=1)
            returns = pd.Series(portfolio_values).pct_change().fillna(0) * 100
            fig.add_trace(go.Scatter(x=timestamps, y=returns, name="Daily Returns", line=dict(color='green'), opacity=0.7), row=2, col=1)
            fig.update_layout(title="Portfolio Performance Analysis", height=800, showlegend=True)
            fig.write_html(str(results_dir / "portfolio_performance.html"))
        except Exception as e:
            logger.error(f"Error creating portfolio performance chart: {e}")

    def _create_drawdown_chart(self, results_dir: Path):
        """Create drawdown analysis chart."""
        try:
            if not PLOTLY_AVAILABLE: return
            portfolio_series = pd.Series(self.results['portfolio_values'][1:], index=self.results['timestamps'])
            cumulative_max = portfolio_series.cummax()
            drawdown = (portfolio_series - cumulative_max) / cumulative_max * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.results['timestamps'], y=drawdown, fill='tonexty', name='Drawdown %', line=dict(color='red'), fillcolor='rgba(255, 0, 0, 0.3)'))
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            fig.update_layout(title="Portfolio Drawdown Analysis", xaxis_title="Date", yaxis_title="Drawdown (%)", height=500)
            fig.write_html(str(results_dir / "drawdown_analysis.html"))
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")

    def _create_attention_heatmap(self, results_dir: Path):
        """Create attention weights heatmap over time."""
        try:
            if not PLOTLY_AVAILABLE: return
            attention_df = pd.DataFrame(self.attention_history)
            if len(attention_df) > 0:
                if len(attention_df) > 1000:
                    attention_df = attention_df.iloc[::len(attention_df)//1000]
                all_weight_cols = ['flow_weight', 'volatility_weight', 'value_trend_weight', 'context_weight', 'precomputed_weight']
                all_y_labels = ['Flow Factors', 'Volatility Factors', 'Value/Trend Factors', 'Context Factors', 'Precomputed Factors']
                existing_cols = [col for col in all_weight_cols if col in attention_df.columns]
                existing_labels = [all_y_labels[i] for i, col in enumerate(all_weight_cols) if col in existing_cols]
                weights_data = attention_df[existing_cols].T
                fig = go.Figure(data=go.Heatmap(z=weights_data.values, x=attention_df['step'], y=existing_labels, colorscale='Viridis', hoverongaps=False))
                fig.update_layout(title="Model Attention Weights Over Time", xaxis_title="Training Step", yaxis_title="Expert Type", height=500)
                fig.write_html(str(results_dir / "attention_heatmap.html"))
        except Exception as e:
            logger.error(f"Error creating attention heatmap: {e}")

    def _create_trade_analysis_chart(self, results_dir: Path):
        """Create trade analysis visualization."""
        try:
            if not PLOTLY_AVAILABLE or not self.trade_log: return
            trade_df = pd.DataFrame(self.trade_log)
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Action Signals Over Time', 'Position Sizes Over Time', 'Reward Distribution', 'Price vs Portfolio Value'))
            fig.add_trace(go.Scatter(x=trade_df['step'], y=trade_df['action_signal'], name="Action Signal", line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=trade_df['step'], y=trade_df['action_size'], name="Position Size", line=dict(color='green')), row=1, col=2)
            fig.add_trace(go.Histogram(x=trade_df['reward'], name="Rewards", nbinsx=50), row=2, col=1)
            fig.add_trace(go.Scatter(x=trade_df['price'], y=trade_df['portfolio_value'], mode='markers', name="Price vs Portfolio", marker=dict(color=trade_df['reward'], colorscale='RdYlGn', colorbar=dict(title="Reward"))), row=2, col=2)
            fig.update_layout(title="Trading Activity Analysis", height=800, showlegend=False)
            fig.write_html(str(results_dir / "trade_analysis.html"))
        except Exception as e:
            logger.error(f"Error creating trade analysis chart: {e}")

    def _create_risk_dashboard(self, results_dir: Path):
        """Create comprehensive risk metrics dashboard."""
        try:
            if not PLOTLY_AVAILABLE: return
            metrics = self.results['metrics']
            fig = make_subplots(rows=2, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}], [{"type": "indicator"}, {"type": "indicator"}]], subplot_titles=['Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Volatility'])
            fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.get('sharpe_ratio', 0), domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Sharpe Ratio"}, gauge={'axis': {'range': [None, 3]}, 'bar': {'color': "darkblue"}, 'steps': [{'range': [0, 1], 'color': "lightgray"}, {'range': [1, 2], 'color': "gray"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1.5}}), row=1, col=1)
            fig.add_trace(go.Indicator(mode="gauge+number", value=abs(metrics.get('max_drawdown', 0)) * 100, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Max Drawdown %"}, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "red"}, 'steps': [{'range': [0, 10], 'color': "lightgreen"}, {'range': [10, 20], 'color': "yellow"}], 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 15}}), row=1, col=2)
            fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.get('win_rate', 0) * 100, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Win Rate %"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}, 'steps': [{'range': [0, 40], 'color': "lightgray"}, {'range': [40, 60], 'color': "yellow"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}), row=2, col=1)
            fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.get('volatility', 0) * 100, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Volatility %"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "orange"}, 'steps': [{'range': [0, 20], 'color': "lightgreen"}, {'range': [20, 40], 'color': "yellow"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 30}}), row=2, col=2)
            fig.update_layout(title="Risk Metrics Dashboard", height=800)
            fig.write_html(str(results_dir / "risk_dashboard.html"))
        except Exception as e:
            logger.error(f"Error creating risk dashboard: {e}")

    def _save_results_to_files(self):
        """Save detailed results to CSV files."""
        try:
            results_dir = Path(self.cfg.base_path) / "backtest_results"
            if self.results.get('metrics'):
                pd.DataFrame([self.results['metrics']]).to_csv(results_dir / "performance_metrics.csv", index=False)
            if self.trade_log:
                pd.DataFrame(self.trade_log).to_csv(results_dir / "detailed_trade_log.csv", index=False)
            if self.attention_history:
                pd.DataFrame(self.attention_history).to_csv(results_dir / "attention_weights_history.csv", index=False)
            if self.results.get('timestamps'):
                pd.DataFrame({
                    'timestamp': self.results['timestamps'], 'portfolio_value': self.results['portfolio_values'][1:],
                    'price': self.results['prices'], 'balance': self.results['balances'][1:],
                    'asset_held': self.results['assets_held'][1:]
                }).to_csv(results_dir / "portfolio_timeseries.csv", index=False)
            logger.info(f"📁 Detailed results saved to: {results_dir}")
        except Exception as e:
            logger.error(f"Error saving results to files: {e}")

# --- ENSEMBLE BACKTESTING ---

class EnsembleBacktester:
    """Backtest ensemble of models with true action averaging."""
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        self.model_paths = model_paths
        self.weights = weights or [1.0] * len(model_paths)
        self.individual_results = []
        self.models = []

    def run_ensemble_backtest(self) -> Dict:
        """Run backtest for ensemble of models with true ensembling."""
        try:
            logger.info(f"🤖 Running ensemble backtest with {len(self.model_paths)} models")
            bars_df = create_bars_from_trades("out_of_sample")
            
            # --- START OF FIX for Optimized Engine ---
            logger.info("Generating stateful (context) features for ensemble backtesting...")
            context_features_df = generate_stateful_features_for_fitting(bars_df, SETTINGS.strategy)
            
            # Merge context features with the precomputed features already in bars_df
            all_features_df = pd.merge(bars_df, context_features_df, on='timestamp', how='left')
            all_features_df.fillna(method='ffill', inplace=True)
            all_features_df.fillna(0.0, inplace=True)
            # --- END OF FIX for Optimized Engine ---

            logger.info("Loading pre-fitted normalizer for ensemble...")
            normalizer = Normalizer(SETTINGS.strategy)
            normalizer_path = Path(SETTINGS.get_normalizer_path())
            try:
                normalizer.load(normalizer_path)
                logger.info(f"✅ Normalizer loaded from {normalizer_path}")
            except FileNotFoundError:
                logger.error(f"FATAL: Normalizer file not found at {normalizer_path}.")
                raise
            
            # --- START OF FIX for Optimized Engine ---
            env = EnhancedHierarchicalTradingEnvironment(
                df_base_ohlc=bars_df, 
                normalizer=normalizer,
                precomputed_features=all_features_df # <-- Pass the MERGED features here
            )
            # --- END OF FIX for Optimized Engine ---
            
            for i, model_path in enumerate(self.model_paths):
                logger.info(f"Loading model {i+1}/{len(self.model_paths)}: {model_path}")
                try:
                    model = PPO.load(model_path, env=env)
                    self.models.append(model)
                except Exception as e:
                    logger.error(f"Failed to load model {model_path}: {e}")
                    continue
            
            if not self.models: raise ValueError("No models could be loaded")
            
            logger.info("--- Running Ensemble Backtest ---")
            ensemble_results = self._run_true_ensemble_backtest(env)
            
            for i, model_path in enumerate(self.model_paths):
                logger.info(f"--- Model {i+1}/{len(self.model_paths)} Individual Backtest ---")
                try:
                    backtester = AdvancedBacktester(model_path, save_results=False)
                    results = backtester.run_backtest(save_detailed_logs=False)
                    self.individual_results.append(results)
                except Exception as e:
                    logger.error(f"Individual backtest failed for model {i+1}: {e}")
            
            self._create_ensemble_comparison()
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Ensemble backtest failed: {e}")
            raise

    def _run_true_ensemble_backtest(self, env: EnhancedHierarchicalTradingEnvironment) -> Dict:
        """True ensemble backtesting with action averaging."""
        try:
            obs, info = env.reset()
            done = False
            step_count = 0
            
            portfolio_values, balances, assets_held, ensemble_actions, rewards, prices, timestamps = [info['balance']], [info['balance']], [info['asset_held']], [], [], [], []
            
            logger.info("🚀 Starting ensemble simulation...")
            
            while not done:
                try:
                    individual_actions = []
                    for i, model in enumerate(self.models):
                        action = model.predict(obs, deterministic=True)
                        if isinstance(action, tuple):
                            action = action[0]
                        individual_actions.append(action)
                    
                    ensemble_action = np.average(individual_actions, weights=self.weights, axis=0)
                    ensemble_actions.append(ensemble_action.copy())
                    
                    obs, reward, terminated, truncated, info = env.step(ensemble_action)
                    done = terminated or truncated
                    
                    step_count += 1
                    rewards.append(reward)
                    portfolio_values.append(info['portfolio_value'])
                    balances.append(info['balance'])
                    assets_held.append(info['asset_held'])
                    
                    current_price = env.timeframes_np[env.cfg.base_bar_timeframe.value]['close'][env.current_step]
                    current_timestamp = env.base_timestamps[env.current_step]
                    prices.append(current_price)
                    timestamps.append(current_timestamp)
                    
                    if step_count % 1000 == 0:
                        initial_value = portfolio_values[0]
                        current_return = (info['portfolio_value'] - initial_value) / initial_value * 100
                        logger.info(f"Step {step_count:,}: Portfolio ${info['portfolio_value']:,.2f} "
                                  f"({current_return:+.2f}%)")
                        
                except Exception as e:
                    logger.error(f"Error in ensemble step {step_count}: {e}")
                    break
            
            backtester = AdvancedBacktester()
            metrics = backtester._calculate_comprehensive_metrics_vectorized(
                portfolio_values, prices, timestamps, ensemble_actions, rewards
            )
            
            logger.info("🎯 ENSEMBLE RESULTS")
            print("="*60)
            backtester._display_results(metrics)
            
            return {
                'metrics': metrics, 'portfolio_values': portfolio_values, 'balances': balances,
                'assets_held': assets_held, 'prices': prices, 'timestamps': timestamps,
                'actions': ensemble_actions, 'rewards': rewards, 'ensemble_type': 'weighted_average',
                'model_weights': self.weights
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble backtest execution: {e}")
            raise

    def _create_ensemble_comparison(self):
        """Create comparison chart of all models."""
        try:
            comparison_data = []
            for i, results in enumerate(self.individual_results):
                if not results or not results.get('metrics'): continue
                metrics = results['metrics']
                comparison_data.append({
                    'Model': f'Model {i+1}', 'Model_Type': 'Individual',
                    'Total Return': metrics.get('total_return', 0), 'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Calmar Ratio': metrics.get('calmar_ratio', 0), 'Max Drawdown': metrics.get('max_drawdown', 0),
                    'Win Rate': metrics.get('win_rate', 0)
                })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                logger.info("📊 MODEL COMPARISON")
                print("="*80)
                print(df.to_string(index=False, float_format='%.3f'))
                results_dir = Path(SETTINGS.base_path) / "backtest_results"
                results_dir.mkdir(exist_ok=True)
                df.to_csv(results_dir / "ensemble_comparison.csv", index=False)
                logger.info(f"📁 Comparison saved to: {results_dir / 'ensemble_comparison.csv'}")
                
        except Exception as e:
            logger.error(f"Error creating ensemble comparison: {e}")

# --- CONVENIENCE FUNCTIONS ---

def run_backtest(model_path: str = None, period: str = "out_of_sample") -> Dict:
    """Convenience function for running enhanced backtest."""
    try:
        backtester = AdvancedBacktester(model_path)
        return backtester.run_backtest(period=period)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

def run_ensemble_backtest(model_paths: List[str], weights: Optional[List[float]] = None) -> Dict:
    """Convenience function for running true ensemble backtest."""
    try:
        ensemble = EnsembleBacktester(model_paths, weights)
        return ensemble.run_ensemble_backtest()
    except Exception as e:
        logger.error(f"Ensemble backtest failed: {e}")
        raise

# Backwards compatibility
def run_backtest_legacy():
    """Legacy function name for backwards compatibility."""
    return run_backtest()

if __name__ == "__main__":
    # Example usage
    try:
        logger.info("Testing refined backtesting system...")
        logger.info("✅ REFINED: Evaluator module loaded successfully with fixed Transformer support!")
        print("🔧 REFINEMENTS APPLIED:")
        print(" - Removed unnecessary LSTM state handling for Transformer models")
        print(" - Fixed model.predict() calls for stateless architecture")
        print(" - Enhanced ensemble backtesting with proper Transformer support")
        print(" - Maintained all existing functionality and visualizations")
        print(" - UPDATED to be compatible with high-performance NumPy-based engine")
        
    except Exception as e:
        logger.error(f"Evaluator test failed: {e}")
