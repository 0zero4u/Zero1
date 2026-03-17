# ==========================================
# 3. PURE AI EXECUTION ENGINE (AUTOPILOT PIPELINE)
# ==========================================
import asyncio
import websockets
import json
import time
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import io
import optuna
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from river import stats

# FIX: MarketSnapshot strictly aligns asynchronous Bid/Ask pricing timestamps
class MarketSnapshot:
    def __init__(self):
        self.bid = 0.0
        self.ask = 0.0
        self.last_book_update = 0

    def update_book(self, bid, ask, timestamp):
        if timestamp >= self.last_book_update:
            self.bid = bid
            self.ask = ask
            self.last_book_update = timestamp

    def get_spread(self):
        return self.ask - self.bid if self.bid > 0 and self.ask > 0 else 0.0

# ─── PRE-OPTUNA SHAP SURROGATE ANALYSIS (FULL XAI) ───────────────────────────
def run_shap_surrogate_analysis(warmup_features_list, warmup_labels_list, top_n=25):
    print("\n[🧠 SYSTEM] Initiating LightGBM Surrogate Training for SHAP Analysis...")
    df_X = pd.DataFrame(warmup_features_list)
    df_y = pd.Series(warmup_labels_list)

    # ─────────────────────────────────────────────────────────
    # 1. DETECT COLLINEARITY (Highly Correlated Features)
    # ─────────────────────────────────────────────────────────
    print(f"\n[🔍 XAI] Checking for Feature Collinearity among all {len(df_X.columns)} features (Spearman ρ > 0.85)...")
    corr_matrix = df_X.corr(method='spearman').abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    collinear_pairs = []
    for col in upper_tri.columns:
        high_corr_idx = upper_tri[col][upper_tri[col] > 0.85].index.tolist()
        for hc in high_corr_idx:
            collinear_pairs.append((hc, col, upper_tri[col][hc]))

    if collinear_pairs:
        for f1, f2, val in sorted(collinear_pairs, key=lambda x: x[2], reverse=True):
            print(f"   ⚠️ WARNING: {f1} <--> {f2} (Correlation: {val:.2f})")
        print("   💡 Note: High collinearity causes Hoeffding Trees to split importance. Consider dropping redundant features.")
    else:
        print("   ✅ No extreme collinearity detected.")

    # ─────────────────────────────────────────────────────────
    # 2. TRAIN SURROGATE & CALCULATE SHAP
    # ─────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=25, num_leaves=50, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(df_X, df_y)
    print("\n[✅ SYSTEM] Surrogate Model Trained. Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X)

    # Handle multi-class vs binary output from SHAP
    if isinstance(shap_values, list):
        shap_abs = np.zeros(df_X.shape[1])
        for sv in shap_values:
            shap_abs += np.abs(sv).mean(axis=0)
    else:
        shap_abs = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({'feature': df_X.columns, 'importance': shap_abs})
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────
    # 3. PRINT ALL FEATURES (Insights)
    # ─────────────────────────────────────────────────────────
    print(f"\n[🏆 SHAP] FULL FEATURE RANKING (All {len(df_X.columns)} Features):")
    for i, row in feature_importance.iterrows():
        print(f"   {i+1:2d}. {row['feature']:<25} (Impact Score: {row['importance']:.4f})")

    # ─────────────────────────────────────────────────────────
    # 4. GENERATE VISUAL BEESWARM INSIGHTS
    # ─────────────────────────────────────────────────────────
    print("\n[📊 XAI] Generating SHAP Insight Plots (Check your folder for PNGs)...")
    plt.figure(figsize=(12, 8))

    target_shap_idx = 1 if isinstance(shap_values, list) and len(shap_values) >= 2 else 0
    shap_to_plot = shap_values[target_shap_idx] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(shap_to_plot, df_X, show=False, max_display=20)
    plt.title("SHAP Feature Insights (Top 20 Drivers)")
    plt.tight_layout()
    plt.savefig("shap_insights_beeswarm.png")
    plt.close()

    top_features = feature_importance.head(top_n)['feature'].tolist()
    print(f"\n[✂️ SYSTEM] Slicing top {top_n} features to pass into Optuna...")
    return top_features

def run_pre_optuna_shap_exploration(stream_data, max_bars=50000):
    print("\n[🔍 SYSTEM] Initiating Pre-Optuna SHAP Feature Exploration...")
    state = SymbolState("SOLUSDT", volume_threshold=3000.0, fracdiff_d=0.35)
    dummy_model = AdaptiveHFTModel("SHAP-Dummy", tp_bps=20, sl_bps=-20)

    bar_count = 0
    snapshot = MarketSnapshot()

    for row in stream_data:
        if row.type == 'book':
            snapshot.update_book(float(row.best_bid), float(row.best_ask), row.time)
        elif row.type == 'trade':
            price, qty = float(row.price), float(row.qty)

            if bar_count > 0:
                dummy_model.process_delayed_labels(bar_count, {"high": price, "low": price, "close": price})

            if state.process_tick(price, qty, bool(row.is_buyer_maker), row.time, snapshot.bid, snapshot.ask):
                bar_count += 1
                features = dummy_model.extract_features(state)
                if features:
                    dummy_model.register_state(bar_count, price, features, pred=0, probas={0: 1.0})

                if bar_count >= max_bars:
                    break

    if len(dummy_model.shap_X) > 1000:
        print(f"[✅ SYSTEM] Harvested {len(dummy_model.shap_X)} resolved labels. Training Surrogate...")
        # Adjusted top_n down since we have fewer total features now (using top 12)
        top_features = run_shap_surrogate_analysis(dummy_model.shap_X, dummy_model.shap_y, top_n=12)
        return top_features
    else:
        print("[⚠️ SYSTEM] Not enough resolved labels to run SHAP reliably. Using all features.")
        return None

# ─── 1. DATA PREPARATION (FULL DAY CACHING) ──────────────────────────────────
def fetch_historical_data():
    print("\n[📥 SYSTEM] Fetching historical data (FULL DAY)...")
    trades_url = "https://data.binance.vision/data/futures/um/daily/trades/SOLUSDT/SOLUSDT-trades-2024-03-19.zip"
    book_url = "https://data.binance.vision/data/futures/um/daily/bookTicker/SOLUSDT/SOLUSDT-bookTicker-2024-03-19.zip"

    def fetch_and_extract(url, col_names):
        req = urllib.request.urlopen(url)
        with zipfile.ZipFile(io.BytesIO(req.read())) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
                if len(df.columns) == len(col_names):
                    df.columns = col_names
                return df

    df_trades = fetch_and_extract(trades_url, ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'])
    df_book = fetch_and_extract(book_url, ['updateId', 'best_bid', 'bid_qty', 'best_ask', 'ask_qty', 'time', 'msg_time'])

    df_trades['type'] = 'trade'
    df_book['type'] = 'book'

    # Filtered down dataframe strips memory overhead
    df_stream = pd.concat([
        df_trades[['time', 'type', 'price', 'qty', 'is_buyer_maker']],
        df_book[['time', 'type', 'best_bid', 'best_ask']]
    ]).sort_values('time').reset_index(drop=True)

    print("[⚡ SYSTEM] Stripping Pandas overhead for pure Python speed...")
    stream_data = list(df_stream.itertuples(index=False))

    print(f"[✅ SYSTEM] Cached {len(stream_data)} historical ticks in ultra-fast RAM array.")
    return stream_data

# ─── 2. OPTUNA OBJECTIVE ──────────────────────────────────────────────────────
def optuna_objective(trial, stream_data, top_features=None):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING OPTUNA TRIAL {trial.number} 🚀")

    tp_bps = trial.suggest_int("tp_bps", 10, 50)
    sl_bps = trial.suggest_int("sl_bps", -50, -10)
    volume_threshold = trial.suggest_float("volume_threshold", 100.0, 15000.0, log=True)

    n_models = trial.suggest_int("n_models", 10, 30)
    max_features = trial.suggest_int("max_features", 8, 12) # Adjusted based on pruned features
    grace_period = trial.suggest_int("grace_period", 50, 500)
    delta = trial.suggest_float("delta", 1e-8, 1e-2, log=True)
    tau = trial.suggest_float("tau", 0.01, 0.1)
    leaf_prediction = trial.suggest_categorical("leaf_prediction", ["nba"])

    max_hold_bars = trial.suggest_int("max_hold_bars", 80, 300)
    fracdiff_d = trial.suggest_float("fracdiff_d", 0.15, 0.85)

    state = SymbolState("SOLUSDT", volume_threshold=volume_threshold, fracdiff_d=fracdiff_d)

    model = AdaptiveHFTModel(
        f"Trial-{trial.number}", tp_bps=tp_bps, sl_bps=sl_bps, fee_bps=2.0,
        n_models=n_models, max_features=max_features, grace_period=grace_period,
        delta=delta, tau=tau, leaf_prediction=leaf_prediction, seed=42,
        max_hold_bars=max_hold_bars, feature_whitelist=top_features
    )

    bar_count = 0
    snapshot = MarketSnapshot()

    for row in stream_data:
        if row.type == 'book':
            snapshot.update_book(float(row.best_bid), float(row.best_ask), row.time)
        elif row.type == 'trade':
            price = float(row.price)
            qty = float(row.qty)
            is_buyer = bool(row.is_buyer_maker)

            if bar_count > 0:
                latest_tick = {"high": price, "low": price, "close": price}
                model.process_delayed_labels(bar_count, latest_tick)

            is_new_bar = state.process_tick(price, qty, is_buyer, row.time, snapshot.bid, snapshot.ask)

            if is_new_bar:
                bar_count += 1
                latest_bar = state.completed_bars[-1]

                features = model.extract_features(state)
                if features:
                    pred, probas = model.predict(features)
                    model.register_state(bar_count, latest_bar["close"], features, pred, probas)

                if bar_count >= 15000:
                    break

    f1_score = model.metric_macro_f1.get()

    print(f"\n🏁 OPTUNA TRIAL {trial.number} COMPLETED 🏁")
    print(f"🧠 MAGIC NUMBERS (Hyperparameters):")
    for key, value in trial.params.items():
        print(f"   ├─ {key}: {value}")

    print("\n📊 TRIAL PERFORMANCE MATRIX (FINAL 10k SNAPSHOT):")
    print(model.format_ev_table())
    print(f"{'='*60}\n")

    return f1_score

# ─── 3. FULL DAY WARM-UP (NO LIMITS) ──────────────────────────────────────────
def warmup_engine_loud(stream_data, symbol_state, micro_model, slow_model, shadow):
    print(f"\n[🔥 SYSTEM] Initiating Full-Day Warm-Up with Winning Parameters...")
    bar_count = 0
    snapshot = MarketSnapshot()

    for row in stream_data:
        if row.type == 'book':
            snapshot.update_book(float(row.best_bid), float(row.best_ask), row.time)
        elif row.type == 'trade':
            price = float(row.price)
            qty = float(row.qty)
            is_buyer = bool(row.is_buyer_maker)

            if bar_count > 0:
                latest_tick = {"high": price, "low": price, "close": price}
                micro_model.process_delayed_labels(bar_count, latest_tick)
                slow_model.process_delayed_labels(bar_count, latest_tick)

            is_new_bar = symbol_state.process_tick(price, qty, is_buyer, row.time, snapshot.bid, snapshot.ask)

            if is_new_bar:
                bar_count += 1
                latest_bar = symbol_state.completed_bars[-1]
                shadow.process_new_bar(bar_count, latest_bar)

                features = micro_model.extract_features(symbol_state)
                if features:
                    micro_pred, micro_probas = micro_model.predict(features)
                    slow_pred, slow_probas = slow_model.predict(features)

                    micro_model.register_state(bar_count, price, features, micro_pred, micro_probas)
                    slow_model.register_state(bar_count, price, features, slow_pred, slow_probas)

                    # FIX: Explicit Memory Leak clear for massive data ingest blocks
                    if bar_count % 5000 == 0:
                        micro_model.cleanup_stale_pending(bar_count - 1000)
                        slow_model.cleanup_stale_pending(bar_count - 1000)

                        print(f"\n{'='*75}")
                        print(f"🔥 WARM-UP PROGRESS: {bar_count} BARS DIGESTED 🔥")
                        print(f"{'='*75}")
                        print(micro_model.format_ev_table(), end="")
                        print(f"{'='*75}\n")

    print(f"[✅ SYSTEM] Warm-up Complete! Processed ALL historical bars.")
    return bar_count

# ─── 4. MAIN ASYNC ENGINE ────────────────────────────────────────────────────
async def live_trading_engine():
    symbol = "solusdt"
    stream_data = fetch_historical_data()

    # XAI FEATURE EXPLORATION - Generates Collinearity + SHAP Ranking
    top_features = run_pre_optuna_shap_exploration(stream_data, max_bars=45000)

    print("\n[🧠 SYSTEM] Starting Optuna Hyperparameter Hunt (10 Trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(lambda trial: optuna_objective(trial, stream_data, top_features), n_trials=10)

    best_params = study.best_params
    print(f"\n[🏆 OPTUNA] Hunt Complete! Best Magic Numbers Found:")
    print(json.dumps(best_params, indent=4))

    base_vol_target = best_params["volume_threshold"]

    symbol_state = SymbolState("SOLUSDT", volume_threshold=base_vol_target, fracdiff_d=best_params["fracdiff_d"])

    micro_model = AdaptiveHFTModel(
        "Micro-Opt",
        tp_bps=best_params["tp_bps"], sl_bps=best_params["sl_bps"], fee_bps=3.0,
        n_models=best_params["n_models"], max_features=best_params["max_features"],
        grace_period=best_params["grace_period"], delta=best_params["delta"],
        tau=best_params["tau"], leaf_prediction=best_params["leaf_prediction"],
        max_hold_bars=best_params["max_hold_bars"], feature_whitelist=top_features, seed=42
    )

    slow_model = AdaptiveHFTModel(
        "Slow-Opt",
        tp_bps=best_params["tp_bps"]*2, sl_bps=best_params["sl_bps"]*2, fee_bps=3.0,
        n_models=15, max_features=best_params["max_features"],
        max_hold_bars=best_params["max_hold_bars"] * 2, feature_whitelist=top_features, seed=42
    )

    shadow = ShadowLabeler(tp_bps=best_params["tp_bps"], sl_bps=best_params["sl_bps"], fee_bps=3.0)

    bar_count = warmup_engine_loud(stream_data, symbol_state, micro_model, slow_model, shadow)

    url = f"wss://fstream.binance.com/stream?streams={symbol}@trade/{symbol}@ticker/{symbol}@bookTicker"
    print(f"\n[🚀 SYSTEM] Transitioning to Live WebSocket at {url}...")
    print(f"\n[💎 MAGIC NUMBERS LOCKED IN FOR LIVE TRADING]")
    print(json.dumps(best_params, indent=4))

    diagnostic_counter = 0
    first_ticker_received = False
    snapshot = MarketSnapshot()

    try:
        async with websockets.connect(url) as ws:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                stream_name = data.get('stream', '')
                payload = data.get('data', {})

                if '@bookTicker' in stream_name:
                    current_bid = float(payload['b'])
                    current_ask = float(payload['a'])
                    ts = payload.get('E', payload.get('u', time.time() * 1000))
                    snapshot.update_book(current_bid, current_ask, ts)

                elif '@ticker' in stream_name:
                    if 'v' in payload:
                        daily_volume = float(payload['v'])
                        dynamic_threshold = max(base_vol_target, daily_volume / 85000.0)
                        symbol_state.update_dynamic_threshold(dynamic_threshold)
                        if not first_ticker_received:
                            print(f"\n[⚙️ SYSTEM] Live Ticker Connected! Dynamic Vol Target: {dynamic_threshold:.1f}")
                            first_ticker_received = True

                elif '@trade' in stream_name:
                    price, qty = float(payload['p']), float(payload['q'])
                    is_buyer_maker, ts = payload['m'], payload['T']

                    if price <= 0.01 or qty <= 0.0: continue

                    latest_tick = {"high": price, "low": price, "close": price}
                    if bar_count > 0:
                        micro_model.process_delayed_labels(bar_count, latest_tick)
                        slow_model.process_delayed_labels(bar_count, latest_tick)

                    if symbol_state.process_tick(price, qty, is_buyer_maker, ts, snapshot.bid, snapshot.ask):
                        bar_count += 1
                        latest_bar = symbol_state.completed_bars[-1]
                        shadow.process_new_bar(bar_count, latest_bar)

                        features = micro_model.extract_features(symbol_state)
                        if features:
                            micro_pred, micro_probas = micro_model.predict(features)
                            slow_pred, slow_probas = slow_model.predict(features)

                            micro_model.register_state(bar_count, price, features, micro_pred, micro_probas)
                            slow_model.register_state(bar_count, price, features, slow_pred, slow_probas)

                            print(micro_model.format_state_readout(bar_count, price, features, slow_model))

                            diagnostic_counter += 1
                            if diagnostic_counter >= 50:
                                diagnostic_counter = 0
                                print("\n" + micro_model.format_ev_table(), end="")

    except Exception as e:
        import traceback
        print(f"\n[🚨 FATAL ERROR] Websocket connection dropped: {e}")
        traceback.print_exc()
    finally:
        print("\n[🛑 SYSTEM] Engine Shutting Down...")

# ─── 5. EXECUTION ────────────────────────────────────────────────────────────
for task in asyncio.all_tasks():
    if task.get_coro().__name__ in ['live_trading_engine']:
        task.cancel()

try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(live_trading_engine())
except KeyboardInterrupt:
    print("\nEngine stopped manually.")
