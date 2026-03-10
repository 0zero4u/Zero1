# ==========================================
# 2. ADVANCED HFT ADAPTIVE MODEL (WINSORIZATION & STATIONARY FPCA UPGRADE)
# ==========================================
import math
import numpy as np
from collections import deque
from river.forest import ARFClassifier
from river import compose, preprocessing, metrics, stats, tree, utils, base

class StreamingWinsorizer(base.Transformer):
    """
    Dynamically bounds extreme outliers to +/- Z standard deviations
    using streaming Welford's algorithm to protect the StandardScaler.
    """
    def __init__(self, clip_z_score=4.0):
        self.clip_z_score = clip_z_score
        self.feature_stats = {}

    def transform_one(self, x):
        x_clipped = {}
        for feat_name, val in x.items():
            if feat_name not in self.feature_stats:
                self.feature_stats[feat_name] = {'mean': stats.Mean(), 'var': stats.Var()}

            f_stats = self.feature_stats[feat_name]

            current_mean = f_stats['mean'].get()
            current_std = math.sqrt(f_stats['var'].get() or 0.0)

            if current_std > 0:
                upper_bound = current_mean + (self.clip_z_score * current_std)
                lower_bound = current_mean - (self.clip_z_score * current_std)
                clipped_val = max(lower_bound, min(upper_bound, val))
            else:
                clipped_val = val

            x_clipped[feat_name] = clipped_val

            # Update stats with the clipped value so extreme anomalies don't poison variance
            f_stats['mean'].update(clipped_val)
            f_stats['var'].update(clipped_val)

        return x_clipped

class EWMACohenKappa:
    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.p_o = stats.EWMean(alpha)
        self.p_true = { -1: stats.EWMean(alpha), 0: stats.EWMean(alpha), 1: stats.EWMean(alpha) }
        self.p_pred = { -1: stats.EWMean(alpha), 0: stats.EWMean(alpha), 1: stats.EWMean(alpha) }

    def update(self, y_true, y_pred):
        self.p_o.update(1.0 if y_true == y_pred else 0.0)
        for k in [-1, 0, 1]:
            self.p_true[k].update(1.0 if y_true == k else 0.0)
            self.p_pred[k].update(1.0 if y_pred == k else 0.0)

    def get(self):
        po = self.p_o.get()
        pe = sum(self.p_true[k].get() * self.p_pred[k].get() for k in [-1, 0, 1])
        if pe < 1.0:
            return (po - pe) / (1.0 - pe)
        return 0.0

class HFTAdaptiveModel:
    def __init__(self, name: str, horizon_bars: int = 30, tp_bps=15.0, sl_bps=-15.0, fee_bps=5.0):
        self.name = name
        self.horizon_bars = horizon_bars
        self.tp_bps = tp_bps
        self.sl_bps = sl_bps
        self.fee_bps = fee_bps

        # Protects pipeline against anomalies natively
        self.model = compose.Pipeline(
            StreamingWinsorizer(clip_z_score=4.0),
            preprocessing.StandardScaler(),
            ARFClassifier(
                n_models=20,
                seed=42,
                grace_period=200,
                leaf_prediction="mc",
                split_criterion="hellinger",
            )
        )

        self.rolling_cm = utils.Rolling(metrics.ConfusionMatrix(), window_size=1000)
        self.metric_acc = metrics.Accuracy(self.rolling_cm)
        self.metric_macro_f1 = metrics.MacroF1(self.rolling_cm)
        self.metric_kappa = metrics.CohenKappa(self.rolling_cm)

        self.vips = ["volatility_3bar", "volatility_30bar", "range_expansion"]

        # Removed 'fpca_3' to prevent absolute price leakage
        self.all_scouts = [
            "tick_imbalance", "z_obi_momentum", "volatility_30bar", "intensity_ratio",
            "intra_volatility", "integrated_obi", "spread_crossing_return",
            "z_obi_mom_1", "z_obi_mom_3", "z_obi_mom_10",
            "fpca_1", "fpca_2", "time_to_fill_zscore"
        ]

        self.scouts = {}
        self.scout_metrics = {}
        self.active_scouts = []

        self.gate_update_counter = 0

        self.rolling_preds = deque(maxlen=500)
        self.rolling_labels = deque(maxlen=500)

        self.pending_samples = {}
        self.buckets = {
            "Long": {f"{p:.2f}-{p+0.05:.2f}": {"trades": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net_bps": 0.0} for p in np.arange(0.50, 1.00, 0.05)},
            "Short": {f"{p:.2f}-{p+0.05:.2f}": {"trades": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net_bps": 0.0} for p in np.arange(0.50, 1.00, 0.05)}
        }

        self.last_features = {}
        self.last_prediction = None

    def extract_features(self, symbol_state):
        if len(symbol_state.completed_bars) <= 10: return None

        latest = symbol_state.completed_bars[-1]
        bars = list(symbol_state.completed_bars)
        features = {"z_obi": latest["z_obi"]}

        z_intensity_ghost = latest["z_intensity"]
        features["z_obi_momentum"] = latest["z_obi"] - bars[-3]["z_obi"]
        features["z_intensity_momentum"] = z_intensity_ghost - bars[-3]["z_intensity"]

        closes = [b["close"] for b in bars]
        def calc_volatility(lookback):
            if len(closes) < 2: return 0.0
            window = closes[-lookback:] if lookback > 0 else closes
            log_returns = [math.log(window[i]/window[i-1]) for i in range(1, len(window))]
            return np.std(log_returns) * 10000 if len(log_returns) > 0 else 0.0

        features["volatility_3bar"] = calc_volatility(3)
        features["volatility_30bar"] = calc_volatility(30)

        current_range = (latest["high"] - latest["low"]) / latest["close"] if latest["close" ] > 0 else 0.0
        prev_range = (bars[-2]["high"] - bars[-2]["low"]) / bars[-2]["close"] if bars[-2]["close"] > 0 else 0.0
        features["range_expansion"] = (current_range - prev_range) * 10000

        features["volatility_50bar"] = calc_volatility(50)
        features["tick_acceleration"] = (bars[-1]["ticks"] - bars[-3]["ticks"]) - (bars[-3]["ticks"] - bars[-5]["ticks"])
        features["obi_x_intensity"] = features["z_obi"] * z_intensity_ghost
        features["intensity_x_vol"] = z_intensity_ghost * features["volatility_3bar"]
        features["whale_concentration_ratio"] = latest.get("whale_concentration_ratio", 0.0)
        features["time_to_fill_zscore"] = latest.get("z_duration", 0.0)

        vol_ratio = features["volatility_3bar"] / (features["volatility_30bar"] + 1e-9)
        dynamic_alpha = max(0.2, min(0.5, 0.2 * vol_ratio))

        vol_bars = bars[-10:]
        abs_imbalance = sum(abs(b["buy_vol"] - b["sell_vol"]) for b in vol_bars)
        if vol_bars:
            ema_vol = vol_bars[0]["buy_vol"] + vol_bars[0]["sell_vol"]
            for b in vol_bars[1:]:
                b_total = b["buy_vol"] + b["sell_vol"]
                ema_vol = dynamic_alpha * b_total + (1.0 - dynamic_alpha) * ema_vol
            smoothed_total_vol = ema_vol * len(vol_bars)
        else:
            smoothed_total_vol = 0.0

        features["vpin_toxicity"] = abs_imbalance / smoothed_total_vol if smoothed_total_vol > 0 else 0.0

        features["intra_volatility"] = latest.get("intra_volatility", 0.0)
        features["integrated_obi"] = latest.get("integrated_obi", 0.0)
        features["spread_crossing_return"] = latest.get("spread_crossing_return", 0.0)

        features["z_obi_mom_1"] = latest.get("z_obi", 0.0) - bars[-2]["z_obi"] if len(bars) >= 2 else 0.0
        features["z_obi_mom_3"] = latest.get("z_obi", 0.0) - bars[-4]["z_obi"] if len(bars) >= 4 else 0.0
        features["z_obi_mom_10"] = latest.get("z_obi", 0.0) - bars[-11]["z_obi"] if len(bars) >= 11 else 0.0

        # [UPGRADED FPCA]: Normalized for Stationarity + Price Leak Removed
        if len(bars) >= 10:
            mid_prices = [(b["high"] + b["low"] + b["close"]) / 3.0 for b in bars[-10:]]

            # Normalize prices to Basis Points (BPS) from the start of the window
            base_px = mid_prices[0]
            if base_px > 0:
                norm_px = [((p - base_px) / base_px) * 10000.0 for p in mid_prices]
            else:
                norm_px = [0.0] * len(mid_prices)

            x = np.arange(len(norm_px))
            poly = np.polyfit(x, norm_px, 2)

            # Retain only Acceleration (a) and Velocity (b).
            features["fpca_1"] = poly[0]  # Acceleration in BPS
            features["fpca_2"] = poly[1]  # Velocity in BPS
        else:
            features["fpca_1"], features["fpca_2"] = 0.0, 0.0

        return features

    def update_scout_gate(self):
        ranked_scouts = sorted(self.scout_metrics.items(), key=lambda x: x[1].get(), reverse=True)
        new_roster = []

        for scout_name, kappa_metric in ranked_scouts:
            if scout_name in self.vips: continue
            new_roster.append(scout_name)
            if len(new_roster) >= 7:
                break

        self.active_scouts = new_roster
        self.gate_update_counter = 0

    def check_gate_trigger(self):
        if len(self.rolling_labels) < 50:
            self.gate_update_counter += 1
            if self.gate_update_counter >= 20:
                self.update_scout_gate()
            return

        active_non_vip = [s for s in self.active_scouts if s not in self.vips and s in self.scout_metrics]
        benched = [s for s in self.scouts.keys() if s not in self.active_scouts and s not in self.vips and s in self.scout_metrics]

        if len(self.active_scouts) < 7:
            self.update_scout_gate()
            return

        if not active_non_vip or not benched:
            return

        worst_active_kappa = min([self.scout_metrics[s].get() for s in active_non_vip])
        best_benched_kappa = max([self.scout_metrics[s].get() for s in benched])

        HYSTERESIS_MARGIN = 0.02
        if best_benched_kappa > (worst_active_kappa + HYSTERESIS_MARGIN):
            self.update_scout_gate()

    def get_main_features(self, full_features):
        main_features = {k: v for k, v in full_features.items() if k in self.vips or k in self.active_scouts}
        return main_features if main_features else full_features

    def predict_and_store(self, features, current_price, current_bar_idx):
        if not features: return 0.0, {"-1": 0.33, "0": 0.34, "1": 0.33}

        self.check_gate_trigger()

        main_features = self.get_main_features(features)

        probs = self.model.predict_proba_one(main_features)
        pred = self.model.predict_one(main_features) or 0

        self.rolling_preds.append(pred)

        self.pending_samples[current_bar_idx] = {
            "features": main_features,
            "full_features": features,
            "entry_price": current_price,
            "y_pred": pred,
            "probs": probs
        }
        return pred, probs

    def generate_mirror_diagnostic(self, current_features, current_pred, bar_count):
        log_output = ""

        if self.last_features and self.last_prediction is not None:
            pred_changed = current_pred != self.last_prediction

            anomalies = {}
            for k, v in current_features.items():
                prev_v = self.last_features.get(k, 0.0)
                delta = v - prev_v
                if abs(delta) > 50.0:
                    anomalies[k] = delta

            if pred_changed or anomalies:
                log_output += f"\n🔍 [MIRROR DIAGNOSTIC - {self.name}] Triggered at Bar #{bar_count}\n"
                if pred_changed:
                    log_output += f"   ├─ Prediction Flip Detected: {self.last_prediction} -> {current_pred}\n"

                if anomalies:
                    log_output += f"   ├─ Anomalies Detected (Δ > 50.0):\n"
                    for k, d in anomalies.items():
                        prev_val = self.last_features.get(k, 0.0)
                        log_output += f"   │    └─ {k}: {d:+.4f} (Prev: {prev_val:.4f} -> Curr: {current_features[k]:.4f})\n"

                log_output += f"   └─ VIP & Active Scout Feature Heatmap:\n"
                keys_to_show = self.vips + self.active_scouts
                for k in keys_to_show:
                    if k in current_features:
                        curr_v = current_features[k]
                        prev_v = self.last_features.get(k, 0.0)
                        diff = curr_v - prev_v
                        log_output += f"        {k:<25}: {prev_v:>10.4f} -> {curr_v:>10.4f} | Δ: {diff:+.4f}\n"

        self.last_features = current_features.copy()
        self.last_prediction = current_pred

        return log_output

    def process_delayed_labels_tbm(self, current_bar_idx, latest_bar):
        keys_to_remove = []
        for start_idx, sample in self.pending_samples.items():
            entry = sample["entry_price"]
            high_bps = math.log(latest_bar["high"] / entry) * 10000.0
            low_bps = math.log(latest_bar["low"] / entry) * 10000.0
            label, bps_realized = None, 0.0

            hit_long_tp = high_bps >= self.tp_bps
            hit_short_tp = low_bps <= self.sl_bps

            if hit_long_tp and hit_short_tp:
                resolved = False
                for price in latest_bar.get("prices", []):
                    tick_bps = math.log(price / entry) * 10000.0
                    if tick_bps >= self.tp_bps:
                        label, bps_realized, resolved = 1, self.tp_bps, True
                        break
                    elif tick_bps <= self.sl_bps:
                        label, bps_realized, resolved = -1, self.sl_bps, True
                        break
                if not resolved:
                    label, bps_realized = (1, self.tp_bps) if high_bps > abs(low_bps) else (-1, self.sl_bps)
            elif hit_long_tp:
                label, bps_realized = 1, self.tp_bps
            elif hit_short_tp:
                label, bps_realized = -1, self.sl_bps

            if label is None and (current_bar_idx - start_idx) >= self.horizon_bars:
                close_bps = math.log(latest_bar["close"] / entry) * 10000.0
                if close_bps > self.fee_bps: label, bps_realized = 1, close_bps
                elif close_bps < -self.fee_bps: label, bps_realized = -1, close_bps
                else: label, bps_realized = 0, close_bps

            if label is not None:
                self.rolling_labels.append(label)

                full_features = sample["full_features"]
                for f_name, f_value in full_features.items():
                    if f_name not in self.scouts:
                        self.scouts[f_name] = tree.HoeffdingTreeClassifier(
                            split_criterion="hellinger",
                            leaf_prediction="mc",
                            grace_period=200
                        )
                        self.scout_metrics[f_name] = EWMACohenKappa(alpha=0.02)

                    s_pred = self.scouts[f_name].predict_one({f_name: f_value})
                    self.scout_metrics[f_name].update(label, s_pred)
                    self.scouts[f_name].learn_one({f_name: f_value}, label)

                self.metric_acc.update(label, sample["y_pred"])
                self.metric_macro_f1.update(label, sample["y_pred"])
                self.metric_kappa.update(label, sample["y_pred"])

                self.model.learn_one(sample["features"], label)

                pred_val = sample["y_pred"]
                probs = sample["probs"]

                if pred_val in [1, -1] and probs:
                    side = "Long" if pred_val == 1 else "Short"
                    conf = probs.get(pred_val, 0)
                    for b_key in self.buckets[side].keys():
                        lower, upper = map(float, b_key.split('-'))
                        if lower <= conf < upper:
                            b = self.buckets[side][b_key]
                            b["trades"] += 1

                            if pred_val == 1:
                                trade_pnl = bps_realized - self.fee_bps
                            else:
                                trade_pnl = -bps_realized - self.fee_bps

                            is_win = trade_pnl > 0

                            if is_win:
                                b["wins"] += 1
                                b["gross_profit"] += trade_pnl
                            else:
                                b["gross_loss"] += abs(trade_pnl)

                            b["net_bps"] += trade_pnl
                            break
                keys_to_remove.append(start_idx)

        for k in keys_to_remove: del self.pending_samples[k]

    def format_kappa_gate_log(self):
        ranked = sorted(self.scout_metrics.items(), key=lambda x: x[1].get(), reverse=True)
        log = "MDI (EWMA KAPPA) GATE] Top Performing Non-Linear Scouts:\n"
        for i, (name, metric) in enumerate(ranked[:10]):
            status = "✅ Active" if name in self.active_scouts else "❌ Bench"
            if name in self.vips: status = "👑 VIP Anchor"
            log += f"    │  ├─ #{i+1} {name}: {metric.get():.4f} [{status}]\n"
        return log

    def format_regime_log(self):
        if len(self.rolling_labels) < 50: return "ROLLING REGIME] Awaiting Label Resolution...\n"

        preds = list(self.rolling_preds)
        labels = list(self.rolling_labels)

        p_l = preds.count(1) / len(preds) if preds else 0
        p_s = preds.count(-1) / len(preds) if preds else 0
        p_n = preds.count(0) / len(preds) if preds else 0

        a_l = labels.count(1) / len(labels) if labels else 0
        a_s = labels.count(-1) / len(labels) if labels else 0
        a_n = labels.count(0) / len(labels) if labels else 0

        ratio = (a_l + 1e-9) / (a_s + 1e-9)

        log = f"ROLLING {len(preds)}-BAR REGIME] Imbalance Ratio: {ratio:.2f}\n"
        log += f"    ├─Actual Tape: L {a_l:.1%} | S {a_s:.1%} | N {a_n:.1%}\n"
        log += f"    ├─ AI Preds  : L {p_l:.1%} | S {p_s:.1%} | N {p_n:.1%}\n"

        health = "✅ HEALTHY" if abs(p_l - a_l) < 0.2 else "⚠️ DIVERGING"
        log += f"    └─ {health}: AI predictions tracking market structure.\n"
        return log

    def format_ev_table(self):
        output = f"║ {self.name.upper()} AI CORE: NET EV & PROFIT FACTOR MATRIX ║\n"
        output += f"╟─ Acc: {self.metric_acc.get():.2%} | MacF1: {self.metric_macro_f1.get():.4f} | Kappa: {self.metric_kappa.get():.4f}\n"

        vips_str = ", ".join(self.vips)
        scouts_str = ", ".join(self.active_scouts) if self.active_scouts else "Awaiting Matrix Gate..."
        output += f"╟─ [VIP Anchors] : {vips_str}\n"
        output += f"╟─ [Active Scouts]: {scouts_str}\n"

        output += "║" + "─"*73 + "║\n"
        for side in ["Long", "Short"]:
            output += f"╟─[{side.upper()} CONFIDENCE BUCKETS]\n"
            active_buckets = {k: v for k, v in self.buckets[side].items() if v["trades"] > 0}
            if not active_buckets:
                output += "   ├─ Awaiting data...\n"
                continue

            for b_key, stats in sorted(active_buckets.items(), reverse=True):
                trades = stats["trades"]
                net_ev = stats["net_bps"] / trades
                win_rate = (stats["wins"] / trades) * 100
                pf = stats["gross_profit"] / stats["gross_loss"] if stats["gross_loss"] > 0 else 99.99
                status = "🔥 PRINTING" if net_ev > 0 and pf > 1.2 else "⚠️ BLEEDING" if net_ev < 0 else "⚖️ CHOPPING"
                output += f"   ├─ P({b_key}) | Trades: {trades:03d} | WR: {win_rate:05.2f}% | EV: {net_ev:+06.2f}bps | PF: {pf:05.2f} | {status}\n"
        return output
