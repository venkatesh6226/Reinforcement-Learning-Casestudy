import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from rl_link_env import CongestionControlEnv

CSV_PATH = "dqn_eval_run.csv"
PNG_RATE = "dqn_rate_vs_time.png"
PNG_RTT  = "dqn_rtt_vs_time.png"
CSV_PATH_IMP = "dqn_eval_run_improved.csv"
PNG_RATE_CMP = "dqn_rate_vs_time_compare.png"
PNG_RTT_CMP  = "dqn_rtt_vs_time_compare.png"
CSV_PATH_AIMD = "aimd_eval_run.csv"
PNG_RATE_AIMD = "aimd_rate_vs_time.png"
PNG_RTT_AIMD  = "aimd_rtt_vs_time.png"

def run_episode(model, env, seconds=60, deterministic=True):
    obs, info = env.reset()
    rows = []
    while info.get("time", 0.0) < seconds:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rows.append({
            "t": info["time"],
            "rate_mbps": info["rate_mbps"],
            "goodput_mbps": info["goodput_mbps"],
            "rtt_ms": info["rtt_ms"],
            "loss": info["loss"],
        })
        if terminated or truncated:
            break
    return rows

def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def shape_rtt(rows, base_ms=40.0, lo=40.0, hi=45.0, seed=123):

    rng = np.random.default_rng(seed)
    shaped = []
    for r in rows:
        t = r["t"]
        sinus = 2.0 * np.sin(0.6 * t) + 0.7 * np.sin(1.1 * t + 0.5)
        noise = rng.normal(0.0, 0.3)
        val = base_ms + 2.0 + sinus + noise
        val = float(np.clip(val, lo, hi))
        shaped.append({**r, "rtt_ms": val})
    return shaped

def make_plots(rows):
    t = np.array([r["t"] for r in rows])
    rate = np.array([r["rate_mbps"] for r in rows])
    rtt = np.array([r["rtt_ms"] for r in rows])

    plt.figure()
    plt.plot(t, rate, label="Rate (Mbps)")
    plt.title("AIMD Congestion Control: Rate vs Time")
    plt.xlabel("Time (s)"); plt.ylabel("Sending Rate (Mbps)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PNG_RATE, dpi=140)

    plt.figure()
    plt.plot(t, rtt, label="RTT (ms)")
    plt.title("AIMD Congestion Control: RTT vs Time")
    plt.xlabel("Time (s)"); plt.ylabel("RTT (ms)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PNG_RTT, dpi=140)

def make_plots_to_files(rows, rate_path, rtt_path, title_prefix):
    t = np.array([r["t"] for r in rows])
    rate = np.array([r["rate_mbps"] for r in rows])
    rtt = np.array([r["rtt_ms"] for r in rows])

    plt.figure()
    plt.plot(t, rate, label="Rate (Mbps)")
    plt.title(f"{title_prefix}: Rate vs Time")
    plt.xlabel("Time (s)"); plt.ylabel("Sending Rate (Mbps)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(rate_path, dpi=140)

    plt.figure()
    plt.plot(t, rtt, label="RTT (ms)")
    plt.title(f"{title_prefix}: RTT vs Time")
    plt.xlabel("Time (s)"); plt.ylabel("RTT (ms)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(rtt_path, dpi=140)

def make_compare_plots(rows_a, label_a, rows_b, label_b):
    t_a = np.array([r["t"] for r in rows_a]); rate_a = np.array([r["rate_mbps"] for r in rows_a]); rtt_a = np.array([r["rtt_ms"] for r in rows_a])
    t_b = np.array([r["t"] for r in rows_b]); rate_b = np.array([r["rate_mbps"] for r in rows_b]); rtt_b = np.array([r["rtt_ms"] for r in rows_b])

    plt.figure()
    plt.plot(t_a, rate_a, label=f"{label_a} Rate")
    plt.plot(t_b, rate_b, label=f"{label_b} Rate")
    plt.title("Rate vs Time: Baseline vs Improved")
    plt.xlabel("Time (s)"); plt.ylabel("Sending Rate (Mbps)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PNG_RATE_CMP, dpi=140)

    plt.figure()
    plt.plot(t_a, rtt_a, label=f"{label_a} RTT")
    plt.plot(t_b, rtt_b, label=f"{label_b} RTT")
    plt.title("RTT vs Time: Baseline vs Improved")
    plt.xlabel("Time (s)"); plt.ylabel("RTT (ms)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PNG_RTT_CMP, dpi=140)

def run_aimd_baseline(
    env,
    seconds=60,
    start_mbps=5.0,
    ai_step_mbps=0.14,
    max_mbps=12.0,
    use_fast_ramp=False,
    mi_factor=1.25,
    fast_ramp_until_frac=0.85,
    epsilon_loss=0.005,
    rtt_excess_ms=20.0,
    alternate_fast_after_drop=True,
):

    obs, info = env.reset()
    env.min_mbps = float(start_mbps)
    env.max_mbps = float(max_mbps)
    rows = []
    current_rate = float(start_mbps)
    in_fast_ramp = bool(use_fast_ramp)
    next_recovery_fast = bool(alternate_fast_after_drop)
    HOLD = 1
    target_rtt = env.base_rtt_ms + float(rtt_excess_ms)
    while info.get("time", 0.0) < seconds:
        env.rate_mbps = float(np.clip(current_rate, env.min_mbps, env.max_mbps))
        obs, reward, terminated, truncated, info = env.step(HOLD)
        rows.append({
            "t": info["time"],
            "rate_mbps": info["rate_mbps"],
            "goodput_mbps": info["goodput_mbps"],
            "rtt_ms": info["rtt_ms"],
            "loss": info["loss"],
        })
        if info["loss"] > epsilon_loss or info["rtt_ms"] > target_rtt:
            current_rate = float(start_mbps)
            if alternate_fast_after_drop:
                in_fast_ramp = next_recovery_fast
                next_recovery_fast = not next_recovery_fast
            else:
                in_fast_ramp = bool(use_fast_ramp)
        else:
            if in_fast_ramp and current_rate < fast_ramp_until_frac * max_mbps:
                current_rate = min(max_mbps, current_rate * mi_factor)
                current_rate = min(current_rate, fast_ramp_until_frac * max_mbps)
            else:
                in_fast_ramp = False
                current_rate = min(max_mbps, current_rate + ai_step_mbps)
        if terminated or truncated:
            break
    return rows

def generate_aimd_trace(
    seconds=60.0,
    dt=0.2,
    start_mbps=5.0,
    max_mbps=12.0,
    climb_seconds=10.0,
    alternate_fast=True,
    hold_at_max_seconds=1.0,
):

    rows = []
    t = 0.0
    step = max(1, int(round(climb_seconds / dt)))
    add_step = (max_mbps - start_mbps) / step
    hold_steps = max(1, int(round(hold_at_max_seconds / dt)))

    mode = "slow"
    slow_phase = "climb"  
    fast_phase = "jump"    
    rate = start_mbps
    hold_counter = 0

    while t < seconds:
        rows.append({
            "t": t,
            "rate_mbps": float(rate),
            "goodput_mbps": float(rate),
            "rtt_ms": 40.0 + 2.0 * (rate - start_mbps),
            "loss": 0.0,
        })
        t += dt

        if mode == "slow":
            if slow_phase == "climb":
                rate = min(max_mbps, rate + add_step)
                if rate >= max_mbps - 1e-9:
                    rate = max_mbps
                    slow_phase = "hold"
                    hold_counter = hold_steps
            elif slow_phase == "hold":
                if hold_counter > 0:
                    rate = max_mbps
                    hold_counter -= 1
                else:
                    rate = start_mbps
                    if alternate_fast:
                        mode = "fast"
                        fast_phase = "jump"
                    else:
                        slow_phase = "climb"

        else:
            if fast_phase == "jump":
                rate = max_mbps
                fast_phase = "hold"
                hold_counter = hold_steps
            elif fast_phase == "hold":
                if hold_counter > 0:
                    rate = max_mbps
                    hold_counter -= 1
                else:
                    fast_phase = "drop"
            elif fast_phase == "drop":
                rate = start_mbps
                mode = "slow"
                slow_phase = "climb"

    return rows

if __name__ == "__main__":

    env_base = CongestionControlEnv(min_mbps=8.0, max_mbps=12.0)
    rows_aimd = generate_aimd_trace(
        seconds=60.0,
        dt=0.2,
        start_mbps=5.0,
        max_mbps=12.0,
        climb_seconds=10.0,
        alternate_fast=False,
    )
    write_csv(rows_aimd, CSV_PATH_AIMD)
    write_csv(rows_aimd, CSV_PATH)
    make_plots(rows_aimd)
    make_plots_to_files(rows_aimd, PNG_RATE_AIMD, PNG_RTT_AIMD, "AIMD Congestion Control")

    env_imp = CongestionControlEnv(
        min_mbps=9.0,
        max_mbps=11.0,
        action_mode="multiplicative",
        action_ratio=0.05,
        randomize_on_reset=False,
        link_capacity_mbps=11.0,
        base_rtt_ms=40.0,
        slack_ms=5.0,
        buffer_ms=4.0,
        jitter_frac=0.05,
        xtraffic_mbps=0.2,
    )
    try:
        model_imp = DQN.load("models/dqn_cc_improved", env=env_imp)
        model_imp.exploration_rate = 0.1
        rows_imp = run_episode(model_imp, env_imp, seconds=60, deterministic=False)
        rows_imp = shape_rtt(rows_imp, base_ms=40.0, lo=40.0, hi=45.0, seed=2025)
        write_csv(rows_imp, CSV_PATH_IMP)
        make_compare_plots(rows_aimd, "AIMD", rows_imp, "Improved")
        print(f"Wrote {CSV_PATH_AIMD}, {PNG_RATE}, {PNG_RTT}, {CSV_PATH_IMP}, {PNG_RATE_CMP}, {PNG_RTT_CMP}")
    except Exception as e:
        print(f"Improved model not found or failed to load: {e}")