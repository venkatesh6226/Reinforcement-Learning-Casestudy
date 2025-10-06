import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from rl_link_env import CongestionControlEnv
from torch import nn

def make_env():
    return CongestionControlEnv(
        link_capacity_mbps=10.0,
        base_rtt_ms=40.0,
        slack_ms=30.0,           
        ctrl_ms=200,
        min_mbps=9.0,
        max_mbps=11.0,

        action_mode="multiplicative",
        action_ratio=0.05,        
        action_delta_mbps=0.25,
        buffer_ms=100.0,
        jitter_frac=0.03,
        loss_floor=0.0,
    
        randomize_on_reset=False,
        rand_capacity_mbps=(9.5, 10.5),
        rand_base_rtt_ms=(35.0, 45.0),
        rand_buffer_ms=(80.0, 150.0),
        rand_xtraffic_mbps=(0.0, 0.5),
        rand_jitter_frac=(0.0, 0.03),
        w_throughput=3.0,
        w_delay=4.0,          
        w_loss=8.0,          
        w_lowband=1.5,  
        w_highband=6.0,
        band_bonus=2.0,    
        band_center_mbps=9.7,
        band_width_mbps=0.5,
    )

if __name__ == "__main__":
    env = make_env()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.25,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="./tb_dqn_cc",
        seed=11,
        policy_kwargs=dict(
            net_arch=[128, 128],
            activation_fn=nn.ReLU,
        ),
    )
    os.makedirs("models", exist_ok=True)
    ckpt = CheckpointCallback(save_freq=20_000, save_path="models", name_prefix="dqn_cc_improved")

    model.learn(total_timesteps=120_000, callback=ckpt, progress_bar=True)
    model.save("models/dqn_cc_improved")

    env_base = CongestionControlEnv(min_mbps=8.0, max_mbps=12.0)
    model_base = DQN(
        "MlpPolicy",
        env_base,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.20,
        exploration_final_eps=0.02,
        verbose=0,
        tensorboard_log="./tb_dqn_cc",
        seed=11,
    )
    model_base.learn(total_timesteps=60_000, progress_bar=False)
    model_base.save("models/dqn_cc_baseline_new")
    env_base.close()
    env.close()