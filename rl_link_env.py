import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CongestionControlEnv(gym.Env):


    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        link_capacity_mbps: float = 10.0,
        base_rtt_ms: float = 40.0,
        slack_ms: float = 200.0,
        ctrl_ms: int = 200,
        min_mbps: float = 5.0,
        max_mbps: float = 12.0,
        action_delta_mbps: float = 0.15,  
        action_mode: str = "additive",   
        action_ratio: float = 0.05,       
        buffer_ms: float = 100.0,
        xtraffic_mbps: float = 0.0,
        jitter_frac: float = 0.03,        
        loss_floor: float = 0.0,

        randomize_on_reset: bool = False,
        rand_capacity_mbps: tuple = (8.0, 12.0),
        rand_base_rtt_ms: tuple = (20.0, 80.0),
        rand_buffer_ms: tuple = (50.0, 250.0),
        rand_xtraffic_mbps: tuple = (0.0, 2.0),
        rand_jitter_frac: tuple = (0.0, 0.05),

 
        w_throughput: float = 1.8,
        w_delay: float = 2.0,
        w_loss: float = 4.0,
        w_lowband: float = 0.8,           
        w_highband: float = 1.0,          
        band_bonus: float = 0.6,       
        explore_bonus: float = 0.02,      
        band_center_mbps: float = 10.0,
        band_width_mbps: float = 2.0      
    ):
        super().__init__()

        self.cap_mbps = float(link_capacity_mbps)
        self.base_rtt_ms = float(base_rtt_ms)
        self.slack_ms = float(slack_ms)
        self.dt = float(ctrl_ms) / 1000.0
        self.min_mbps = float(min_mbps)
        self.max_mbps = float(max_mbps)
        self.delta = float(action_delta_mbps)
        self.action_mode = str(action_mode)
        self.action_ratio = float(action_ratio)
        self.buffer_ms = float(buffer_ms)
        self.xtraffic_mbps = float(xtraffic_mbps)
        self.jitter_frac = float(jitter_frac)
        self.loss_floor = float(loss_floor)

        self.randomize_on_reset = bool(randomize_on_reset)
        self.rand_capacity_mbps = tuple(rand_capacity_mbps)
        self.rand_base_rtt_ms = tuple(rand_base_rtt_ms)
        self.rand_buffer_ms = tuple(rand_buffer_ms)
        self.rand_xtraffic_mbps = tuple(rand_xtraffic_mbps)
        self.rand_jitter_frac = tuple(rand_jitter_frac)

        self.w_t = float(w_throughput)
        self.w_d = float(w_delay)
        self.w_l = float(w_loss)
        self.w_lowband = float(w_lowband)
        self.w_highband = float(w_highband)
        self.band_bonus = float(band_bonus)
        self.explore_bonus = float(explore_bonus)
        self.band_center = float(band_center_mbps)
        self.band_halfw = float(band_width_mbps)

        self.cap_Bps_nom = self.cap_mbps * 1e6 / 8.0
        self.buf_bytes = (self.buffer_ms / 1000.0) * self.cap_Bps_nom

        self.action_space = spaces.Discrete(3)  # dec/hold/inc
        high = np.array([2.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, shape=(3,), dtype=np.float32)

        self.reset(seed=None)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Optional domain randomization
        if self.randomize_on_reset:
            self.cap_mbps = float(self.np_random.uniform(*self.rand_capacity_mbps))
            self.base_rtt_ms = float(self.np_random.uniform(*self.rand_base_rtt_ms))
            self.buffer_ms = float(self.np_random.uniform(*self.rand_buffer_ms))
            self.xtraffic_mbps = float(self.np_random.uniform(*self.rand_xtraffic_mbps))
            self.jitter_frac = float(self.np_random.uniform(*self.rand_jitter_frac))
            # recompute derived
            self.cap_Bps_nom = self.cap_mbps * 1e6 / 8.0
            self.buf_bytes = (self.buffer_ms / 1000.0) * self.cap_Bps_nom
        self.rate_mbps = self.min_mbps
        self.queue_bytes = 0.0
        self.last_loss_frac = 0.0
        self.t = 0.0
        return self._obs(), {}

    def step(self, action: int):

        if self.action_mode == "multiplicative":
            dec_factor = max(0.0, 1.0 - self.action_ratio)
            inc_factor = 1.0 + self.action_ratio
            if action == 0:
                self.rate_mbps = max(self.min_mbps, self.rate_mbps * dec_factor)
            elif action == 2:
                self.rate_mbps = min(self.max_mbps, self.rate_mbps * inc_factor)
        else:
            if action == 0:
                self.rate_mbps = max(self.min_mbps, self.rate_mbps - self.delta)
            elif action == 2:
                self.rate_mbps = min(self.max_mbps, self.rate_mbps + self.delta)

        cap_Bps = self.cap_Bps_nom * (1.0 + self.jitter_frac * self.np_random.uniform(-1.0, 1.0))
        cap_Bps = max(1.0, cap_Bps)
        x_Bps = max(0.0, self.xtraffic_mbps) * 1e6 / 8.0
        avail_Bps = max(1.0, cap_Bps - x_Bps)

        send_Bps = self.rate_mbps * 1e6 / 8.0
        offered = send_Bps * self.dt

        service = min(avail_Bps * self.dt, self.queue_bytes)
        self.queue_bytes -= service
        self.queue_bytes += offered

        drop = 0.0
        if self.queue_bytes > self.buf_bytes:
            drop = self.queue_bytes - self.buf_bytes
            self.queue_bytes = self.buf_bytes

        delivered = min(avail_Bps * self.dt, service + (offered - drop))
        delivered = max(0.0, delivered)

        loss_bytes = drop + offered * self.loss_floor * self.dt
        self.last_loss_frac = float(loss_bytes / max(1.0, offered))

        q_delay_s = self.queue_bytes / max(1.0, avail_Bps)
        rtt_ms = self.base_rtt_ms + 1000.0 * q_delay_s

        goodput_mbps = (delivered / self.dt) * 8.0 / 1e6
        norm_thr = goodput_mbps / max(1e-6, self.max_mbps)

        target_rtt = self.base_rtt_ms + self.slack_ms
        delay_excess = max(0.0, rtt_ms - target_rtt) / max(1.0, self.slack_ms)

        low_gap = max(0.0, 8.0 - self.rate_mbps)
        high_gap = max(0.0, self.rate_mbps - 12.0)

        bonus = 0.0
        if rtt_ms <= target_rtt:
            
            dist = abs(self.rate_mbps - self.band_center)
            bonus = max(0.0, 1.0 - dist / max(1e-6, self.band_halfw)) * self.band_bonus

        probe = self.explore_bonus if action != 1 else 0.0

        reward = (
            self.w_t * norm_thr
            - self.w_d * delay_excess
            - self.w_l * self.last_loss_frac
            - self.w_lowband * low_gap
            - self.w_highband * high_gap
            + bonus
            + probe
        )
        reward = float(np.clip(reward, -5.0, 5.0))

        self.t += self.dt
        obs = self._obs(rtt_ms=rtt_ms)
        info = {
            "time": self.t,
            "rate_mbps": float(self.rate_mbps),
            "goodput_mbps": float(goodput_mbps),
            "rtt_ms": float(rtt_ms),
            "loss": float(self.last_loss_frac),
            "queue_ms": 1000.0 * self.queue_bytes / self.cap_Bps_nom
        }
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info


    def _obs(self, rtt_ms=None):
        if rtt_ms is None:
            q_delay_s = self.queue_bytes / max(1.0, self.cap_Bps_nom)
            rtt_ms = self.base_rtt_ms + 1000.0 * q_delay_s
        return np.array([
            np.clip(self.rate_mbps / self.cap_mbps, 0.0, 2.0),
            np.clip(max(0.0, (rtt_ms - self.base_rtt_ms)) / max(1.0, self.slack_ms), 0.0, 10.0),
            np.clip(self.last_loss_frac, 0.0, 1.0),
        ], dtype=np.float32)

    def render(self):
        q_ms = self.queue_bytes / self.cap_Bps_nom * 1000.0
        print(f"t={self.t:.1f}s rate={self.rate_mbps:.2f} Mbps queue={q_ms:.1f} ms loss={self.last_loss_frac:.3f}")