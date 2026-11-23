"""Measure SB3 PPO and probe the cost of a batched Metal dynamics kernel."""

from argparse import ArgumentParser
from statistics import median
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from chauffeur_evader.chauffeur_classical import _path
from chauffeur_evader.chauffeur_ppo import ChauffeurPPOEnv
from chauffeur_evader.homicidal_chauffeur import GameConfig


class TimedPPO(PPO):
    def _measure(self, samples, function, *args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        if self.device.type == "mps":
            torch.mps.synchronize()
        samples.append(perf_counter() - start)
        return result

    def collect_rollouts(self, *args, **kwargs):
        return self._measure(self.rollout_times, super().collect_rollouts, *args, **kwargs)

    def train(self):
        self._measure(self.update_times, super().train)


def ppo_times(device, workers, steps):
    env = make_vec_env(
        ChauffeurPPOEnv,
        n_envs=workers,
        seed=17,
        vec_env_cls=SubprocVecEnv if workers > 1 else None,
        vec_env_kwargs={"start_method": "spawn"} if workers > 1 else None,
    )
    try:
        model = TimedPPO(
            "MlpPolicy",
            env,
            device=device,
            seed=17,
            n_steps=1024 // workers,
            batch_size=64,
            n_epochs=10,
            policy_kwargs={"net_arch": [32, 32]},
        )
        model.rollout_times, model.update_times = [], []
        assert model.device.type == device
        model.learn(steps + 1024)
        assert model.num_timesteps == steps + 1024
        return sum(model.rollout_times[1:]), sum(model.update_times[1:])
    finally:
        env.close()


def precision_probe():
    env = ChauffeurPPOEnv(GameConfig(dt=0.02, horizon=4.0))
    starts = [2.3, 2.3 + 1e-12]
    outcomes = []
    action = np.zeros(1, dtype=np.float32)
    for y in starts:
        env.reset(options={"state": (0, y)})
        while True:
            _, _, done, _, info = env.step(action)
            if done:
                outcomes.append(info["termination_reason"])
                break
    assert outcomes == ["capture", "time_limit"] and np.float32(starts[0]) == np.float32(starts[1])
    print(f"Same float32 start, different float64 outcomes: {outcomes}", flush=True)


def numpy_advance(inputs, output):
    x, y, phi, psi = inputs.T
    dt = np.float32(0.05)
    a = phi * dt
    s, c, h = np.sin(a), np.cos(a), np.sin(a / 2)
    dx = np.divide(2 * h * h, phi, out=np.zeros_like(phi), where=phi != 0)
    dy = np.divide(s, phi, out=np.full_like(phi, dt), where=phi != 0)
    x = x + np.float32(0.5) * dt * np.sin(psi) - dx
    y = y + np.float32(0.5) * dt * np.cos(psi) - dy
    output[:, 0], output[:, 1] = x * c - y * s, y * c + x * s


def metal_probe():
    try:
        torch.empty(1, dtype=torch.float64, device="mps")
    except TypeError as exc:
        print(f"MPS float64: {exc}")

    # Only free motion with known controls over dt=0.05. Feedback, crossings,
    # capture, rewards and episode resets are intentionally outside this probe.
    kernel = torch.mps.compile_shader("""
        using namespace metal;

        kernel void advance(device const float4* input, device float2* output,
                            uint i [[thread_position_in_grid]]) {
            float4 v = input[i];  // x, y, phi, psi
            const float dt = 0.05f;
            float a = v.z * dt;
            float s = sin(a), c = cos(a), h = sin(a / 2);
            float dx = v.z == 0 ? 0 : 2 * h * h / v.z;
            float dy = v.z == 0 ? dt : s / v.z;
            float x = v.x + 0.5f * dt * sin(v.w) - dx;
            float y = v.y + 0.5f * dt * cos(v.w) - dy;
            output[i] = float2(x * c - y * s, y * c + x * s);
        }
    """).advance

    rng = np.random.default_rng(17)
    print("Free-motion endpoint (batched calls, transfers excluded):")
    for size in (1, 64, 1024, 8192):
        inputs = rng.uniform([-4, -4, -1, -np.pi], [4, 4, 1, np.pi], (size, 4))
        inputs[0, 2] = 0
        inputs32 = inputs.astype(np.float32)
        numpy_output = np.empty((size, 2), dtype=np.float32)
        device_input = torch.from_numpy(inputs32).to("mps")
        output = torch.empty((size, 2), dtype=torch.float32, device="mps")
        expected = np.asarray([_path((x, y), phi, psi)(0.05) for x, y, phi, psi in inputs])
        numpy_advance(inputs32, numpy_output)
        np.testing.assert_allclose(numpy_output, expected, rtol=0, atol=1e-6)
        kernel(device_input, output, threads=size)
        actual = output.cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        error = np.max(np.abs(actual - expected))

        start = perf_counter()
        for _ in range(1000):
            numpy_advance(inputs32, numpy_output)
        cpu_us = (perf_counter() - start) * 1e3
        start = perf_counter()
        for _ in range(1000):
            kernel(device_input, output, threads=size)
        torch.mps.synchronize()
        mps_us = (perf_counter() - start) * 1e3
        print(
            f"  N={size:>4}: NumPy {cpu_us:5.1f} / Metal {mps_us:5.1f} µs/batch; max error={error:.2g}"
        )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8192, help="measured transitions per run")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1024 or args.steps % 1024 or args.repeats < 1:
        parser.error("--steps must be a positive multiple of 1024 and --repeats must be positive")
    torch.set_num_threads(1)
    precision_probe()
    configs = [("cpu", 1), ("cpu", 4)]
    if torch.backends.mps.is_available():
        metal_probe()
        configs.insert(1, ("mps", 1))
    else:
        print("MPS unavailable; running CPU measurements only.")

    print(f"PPO: {args.steps} transitions per run, plus 1024 for warm-up", flush=True)
    for device, workers in configs:
        times = [ppo_times(device, workers, args.steps) for _ in range(args.repeats)]
        rollout, update = map(median, zip(*times))
        total = median(r + u for r, u in times)
        print(
            f"  {device}/{workers}: rollout {rollout:.2f} s, update {update:.2f} s, "
            f"total {total:.2f} s, {args.steps / total:.0f} steps/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
