"""SAC/PPO reinforcement learning agent for options position management.

Implements a custom Gymnasium environment that models options position
dynamics and a manager class that trains and deploys Soft Actor-Critic
(SAC) or Proximal Policy Optimization (PPO) agents via Stable-Baselines3.

The RL agent learns optimal position scaling decisions: when to hold,
reduce, add to, or close options positions based on Greeks, implied
volatility, market regime, and unrealized P&L dynamics.

Usage::

    from src.ml.rl_agent import RLPositionManager, RLConfig

    config = RLConfig(algorithm="sac", total_timesteps=100_000)
    manager = RLPositionManager(config=config)
    result = manager.train(trade_history)
    action = manager.predict_action(current_state)
    recommendation = manager.get_position_recommendation(position_data)
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    import structlog
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# State space dimensionality
STATE_DIM: int = 13

# Default training hyperparameters
DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_BUFFER_SIZE: int = 100_000
DEFAULT_GAMMA: float = 0.99
DEFAULT_TAU: float = 0.005
DEFAULT_TOTAL_TIMESTEPS: int = 100_000

# Reward shaping constants
SHARPE_WINDOW: int = 20
HOLDING_PENALTY_SCALE: float = 0.01
DRAWDOWN_PENALTY_SCALE: float = 2.0
PROFIT_REVERSAL_PENALTY_SCALE: float = 1.5
PROFIT_REVERSAL_THRESHOLD: float = 0.50

# Action discretization thresholds
ACTION_CLOSE_THRESHOLD: float = -0.85
ACTION_REDUCE_75_THRESHOLD: float = -0.60
ACTION_REDUCE_50_THRESHOLD: float = -0.35
ACTION_REDUCE_25_THRESHOLD: float = -0.10
ACTION_ADD_25_THRESHOLD: float = 0.35
ACTION_ADD_50_THRESHOLD: float = 0.70

# State normalization bounds
PRICE_MAX: float = 10000.0
PORTFOLIO_VALUE_MAX: float = 500_000.0
DELTA_MAX: float = 1.0
GAMMA_MAX: float = 0.5
THETA_MAX: float = 100.0
VEGA_MAX: float = 200.0
IV_MAX: float = 2.0
DTE_MAX: float = 365.0
VIX_MAX: float = 80.0
MAX_DAYS_HELD: float = 90.0

# Training configuration
TRAIN_VAL_SPLIT: float = 0.80
EARLY_STOPPING_PATIENCE: int = 10
MIN_EPISODES_FOR_TRAINING: int = 50


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PositionAction(StrEnum):
    """Discretized position management actions."""

    CLOSE = "CLOSE"
    REDUCE_75 = "REDUCE_75"
    REDUCE_50 = "REDUCE_50"
    REDUCE_25 = "REDUCE_25"
    HOLD = "HOLD"
    ADD_25 = "ADD_25"
    ADD_50 = "ADD_50"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RLConfig(BaseModel):
    """Configuration for the RL position management agent.

    Attributes:
        algorithm: RL algorithm to use. SAC for continuous control
            with entropy regularization, PPO for on-policy stability.
        learning_rate: Optimizer learning rate.
        batch_size: Mini-batch size for gradient updates.
        buffer_size: Replay buffer capacity (SAC only).
        gamma: Discount factor for future rewards.
        tau: Soft update coefficient for target networks (SAC only).
        total_timesteps: Total environment steps for training.
        target_dte: Target days to expiration for position holding.
        max_drawdown_pct: Maximum allowable drawdown before penalty.
        model_dir: Directory for saving/loading model artifacts.
        seed: Random seed for reproducibility.
        verbose: Verbosity level for SB3 training output.
        n_eval_episodes: Number of episodes for evaluation.
        early_stopping_patience: Patience epochs for early stopping.
    """

    algorithm: Literal["sac", "ppo"] = Field(
        default="sac",
        description="RL algorithm: 'sac' or 'ppo'",
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        gt=0.0,
        description="Optimizer learning rate",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        gt=0,
        description="Mini-batch size for training",
    )
    buffer_size: int = Field(
        default=DEFAULT_BUFFER_SIZE,
        gt=0,
        description="Replay buffer capacity (SAC only)",
    )
    gamma: float = Field(
        default=DEFAULT_GAMMA,
        ge=0.0,
        le=1.0,
        description="Discount factor for future rewards",
    )
    tau: float = Field(
        default=DEFAULT_TAU,
        ge=0.0,
        le=1.0,
        description="Soft update coefficient (SAC only)",
    )
    total_timesteps: int = Field(
        default=DEFAULT_TOTAL_TIMESTEPS,
        gt=0,
        description="Total training timesteps",
    )
    target_dte: float = Field(
        default=45.0,
        gt=0.0,
        description="Target DTE for position holding period",
    )
    max_drawdown_pct: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Maximum drawdown fraction before harsh penalty",
    )
    model_dir: str = Field(
        default="models/",
        description="Directory for model artifact storage",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    verbose: int = Field(
        default=0,
        ge=0,
        le=2,
        description="SB3 training verbosity (0=silent, 1=info, 2=debug)",
    )
    n_eval_episodes: int = Field(
        default=20,
        gt=0,
        description="Number of episodes for evaluation",
    )
    early_stopping_patience: int = Field(
        default=EARLY_STOPPING_PATIENCE,
        gt=0,
        description="Early stopping patience in evaluation rounds",
    )


class RLState(BaseModel):
    """Observation state for the RL agent at a single timestep.

    All values should be raw (unnormalized); the environment handles
    normalization to [-1, 1] internally.

    Attributes:
        price: Current underlying price.
        portfolio_value: Total portfolio value in dollars.
        delta: Position net delta exposure.
        gamma: Position net gamma exposure.
        theta: Position net theta (daily decay).
        vega: Position net vega exposure.
        iv: Current implied volatility of the position.
        time_to_expiry: Days remaining until expiration.
        vix: Current VIX level.
        regime_encoded: Numeric regime encoding (0-3).
        unrealized_pnl_pct: Unrealized P&L as fraction of max risk.
        days_held: Number of calendar days the position has been held.
        max_profit_pct_reached: Highest unrealized profit fraction seen.
    """

    price: float = Field(description="Current underlying price")
    portfolio_value: float = Field(
        description="Total portfolio value in dollars",
    )
    delta: float = Field(description="Position net delta")
    gamma: float = Field(description="Position net gamma")
    theta: float = Field(description="Position net theta (daily)")
    vega: float = Field(description="Position net vega")
    iv: float = Field(description="Current implied volatility")
    time_to_expiry: float = Field(description="Days to expiration")
    vix: float = Field(description="Current VIX level")
    regime_encoded: float = Field(
        description="Regime encoding: 0=low_vol, 1=high_vol, 2=range, 3=crisis",
    )
    unrealized_pnl_pct: float = Field(
        description="Unrealized P&L as fraction of max risk",
    )
    days_held: float = Field(description="Calendar days position held")
    max_profit_pct_reached: float = Field(
        description="Peak unrealized profit fraction observed",
    )


class RLAction(BaseModel):
    """Action output from the RL agent.

    Attributes:
        raw_action: Continuous action value in [-1, 1].
        discrete_action: Discretized action label.
        confidence: Agent confidence (entropy-based for SAC).
    """

    raw_action: float = Field(
        ge=-1.0,
        le=1.0,
        description="Continuous action in [-1, 1]",
    )
    discrete_action: PositionAction = Field(
        description="Discretized position action",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Action confidence (0=uncertain, 1=certain)",
    )


class TrainingResult(BaseModel):
    """Results from RL agent training.

    Attributes:
        algorithm: Algorithm used for training.
        total_timesteps: Total environment steps completed.
        training_time_seconds: Wall-clock training duration.
        mean_reward: Mean episode reward during training.
        std_reward: Standard deviation of episode rewards.
        mean_episode_length: Average episode length in steps.
        val_mean_reward: Mean reward on validation episodes.
        val_std_reward: Reward standard deviation on validation set.
        best_mean_reward: Best rolling mean reward observed.
        n_train_episodes: Number of training episodes used.
        n_val_episodes: Number of validation episodes used.
        model_path: Path where the trained model was saved.
    """

    algorithm: str = Field(description="Training algorithm name")
    total_timesteps: int = Field(
        description="Total environment steps completed",
    )
    training_time_seconds: float = Field(
        description="Wall-clock training time in seconds",
    )
    mean_reward: float = Field(
        description="Mean episode reward during training",
    )
    std_reward: float = Field(
        description="Standard deviation of episode rewards",
    )
    mean_episode_length: float = Field(
        description="Average episode length in steps",
    )
    val_mean_reward: float = Field(
        description="Mean reward on validation episodes",
    )
    val_std_reward: float = Field(
        description="Reward std dev on validation set",
    )
    best_mean_reward: float = Field(
        description="Best rolling mean reward seen during training",
    )
    n_train_episodes: int = Field(
        description="Number of training episodes",
    )
    n_val_episodes: int = Field(
        description="Number of validation episodes",
    )
    model_path: str = Field(
        description="Path to saved model artifact",
    )


class PositionEnvironmentConfig(BaseModel):
    """Configuration for the options position Gymnasium environment.

    Attributes:
        target_dte: Ideal holding period in days to expiration.
        max_drawdown_pct: Drawdown fraction triggering harsh penalty.
        episode_max_steps: Maximum steps per episode.
        reward_scale: Multiplier applied to all reward values.
        slippage_bps: Assumed slippage in basis points per action.
    """

    target_dte: float = Field(
        default=45.0,
        gt=0.0,
        description="Target DTE for position holding",
    )
    max_drawdown_pct: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Max drawdown before harsh penalty",
    )
    episode_max_steps: int = Field(
        default=200,
        gt=0,
        description="Maximum timesteps per episode",
    )
    reward_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Reward scaling multiplier",
    )
    slippage_bps: float = Field(
        default=15.0,
        ge=0.0,
        description="Assumed slippage in basis points per trade",
    )


# ---------------------------------------------------------------------------
# Custom Gymnasium Environment
# ---------------------------------------------------------------------------


class OptionsPositionEnv(gym.Env):
    """Gymnasium environment simulating options position management.

    The agent observes a 13-dimensional state vector describing the
    current position and market conditions, and outputs a continuous
    action in [-1, 1] representing a position scale factor:
      -1 = close the position entirely
       0 = hold at current size
      +1 = add maximally to the position

    Rewards are based on a rolling Sharpe ratio over the most recent
    trades, with penalties for excessive holding, drawdowns, and
    allowing profitable positions to reverse.

    Args:
        trade_episodes: List of episode data, each a list of
            state dictionaries representing sequential timesteps
            of a single position's life.
        config: Environment configuration parameters.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        trade_episodes: list[list[dict[str, float]]],
        config: PositionEnvironmentConfig | None = None,
    ) -> None:
        super().__init__()
        self._config = config or PositionEnvironmentConfig()
        self._trade_episodes = trade_episodes
        self._log: structlog.stdlib.BoundLogger = get_logger(
            "ml.rl_agent.env",
        )

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Episode state
        self._current_episode_idx: int = 0
        self._current_step: int = 0
        self._current_episode: list[dict[str, float]] = []
        self._episode_rewards: list[float] = []
        self._recent_trade_returns: list[float] = []
        self._cumulative_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._position_scale: float = 1.0
        self._rng = np.random.default_rng(42)

        self._log.info(
            "options_position_env_initialized",
            n_episodes=len(trade_episodes),
            max_steps=self._config.episode_max_steps,
            target_dte=self._config.target_dte,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Initialize a new episode from historical trade data.

        Selects the next episode from the trade history (cycling
        through all available episodes) and returns the initial
        observation.

        Args:
            seed: Optional random seed for reproducibility.
            options: Additional reset options (unused).

        Returns:
            Tuple of (observation, info_dict) where observation is
            a normalized state vector and info contains episode metadata.
        """
        super().reset(seed=seed)

        if not self._trade_episodes:
            # Return a zero state if no episodes available
            obs = np.zeros(STATE_DIM, dtype=np.float32)
            return obs, {"episode_idx": -1, "episode_length": 0}

        # Select episode (sequential with wrap-around)
        self._current_episode_idx = self._current_episode_idx % len(
            self._trade_episodes
        )
        self._current_episode = self._trade_episodes[self._current_episode_idx]
        self._current_episode_idx += 1

        # Reset episode state
        self._current_step = 0
        self._episode_rewards = []
        self._cumulative_pnl = 0.0
        self._peak_pnl = 0.0
        self._position_scale = 1.0

        obs = self._get_observation()
        info: dict[str, Any] = {
            "episode_idx": self._current_episode_idx - 1,
            "episode_length": len(self._current_episode),
        }

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one timestep and compute reward.

        Applies the agent's action to the position, calculates the
        resulting reward with Sharpe-based shaping and penalties,
        and determines whether the episode has terminated.

        Args:
            action: Continuous action array of shape (1,) in [-1, 1].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action_value = float(np.clip(action[0], -1.0, 1.0))

        # Update position scale based on action
        self._position_scale = self._apply_action(action_value)

        # Get current and next state data
        current_data = self._get_step_data(self._current_step)
        self._current_step += 1
        next_data = self._get_step_data(self._current_step)

        # Calculate step P&L
        step_pnl = self._calculate_step_pnl(
            current_data,
            next_data,
            self._position_scale,
        )

        # Apply slippage for non-hold actions
        if abs(action_value) > ACTION_REDUCE_25_THRESHOLD:
            slippage_cost = (
                self._config.slippage_bps
                / 10_000.0
                * abs(action_value)
                * current_data.get("price", 100.0)
            )
            step_pnl -= slippage_cost

        # Update cumulative tracking
        self._cumulative_pnl += step_pnl
        self._peak_pnl = max(self._peak_pnl, self._cumulative_pnl)

        # Calculate reward
        reward = self._calculate_reward(
            step_pnl,
            action_value,
            current_data,
        )
        self._episode_rewards.append(reward)

        # Track trade returns for Sharpe calculation
        self._recent_trade_returns.append(step_pnl)
        if len(self._recent_trade_returns) > SHARPE_WINDOW:
            self._recent_trade_returns.pop(0)

        # Check termination conditions
        terminated = self._check_terminated(action_value, current_data)
        truncated = self._current_step >= min(
            len(self._current_episode) - 1,
            self._config.episode_max_steps,
        )

        obs = self._get_observation()
        info: dict[str, Any] = {
            "step_pnl": step_pnl,
            "cumulative_pnl": self._cumulative_pnl,
            "position_scale": self._position_scale,
            "action_value": action_value,
            "discrete_action": discretize_action(action_value).value,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build and normalize the current observation vector.

        Returns:
            Normalized state vector of shape (STATE_DIM,) with
            values in [-1, 1].
        """
        data = self._get_step_data(self._current_step)
        return self._normalize_state(data)

    def _get_step_data(self, step: int) -> dict[str, float]:
        """Retrieve state data for a given step index.

        Safely handles out-of-bounds steps by clamping to the
        last available step.

        Args:
            step: Zero-based step index within the current episode.

        Returns:
            Dictionary of state values for the requested step.
        """
        if not self._current_episode:
            return self._empty_state()

        clamped_step = min(step, len(self._current_episode) - 1)
        return self._current_episode[clamped_step]

    def _normalize_state(
        self,
        data: dict[str, float],
    ) -> np.ndarray:
        """Normalize raw state values to the [-1, 1] range.

        Each feature is scaled by its expected maximum value and
        clipped to ensure the result stays within bounds. This is
        critical for stable neural network training.

        Args:
            data: Raw state dictionary from trade episode data.

        Returns:
            Normalized numpy array of shape (STATE_DIM,) in [-1, 1].
        """
        state = np.array(
            [
                data.get("price", 0.0) / PRICE_MAX,
                data.get("portfolio_value", 0.0) / PORTFOLIO_VALUE_MAX,
                data.get("delta", 0.0) / DELTA_MAX,
                data.get("gamma", 0.0) / GAMMA_MAX,
                data.get("theta", 0.0) / THETA_MAX,
                data.get("vega", 0.0) / VEGA_MAX,
                data.get("iv", 0.0) / IV_MAX,
                data.get("time_to_expiry", 0.0) / DTE_MAX,
                data.get("vix", 0.0) / VIX_MAX,
                data.get("regime_encoded", 0.0) / 3.0,
                data.get("unrealized_pnl_pct", 0.0),
                data.get("days_held", 0.0) / MAX_DAYS_HELD,
                data.get("max_profit_pct_reached", 0.0),
            ],
            dtype=np.float32,
        )
        return np.clip(state, -1.0, 1.0)

    def _calculate_reward(
        self,
        step_pnl: float,
        action_value: float,
        state_data: dict[str, float],
    ) -> float:
        """Compute the shaped reward for a single timestep.

        The reward combines a rolling Sharpe ratio component with
        penalties for undesirable behaviors:
          - Holding too long past target DTE
          - Drawdown exceeding the configured threshold
          - Allowing profitable trades to reverse significantly

        Args:
            step_pnl: Dollar P&L for this timestep.
            action_value: The continuous action taken by the agent.
            state_data: Current state dictionary.

        Returns:
            Scalar reward value.
        """
        reward = 0.0

        # --- Sharpe-based reward component ---
        if len(self._recent_trade_returns) >= 2:
            returns = np.array(
                self._recent_trade_returns,
                dtype=np.float32,
            )
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))
            if std_ret > 1e-8:
                sharpe = mean_ret / std_ret
                reward += sharpe
            else:
                # If std is near zero, use sign of mean return
                reward += np.sign(mean_ret) * 0.1
        else:
            # Early in episode, use raw P&L signal
            reward += step_pnl / max(
                state_data.get("price", 100.0),
                1.0,
            )

        # --- Penalty: excessive holding beyond target DTE ---
        days_held = state_data.get("days_held", 0.0)
        time_to_expiry = state_data.get("time_to_expiry", 45.0)
        if time_to_expiry < self._config.target_dte * 0.3:
            # Position is getting close to expiration
            overheld_fraction = max(
                0.0,
                1.0 - time_to_expiry / (self._config.target_dte * 0.3),
            )
            holding_penalty = HOLDING_PENALTY_SCALE * overheld_fraction * days_held
            reward -= holding_penalty

        # --- Penalty: drawdown beyond threshold ---
        if self._peak_pnl > 0.0:
            current_drawdown = (self._peak_pnl - self._cumulative_pnl) / self._peak_pnl
        else:
            current_drawdown = 0.0

        if current_drawdown > self._config.max_drawdown_pct:
            excess_dd = current_drawdown - self._config.max_drawdown_pct
            reward -= DRAWDOWN_PENALTY_SCALE * excess_dd

        # --- Penalty: letting profitable trades reverse ---
        max_profit_pct = state_data.get("max_profit_pct_reached", 0.0)
        unrealized_pnl_pct = state_data.get("unrealized_pnl_pct", 0.0)
        if max_profit_pct > 0.1:
            profit_given_back = max_profit_pct - unrealized_pnl_pct
            if profit_given_back > (PROFIT_REVERSAL_THRESHOLD * max_profit_pct):
                reversal_ratio = profit_given_back / max(
                    max_profit_pct,
                    1e-8,
                )
                reward -= PROFIT_REVERSAL_PENALTY_SCALE * reversal_ratio

        # Apply reward scaling
        reward *= self._config.reward_scale

        return float(reward)

    def _apply_action(self, action_value: float) -> float:
        """Map continuous action to a position scale factor.

        The action is interpreted as a modification to the current
        position scale. Negative values reduce the position and
        positive values increase it, with the scale clamped to [0, 2].

        Args:
            action_value: Continuous action in [-1, 1].

        Returns:
            Updated position scale factor in [0.0, 2.0].
        """
        if action_value <= ACTION_CLOSE_THRESHOLD:
            return 0.0  # Close position entirely

        # Interpolate: action maps to scale adjustment
        # -0.85 to 0 maps to reducing scale
        # 0 to 1 maps to increasing scale
        new_scale = self._position_scale + action_value * 0.25
        return float(np.clip(new_scale, 0.0, 2.0))

    def _calculate_step_pnl(
        self,
        current_data: dict[str, float],
        next_data: dict[str, float],
        scale: float,
    ) -> float:
        """Calculate P&L for a single timestep transition.

        Uses the position's delta to estimate P&L from the underlying
        price change, plus theta decay. This is a first-order
        approximation suitable for RL training.

        Args:
            current_data: State at the current timestep.
            next_data: State at the next timestep.
            scale: Current position scale factor.

        Returns:
            Estimated dollar P&L for the step.
        """
        price_current = current_data.get("price", 0.0)
        price_next = next_data.get("price", 0.0)
        delta = current_data.get("delta", 0.0)
        theta = current_data.get("theta", 0.0)
        gamma = current_data.get("gamma", 0.0)

        price_change = price_next - price_current

        # First-order Greeks approximation
        # P&L ~ delta * dS + 0.5 * gamma * dS^2 + theta * dt
        delta_pnl = delta * price_change * scale
        gamma_pnl = 0.5 * gamma * price_change**2 * scale
        theta_pnl = theta * scale  # theta is daily, one step = one day

        return float(delta_pnl + gamma_pnl + theta_pnl)

    def _check_terminated(
        self,
        action_value: float,
        state_data: dict[str, float],
    ) -> bool:
        """Determine whether the episode should terminate.

        Termination occurs when:
          - The agent chooses to close the position
          - Time to expiry reaches zero
          - The position is fully scaled down

        Args:
            action_value: Action taken this step.
            state_data: Current state dictionary.

        Returns:
            True if the episode should terminate.
        """
        # Agent explicitly closes position
        if action_value <= ACTION_CLOSE_THRESHOLD:
            return True

        # Position expired
        if state_data.get("time_to_expiry", 1.0) <= 0.0:
            return True

        # Position fully scaled down
        return self._position_scale <= 0.0

    @staticmethod
    def _empty_state() -> dict[str, float]:
        """Return a zeroed state dictionary.

        Returns:
            Dictionary with all state keys set to 0.0.
        """
        return {
            "price": 0.0,
            "portfolio_value": 0.0,
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "iv": 0.0,
            "time_to_expiry": 0.0,
            "vix": 0.0,
            "regime_encoded": 0.0,
            "unrealized_pnl_pct": 0.0,
            "days_held": 0.0,
            "max_profit_pct_reached": 0.0,
        }


# ---------------------------------------------------------------------------
# Replay Buffer Builder
# ---------------------------------------------------------------------------


class ReplayBufferBuilder:
    """Converts historical trade data into RL-compatible episode lists.

    Takes raw trade outcome DataFrames and produces the episode format
    expected by :class:`OptionsPositionEnv`.  Each episode represents
    the complete lifecycle of a single options position.

    The builder maps trade outcomes to (state, action, reward,
    next_state, done) tuples internally, though the primary output
    format is episode lists of state dictionaries for the custom
    environment.
    """

    def __init__(self) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger(
            "ml.rl_agent.replay_buffer",
        )

    def build_episodes(
        self,
        trades: pd.DataFrame,
    ) -> list[list[dict[str, float]]]:
        """Convert trade history into episode lists for the environment.

        Each row in the input DataFrame should represent a timestep
        within a position's lifecycle, grouped by a ``trade_id`` column.
        Required columns map to the state space fields.

        Args:
            trades: DataFrame with columns including ``trade_id``,
                ``price``, ``portfolio_value``, ``delta``, ``gamma``,
                ``theta``, ``vega``, ``iv``, ``time_to_expiry``,
                ``vix``, ``regime_encoded``, ``unrealized_pnl_pct``,
                ``days_held``, ``max_profit_pct_reached``.

        Returns:
            List of episodes, where each episode is a list of state
            dictionaries representing sequential timesteps.
        """
        if trades.empty:
            self._log.warning("empty_trade_dataframe_for_replay_buffer")
            return []

        state_columns = [
            "price",
            "portfolio_value",
            "delta",
            "gamma",
            "theta",
            "vega",
            "iv",
            "time_to_expiry",
            "vix",
            "regime_encoded",
            "unrealized_pnl_pct",
            "days_held",
            "max_profit_pct_reached",
        ]

        # Verify required columns exist
        missing_cols = [c for c in state_columns if c not in trades.columns]
        if missing_cols:
            self._log.error(
                "missing_columns_for_replay_buffer",
                missing=missing_cols,
            )
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Group by trade_id to form episodes
        group_col = "trade_id"
        episodes: list[list[dict[str, float]]] = []
        if group_col not in trades.columns:
            self._log.warning(
                "no_trade_id_column_treating_as_single_episode",
            )
            episode_data = trades[state_columns].to_dict("records")
            episodes.append(
                [{k: float(v) for k, v in row.items()} for row in episode_data]
            )
        else:
            for trade_id, group in trades.groupby(group_col):
                if len(group) < 2:
                    self._log.debug(
                        "skipping_single_step_episode",
                        trade_id=trade_id,
                    )
                    continue

                episode_data = group[state_columns].to_dict("records")
                episode = [
                    {k: float(v) for k, v in row.items()} for row in episode_data
                ]
                episodes.append(episode)

        self._log.info(
            "replay_episodes_built",
            n_episodes=len(episodes),
            avg_episode_length=(
                round(
                    np.mean([len(ep) for ep in episodes]),
                    1,
                )
                if episodes
                else 0.0
            ),
        )

        return episodes

    def build_buffer(
        self,
        trades: pd.DataFrame,
    ) -> list[
        tuple[
            dict[str, float],
            float,
            float,
            dict[str, float],
            bool,
        ]
    ]:
        """Create a flat replay buffer of transition tuples.

        Converts trade history into (state, action, reward, next_state,
        done) tuples suitable for offline RL training or analysis.

        The action and reward are inferred from the trade data:
          - Action is derived from position scale changes
          - Reward is the step-level P&L change

        Args:
            trades: DataFrame with state columns and optionally
                ``action`` and ``realized_pnl`` columns.

        Returns:
            List of (state, action, reward, next_state, done) tuples.
        """
        episodes = self.build_episodes(trades)

        transitions: list[
            tuple[
                dict[str, float],
                float,
                float,
                dict[str, float],
                bool,
            ]
        ] = []

        for episode in episodes:
            for i in range(len(episode) - 1):
                state = episode[i]
                next_state = episode[i + 1]
                done = i == len(episode) - 2

                # Infer action from PnL trajectory
                pnl_current = state.get("unrealized_pnl_pct", 0.0)
                pnl_next = next_state.get("unrealized_pnl_pct", 0.0)
                action = float(np.clip(pnl_next - pnl_current, -1.0, 1.0))

                # Reward is the change in unrealized P&L
                reward = pnl_next - pnl_current

                transitions.append(
                    (state, action, reward, next_state, done),
                )

        self._log.info(
            "replay_buffer_built",
            n_transitions=len(transitions),
            n_episodes=len(episodes),
        )

        return transitions


# ---------------------------------------------------------------------------
# Training Callback
# ---------------------------------------------------------------------------


class PrometheusTrainingCallback:
    """Stable-Baselines3 callback that logs training metrics.

    Publishes episode rewards, episode lengths, and loss values to
    the structlog logger and optionally to Prometheus gauges. This
    callback is compatible with SB3's callback interface.

    Args:
        log_interval: Log metrics every N timesteps.
    """

    def __init__(self, log_interval: int = 1000) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        self._base_cls = BaseCallback
        self._log_interval = log_interval
        self._logger: structlog.stdlib.BoundLogger = get_logger(
            "ml.rl_agent.callback",
        )
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def create_callback(self) -> BaseCallback:
        """Create a SB3-compatible callback instance.

        Returns:
            A BaseCallback subclass instance that logs training
            progress to structlog and Prometheus.
        """
        from stable_baselines3.common.callbacks import BaseCallback

        outer_self = self

        class _InnerCallback(BaseCallback):
            """SB3 callback that delegates to PrometheusTrainingCallback."""

            def __init__(self) -> None:
                super().__init__(verbose=0)
                self._step_count: int = 0

            def _on_step(self) -> bool:
                self._step_count += 1

                # Collect episode info from the monitor wrapper
                infos = self.locals.get("infos", [])
                for info in infos:
                    if "episode" in info:
                        ep_reward = info["episode"]["r"]
                        ep_length = info["episode"]["l"]
                        outer_self._episode_rewards.append(
                            float(ep_reward),
                        )
                        outer_self._episode_lengths.append(
                            int(ep_length),
                        )

                # Log at interval
                if (
                    self._step_count % outer_self._log_interval == 0
                    and outer_self._episode_rewards
                ):
                    recent_rewards = outer_self._episode_rewards[-100:]
                    mean_reward = float(np.mean(recent_rewards))
                    std_reward = float(np.std(recent_rewards))

                    outer_self._logger.info(
                        "rl_training_progress",
                        timestep=self._step_count,
                        mean_reward_100=round(mean_reward, 4),
                        std_reward_100=round(std_reward, 4),
                        n_episodes=len(
                            outer_self._episode_rewards,
                        ),
                    )

                    # Update Prometheus metrics if available
                    try:
                        from src.utils.metrics import (
                            ML_CONFIDENCE,
                        )

                        ML_CONFIDENCE.observe(
                            max(0.0, min(1.0, (mean_reward + 1) / 2)),
                        )
                    except ImportError:
                        pass

                return True

        return _InnerCallback()

    @property
    def episode_rewards(self) -> list[float]:
        """Return collected episode rewards."""
        return list(self._episode_rewards)

    @property
    def episode_lengths(self) -> list[int]:
        """Return collected episode lengths."""
        return list(self._episode_lengths)


# ---------------------------------------------------------------------------
# RL Position Manager
# ---------------------------------------------------------------------------


class RLPositionManager:
    """Manages RL-based position sizing and exit decisions.

    Wraps the Stable-Baselines3 SAC or PPO algorithm with a custom
    Gymnasium environment to learn optimal options position management
    from historical trade data.

    Args:
        config: RL configuration parameters.
        model_path: Optional path to a pre-trained model to load.
    """

    def __init__(
        self,
        config: RLConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        self._config = config or RLConfig()
        self._model_dir = Path(self._config.model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log: structlog.stdlib.BoundLogger = get_logger(
            "ml.rl_agent",
        )
        self._model: BaseAlgorithm | None = None
        self._env: OptionsPositionEnv | None = None
        self._training_callback = PrometheusTrainingCallback()
        self._is_trained: bool = False

        if model_path is not None:
            self.load(model_path)

        self._log.info(
            "rl_position_manager_initialized",
            algorithm=self._config.algorithm,
            learning_rate=self._config.learning_rate,
            model_path=model_path,
        )

    def train(
        self,
        trade_history: pd.DataFrame,
        epochs: int | None = None,
    ) -> TrainingResult:
        """Train the RL agent on historical trade outcomes.

        Builds episodes from trade history, creates a temporal
        train/validation split, trains the agent, and evaluates
        on the held-out validation set.

        Args:
            trade_history: DataFrame of historical trade outcomes.
                Must contain state columns required by
                :class:`ReplayBufferBuilder`.
            epochs: Override for total training timesteps. If None,
                uses the value from config.

        Returns:
            A :class:`TrainingResult` with training and validation
            metrics.

        Raises:
            ValueError: If insufficient trade history is provided.
        """
        start_time = time.monotonic()
        total_timesteps = epochs or self._config.total_timesteps

        self._log.info(
            "rl_training_started",
            algorithm=self._config.algorithm,
            total_timesteps=total_timesteps,
            n_trades=len(trade_history),
        )

        # Build episodes from trade history
        buffer_builder = ReplayBufferBuilder()
        all_episodes = buffer_builder.build_episodes(trade_history)

        if len(all_episodes) < MIN_EPISODES_FOR_TRAINING:
            raise ValueError(
                f"Insufficient episodes for training: "
                f"{len(all_episodes)} < {MIN_EPISODES_FOR_TRAINING}. "
                f"Need at least {MIN_EPISODES_FOR_TRAINING} complete "
                f"trade episodes."
            )

        # Temporal train/validation split
        split_idx = int(len(all_episodes) * TRAIN_VAL_SPLIT)
        train_episodes = all_episodes[:split_idx]
        val_episodes = all_episodes[split_idx:]

        self._log.info(
            "episode_split",
            n_train=len(train_episodes),
            n_val=len(val_episodes),
            split_ratio=TRAIN_VAL_SPLIT,
        )

        # Create training environment
        env_config = PositionEnvironmentConfig(
            target_dte=self._config.target_dte,
            max_drawdown_pct=self._config.max_drawdown_pct,
        )

        train_env = self._wrap_env(
            OptionsPositionEnv(
                trade_episodes=train_episodes,
                config=env_config,
            ),
        )

        # Create the RL model
        self._model = self._create_model(train_env)

        # Train with callback
        callback = self._training_callback.create_callback()
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=self._config.verbose > 0,
        )

        self._is_trained = True
        self._env = OptionsPositionEnv(
            trade_episodes=train_episodes,
            config=env_config,
        )

        # Evaluate on validation set
        val_mean, val_std = self._evaluate_episodes(val_episodes)

        # Compute training statistics
        ep_rewards = self._training_callback.episode_rewards
        ep_lengths = self._training_callback.episode_lengths
        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        std_reward = float(np.std(ep_rewards)) if ep_rewards else 0.0
        mean_length = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        best_reward = float(np.max(ep_rewards)) if ep_rewards else 0.0

        # Save model
        model_path = self._save_model()

        elapsed = time.monotonic() - start_time

        result = TrainingResult(
            algorithm=self._config.algorithm,
            total_timesteps=total_timesteps,
            training_time_seconds=round(elapsed, 2),
            mean_reward=round(mean_reward, 4),
            std_reward=round(std_reward, 4),
            mean_episode_length=round(mean_length, 1),
            val_mean_reward=round(val_mean, 4),
            val_std_reward=round(val_std, 4),
            best_mean_reward=round(best_reward, 4),
            n_train_episodes=len(train_episodes),
            n_val_episodes=len(val_episodes),
            model_path=model_path,
        )

        self._log.info(
            "rl_training_complete",
            algorithm=self._config.algorithm,
            mean_reward=result.mean_reward,
            val_mean_reward=result.val_mean_reward,
            training_time_seconds=result.training_time_seconds,
            model_path=model_path,
        )

        return result

    def predict_action(self, state: RLState) -> RLAction:
        """Given current position state, recommend a management action.

        Converts the Pydantic state model to a numpy observation,
        runs inference through the trained model, and returns a
        structured action with confidence estimate.

        Args:
            state: Current position state observation.

        Returns:
            An :class:`RLAction` with raw continuous value,
            discretized label, and confidence.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not available. Call train() or load() first.")

        # Convert state to normalized observation
        state_dict: dict[str, float] = {
            "price": state.price,
            "portfolio_value": state.portfolio_value,
            "delta": state.delta,
            "gamma": state.gamma,
            "theta": state.theta,
            "vega": state.vega,
            "iv": state.iv,
            "time_to_expiry": state.time_to_expiry,
            "vix": state.vix,
            "regime_encoded": state.regime_encoded,
            "unrealized_pnl_pct": state.unrealized_pnl_pct,
            "days_held": state.days_held,
            "max_profit_pct_reached": state.max_profit_pct_reached,
        }

        obs = _normalize_state_dict(state_dict)

        # Get action from model (deterministic for inference)
        action_array, _states = self._model.predict(
            obs,
            deterministic=True,
        )
        raw_action = float(np.clip(action_array[0], -1.0, 1.0))

        # Estimate confidence from action magnitude
        # Higher magnitude = more decisive = higher confidence
        confidence = min(1.0, abs(raw_action) * 1.2 + 0.1)

        # Discretize
        discrete = discretize_action(raw_action)

        action = RLAction(
            raw_action=round(raw_action, 4),
            discrete_action=discrete,
            confidence=round(confidence, 4),
        )

        self._log.debug(
            "rl_action_predicted",
            raw_action=action.raw_action,
            discrete_action=action.discrete_action.value,
            confidence=action.confidence,
        )

        return action

    def evaluate(
        self,
        trade_history: pd.DataFrame,
    ) -> dict[str, float]:
        """Evaluate agent performance on held-out trade data.

        Runs the trained agent through episodes built from the
        provided trade history and collects aggregate statistics.

        Args:
            trade_history: DataFrame of trade outcomes to evaluate on.

        Returns:
            Dictionary of evaluation metrics including mean_reward,
            std_reward, mean_episode_length, win_rate, and
            mean_cumulative_pnl.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not available. Call train() or load() first.")

        buffer_builder = ReplayBufferBuilder()
        episodes = buffer_builder.build_episodes(trade_history)

        if not episodes:
            self._log.warning("no_episodes_for_evaluation")
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_episode_length": 0.0,
                "win_rate": 0.0,
                "mean_cumulative_pnl": 0.0,
            }

        mean_reward, std_reward = self._evaluate_episodes(episodes)

        # Detailed per-episode evaluation
        episode_pnls: list[float] = []
        episode_lengths: list[int] = []

        env_config = PositionEnvironmentConfig(
            target_dte=self._config.target_dte,
            max_drawdown_pct=self._config.max_drawdown_pct,
        )

        for episode in episodes:
            env = OptionsPositionEnv(
                trade_episodes=[episode],
                config=env_config,
            )
            obs, _info = env.reset()
            cumulative_pnl = 0.0
            steps = 0

            done = False
            while not done:
                action, _states = self._model.predict(
                    obs,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = env.step(
                    action,
                )
                cumulative_pnl = info.get(
                    "cumulative_pnl",
                    cumulative_pnl,
                )
                steps += 1
                done = terminated or truncated

            episode_pnls.append(cumulative_pnl)
            episode_lengths.append(steps)

        win_rate = (
            sum(1 for p in episode_pnls if p > 0) / len(episode_pnls)
            if episode_pnls
            else 0.0
        )

        metrics = {
            "mean_reward": round(mean_reward, 4),
            "std_reward": round(std_reward, 4),
            "mean_episode_length": round(
                float(np.mean(episode_lengths)),
                1,
            ),
            "win_rate": round(win_rate, 4),
            "mean_cumulative_pnl": round(
                float(np.mean(episode_pnls)),
                2,
            ),
            "std_cumulative_pnl": round(
                float(np.std(episode_pnls)),
                2,
            ),
            "max_cumulative_pnl": round(
                float(np.max(episode_pnls)),
                2,
            ),
            "min_cumulative_pnl": round(
                float(np.min(episode_pnls)),
                2,
            ),
            "n_episodes": len(episodes),
        }

        self._log.info(
            "rl_evaluation_complete",
            mean_reward=metrics["mean_reward"],
            win_rate=metrics["win_rate"],
            mean_pnl=metrics["mean_cumulative_pnl"],
            n_episodes=metrics["n_episodes"],
        )

        return metrics

    def save(self, path: str) -> str:
        """Save the trained model and metadata to disk.

        Args:
            path: File path for the model artifact (without extension).

        Returns:
            Absolute path to the saved model file.

        Raises:
            RuntimeError: If no trained model is available to save.
        """
        if self._model is None:
            raise RuntimeError("No model to save. Train first.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self._model.save(str(model_path))

        # Save metadata sidecar
        metadata = {
            "algorithm": self._config.algorithm,
            "learning_rate": self._config.learning_rate,
            "batch_size": self._config.batch_size,
            "gamma": self._config.gamma,
            "tau": self._config.tau,
            "saved_at": datetime.now(UTC).isoformat(),
            "is_trained": self._is_trained,
        }
        meta_path = model_path.parent / f"{model_path.stem}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._log.info(
            "rl_model_saved",
            path=str(model_path.resolve()),
            metadata_path=str(meta_path.resolve()),
        )

        return str(model_path.resolve())

    def load(self, path: str) -> None:
        """Load a pre-trained model from disk.

        Detects the algorithm type from the metadata sidecar file
        and loads the appropriate SB3 model class.

        Args:
            path: Path to the saved model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(path)

        # SB3 appends .zip if not present
        if not model_path.exists() and not model_path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Try to load metadata for algorithm detection
        meta_path = model_path.parent / f"{model_path.stem}_metadata.json"
        algorithm = self._config.algorithm

        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            algorithm = metadata.get("algorithm", algorithm)
            self._log.debug(
                "rl_metadata_loaded",
                algorithm=algorithm,
                metadata=metadata,
            )

        # Load the model with the correct class
        if algorithm == "ppo":
            from stable_baselines3 import PPO

            self._model = PPO.load(str(model_path))
        else:
            from stable_baselines3 import SAC

            self._model = SAC.load(str(model_path))

        self._is_trained = True

        self._log.info(
            "rl_model_loaded",
            path=str(model_path),
            algorithm=algorithm,
        )

    def get_position_recommendation(
        self,
        position_data: dict[str, float],
    ) -> dict[str, Any]:
        """High-level API returning an action recommendation string.

        Accepts a raw dictionary of position metrics and returns
        a structured recommendation with the action, confidence,
        and reasoning.

        Args:
            position_data: Dictionary with keys matching
                :class:`RLState` field names.

        Returns:
            Dictionary containing ``action`` (string), ``confidence``
            (float), ``raw_value`` (float), and ``reasoning`` (string).

        Raises:
            RuntimeError: If the model is not trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not available. Call train() or load() first.")

        # Build RLState from position data
        state = RLState(
            price=position_data.get("price", 0.0),
            portfolio_value=position_data.get(
                "portfolio_value",
                0.0,
            ),
            delta=position_data.get("delta", 0.0),
            gamma=position_data.get("gamma", 0.0),
            theta=position_data.get("theta", 0.0),
            vega=position_data.get("vega", 0.0),
            iv=position_data.get("iv", 0.0),
            time_to_expiry=position_data.get("time_to_expiry", 0.0),
            vix=position_data.get("vix", 0.0),
            regime_encoded=position_data.get("regime_encoded", 0.0),
            unrealized_pnl_pct=position_data.get(
                "unrealized_pnl_pct",
                0.0,
            ),
            days_held=position_data.get("days_held", 0.0),
            max_profit_pct_reached=position_data.get(
                "max_profit_pct_reached",
                0.0,
            ),
        )

        action = self.predict_action(state)

        # Build reasoning string
        reasoning = _build_action_reasoning(
            action=action,
            state=state,
        )

        recommendation: dict[str, Any] = {
            "action": action.discrete_action.value,
            "confidence": action.confidence,
            "raw_value": action.raw_action,
            "reasoning": reasoning,
        }

        self._log.info(
            "position_recommendation",
            action=recommendation["action"],
            confidence=recommendation["confidence"],
            price=state.price,
            days_held=state.days_held,
            unrealized_pnl_pct=state.unrealized_pnl_pct,
        )

        return recommendation

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _create_model(self, env: gym.Env) -> BaseAlgorithm:
        """Instantiate the SB3 model with configured hyperparameters.

        Args:
            env: The Gymnasium environment to train on.

        Returns:
            An untrained SB3 model instance.
        """
        if self._config.algorithm == "ppo":
            from stable_baselines3 import PPO

            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=self._config.learning_rate,
                batch_size=self._config.batch_size,
                gamma=self._config.gamma,
                seed=self._config.seed,
                verbose=self._config.verbose,
                n_steps=2048,
                n_epochs=10,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs={
                    "net_arch": [256, 256],
                },
            )
            self._log.info(
                "ppo_model_created",
                learning_rate=self._config.learning_rate,
                batch_size=self._config.batch_size,
            )
            return model

        # Default: SAC
        from stable_baselines3 import SAC

        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=self._config.learning_rate,
            batch_size=self._config.batch_size,
            buffer_size=self._config.buffer_size,
            gamma=self._config.gamma,
            tau=self._config.tau,
            seed=self._config.seed,
            verbose=self._config.verbose,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs={
                "net_arch": [256, 256],
            },
        )
        self._log.info(
            "sac_model_created",
            learning_rate=self._config.learning_rate,
            batch_size=self._config.batch_size,
            buffer_size=self._config.buffer_size,
            tau=self._config.tau,
        )
        return model

    def _wrap_env(self, env: OptionsPositionEnv) -> gym.Env:
        """Wrap the environment with SB3-compatible wrappers.

        Applies a Monitor wrapper for episode statistics tracking
        and ensures the environment is compatible with SB3 training.

        Args:
            env: Raw options position environment.

        Returns:
            Wrapped Gymnasium environment.
        """
        from stable_baselines3.common.monitor import Monitor

        return Monitor(env)

    def _evaluate_episodes(
        self,
        episodes: list[list[dict[str, float]]],
    ) -> tuple[float, float]:
        """Evaluate the model on a set of episodes.

        Runs the agent through each episode deterministically and
        collects total rewards per episode.

        Args:
            episodes: List of episode state sequences.

        Returns:
            Tuple of (mean_reward, std_reward) across all episodes.
        """
        if self._model is None or not episodes:
            return 0.0, 0.0

        env_config = PositionEnvironmentConfig(
            target_dte=self._config.target_dte,
            max_drawdown_pct=self._config.max_drawdown_pct,
        )

        episode_rewards: list[float] = []

        for episode in episodes:
            env = OptionsPositionEnv(
                trade_episodes=[episode],
                config=env_config,
            )
            obs, _info = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action, _states = self._model.predict(
                    obs,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, _info = env.step(
                    action,
                )
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        return mean_reward, std_reward

    def _save_model(self) -> str:
        """Save the model with auto-generated filename.

        Returns:
            Absolute path to the saved model file.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"rl_{self._config.algorithm}_{timestamp}"
        path = self._model_dir / filename
        return self.save(str(path))


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def discretize_action(raw_action: float) -> PositionAction:
    """Map a continuous action value to a discrete position action.

    The continuous action space [-1, 1] is partitioned into seven
    discrete zones corresponding to position management actions.

    Args:
        raw_action: Continuous action value in [-1, 1].

    Returns:
        The corresponding :class:`PositionAction` enum member.
    """
    if raw_action <= ACTION_CLOSE_THRESHOLD:
        return PositionAction.CLOSE
    if raw_action <= ACTION_REDUCE_75_THRESHOLD:
        return PositionAction.REDUCE_75
    if raw_action <= ACTION_REDUCE_50_THRESHOLD:
        return PositionAction.REDUCE_50
    if raw_action <= ACTION_REDUCE_25_THRESHOLD:
        return PositionAction.REDUCE_25
    if raw_action <= ACTION_ADD_25_THRESHOLD:
        return PositionAction.HOLD
    if raw_action <= ACTION_ADD_50_THRESHOLD:
        return PositionAction.ADD_25
    return PositionAction.ADD_50


def _normalize_state_dict(
    state_dict: dict[str, float],
) -> np.ndarray:
    """Normalize a state dictionary to a numpy observation vector.

    Applies the same normalization as :meth:`OptionsPositionEnv._normalize_state`
    for consistency between training and inference.

    Args:
        state_dict: Raw state values keyed by field name.

    Returns:
        Normalized numpy array of shape (STATE_DIM,) in [-1, 1].
    """
    state = np.array(
        [
            state_dict.get("price", 0.0) / PRICE_MAX,
            state_dict.get("portfolio_value", 0.0) / PORTFOLIO_VALUE_MAX,
            state_dict.get("delta", 0.0) / DELTA_MAX,
            state_dict.get("gamma", 0.0) / GAMMA_MAX,
            state_dict.get("theta", 0.0) / THETA_MAX,
            state_dict.get("vega", 0.0) / VEGA_MAX,
            state_dict.get("iv", 0.0) / IV_MAX,
            state_dict.get("time_to_expiry", 0.0) / DTE_MAX,
            state_dict.get("vix", 0.0) / VIX_MAX,
            state_dict.get("regime_encoded", 0.0) / 3.0,
            state_dict.get("unrealized_pnl_pct", 0.0),
            state_dict.get("days_held", 0.0) / MAX_DAYS_HELD,
            state_dict.get("max_profit_pct_reached", 0.0),
        ],
        dtype=np.float32,
    )
    return np.clip(state, -1.0, 1.0)


def _build_action_reasoning(
    action: RLAction,
    state: RLState,
) -> str:
    """Build a human-readable reasoning string for an RL action.

    Generates a short explanation of why the agent chose the given
    action based on the current position state. This is included
    in recommendations for operator transparency.

    Args:
        action: The predicted RL action.
        state: The current position state.

    Returns:
        A reasoning string explaining the recommended action.
    """
    parts: list[str] = []

    # Action description
    action_descriptions = {
        PositionAction.CLOSE: "Close the position entirely",
        PositionAction.REDUCE_75: "Reduce position by 75%",
        PositionAction.REDUCE_50: "Reduce position by 50%",
        PositionAction.REDUCE_25: "Reduce position by 25%",
        PositionAction.HOLD: "Hold current position",
        PositionAction.ADD_25: "Add 25% to position",
        PositionAction.ADD_50: "Add 50% to position",
    }
    parts.append(f"Action: {action_descriptions[action.discrete_action]}.")

    # Time-based reasoning
    if state.time_to_expiry < 14:
        parts.append(f"Position nearing expiration ({state.time_to_expiry:.0f} DTE).")
    elif state.days_held > 30:
        parts.append(f"Position held for {state.days_held:.0f} days.")

    # P&L reasoning
    if state.unrealized_pnl_pct > 0.5:
        parts.append(f"Significant unrealized profit ({state.unrealized_pnl_pct:.1%}).")
    elif state.unrealized_pnl_pct < -0.3:
        parts.append(f"Position in drawdown ({state.unrealized_pnl_pct:.1%}).")

    # Profit reversal warning
    if state.max_profit_pct_reached > 0.1:
        given_back = state.max_profit_pct_reached - state.unrealized_pnl_pct
        if given_back > PROFIT_REVERSAL_THRESHOLD * state.max_profit_pct_reached:
            parts.append(
                f"Profit reversal detected: peak was "
                f"{state.max_profit_pct_reached:.1%}, "
                f"now {state.unrealized_pnl_pct:.1%}."
            )

    # Volatility context
    if state.vix > 30:
        parts.append(f"Elevated VIX ({state.vix:.1f}).")

    # Regime context
    regime_names = {
        0.0: "low-volatility trend",
        1.0: "high-volatility trend",
        2.0: "range-bound",
        3.0: "crisis",
    }
    regime_name = regime_names.get(
        state.regime_encoded,
        "unknown",
    )
    parts.append(f"Market regime: {regime_name}.")

    # Confidence note
    if action.confidence < 0.4:
        parts.append("Low confidence in this action; consider manual review.")

    return " ".join(parts)
