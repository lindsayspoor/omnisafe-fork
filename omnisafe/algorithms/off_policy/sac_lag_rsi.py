# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Lagrangian version of Soft Actor-Critic algorithm (MODIFIED WITH RSI)."""

from __future__ import annotations

from typing import Optional

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange


def _global_grad_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    """Compute global L2 norm of a list of gradient tensors."""
    return torch.sqrt(torch.sum(torch.stack([g.detach().pow(2).sum() for g in grads])))


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACLagRSI(SAC):
    """SAC with Lagrangian cost penalty, modified with Stooke20 Sec.7 reward-scale invariance (beta-grad).

    This modifies the actor loss by scaling the cost term with:

        beta = ||∇θ J|| / (||∇θ J_c|| + eps)

    where J is the reward objective and J_c is the cost objective, both w.r.t. actor parameters.
    """

    def _init(self) -> None:
        """Initialize SAC + Lagrange + RSI beta-grad state."""
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        # ---- RSI beta-grad settings (match PPOLagRSI style) ----
        self._beta_grad_ema = getattr(self._cfgs.algo_cfgs, "beta_grad_ema", 0.9)
        self._beta_grad_eps = getattr(self._cfgs.algo_cfgs, "beta_grad_eps", 1e-8)
        self._beta_grad_clip = getattr(self._cfgs.algo_cfgs, "beta_grad_clip", None)
        self._beta_grad_every = getattr(self._cfgs.algo_cfgs, "beta_grad_every", 50)
        self._beta_grad_batch = getattr(self._cfgs.algo_cfgs, "beta_grad_batch", 128)
        self._pi_update_count = 0
        self._beta_grad: Optional[torch.Tensor] = None  # running value (EMA)

    def _init_log(self) -> None:
        """Register SACLagRSI logging keys (match PPOLagRSI style)."""
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Misc/BetaGrad', min_and_max=True)
        self._logger.register_key('Misc/GradNormR', min_and_max=True)
        self._logger.register_key('Misc/GradNormC', min_and_max=True)
        self._logger.register_key('Misc/Penalty', min_and_max=True)

    def _update(self) -> None:
        """Update SAC components, then update Lagrange multiplier (same as SACLag)."""
        super()._update()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute SAC actor loss with RSI beta-grad scaling (optimized).

        Changes vs your original:
        - Recompute beta-grad only every `algo_cfgs.beta_grad_every` policy updates (default 50).
        - Compute beta-grad on a smaller sub-batch `algo_cfgs.beta_grad_batch` (default 128).
        - Avoid retaining autograd graph longer than needed (retain_graph only for first grad).
        - Reuse EMA beta between recomputations; default beta=1.0 until first compute.
        """
        # ----- standard SAC forward -----
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)

        q_r_1, q_r_2 = self._actor_critic.reward_critic(obs, action)
        q_r = torch.min(q_r_1, q_r_2)

        q_c = self._actor_critic.cost_critic(obs, action)[0]

        # penalty (lambda)
        penalty = self._lagrange.lagrangian_multiplier.detach()

        # reward-only and cost-only losses (for beta-grad computation)
        loss_r = (self._alpha * log_prob - q_r).mean()
        loss_c = q_c.mean()

        # ----- RSI beta-grad (throttled) -----
        # Read settings (safe defaults if not defined in cfgs)
        beta_grad_every = getattr(self._cfgs.algo_cfgs, "beta_grad_every", 50)
        beta_grad_batch = getattr(self._cfgs.algo_cfgs, "beta_grad_batch", 128)

        # Initialize/update a counter (don’t assume _init was patched)
        if not hasattr(self, "_pi_update_count"):
            self._pi_update_count = 0
        self._pi_update_count += 1

        # Default beta_used: previous EMA value, else 1.0 until first compute
        if self._beta_grad is None:
            beta_used = torch.tensor(1.0, device=obs.device)
        else:
            beta_used = self._beta_grad

        do_recompute = (self._beta_grad is None) or (beta_grad_every > 0 and self._pi_update_count % beta_grad_every == 0)

        norm_r = torch.tensor(float("nan"), device=obs.device)
        norm_c = torch.tensor(float("nan"), device=obs.device)

        if do_recompute:
            # Use a smaller sub-batch for beta-grad to reduce overhead
            obs_beta = obs
            if beta_grad_batch is not None and beta_grad_batch > 0 and obs.shape[0] > beta_grad_batch:
                idx = torch.randint(0, obs.shape[0], (beta_grad_batch,), device=obs.device)
                obs_beta = obs[idx]

            # Recompute losses on obs_beta (so grads correspond to the sub-batch)
            action_b = self._actor_critic.actor.predict(obs_beta, deterministic=False)
            log_prob_b = self._actor_critic.actor.log_prob(action_b)

            q_r_1_b, q_r_2_b = self._actor_critic.reward_critic(obs_beta, action_b)
            q_r_b = torch.min(q_r_1_b, q_r_2_b)
            q_c_b = self._actor_critic.cost_critic(obs_beta, action_b)[0]

            loss_r_b = (self._alpha * log_prob_b - q_r_b).mean()
            loss_c_b = q_c_b.mean()

            actor_params = [p for p in self._actor_critic.actor.parameters() if p.requires_grad]

            # Retain graph only until both grads are extracted
            g_r = torch.autograd.grad(loss_r_b, actor_params, retain_graph=True, create_graph=False)
            g_c = torch.autograd.grad(loss_c_b, actor_params, retain_graph=False, create_graph=False)

            norm_r = _global_grad_norm(g_r)
            norm_c = _global_grad_norm(g_c)

            beta = (norm_r / (norm_c + self._beta_grad_eps)).detach()

            # EMA smoothing
            if self._beta_grad is None:
                self._beta_grad = beta
            else:
                self._beta_grad = self._beta_grad_ema * self._beta_grad + (1 - self._beta_grad_ema) * beta

            beta_used = self._beta_grad

        # Optional clipping for stability
        if self._beta_grad_clip is not None:
            beta_used = torch.clamp(beta_used, 1.0 / self._beta_grad_clip, self._beta_grad_clip)

        # ----- final RSI-mixed SAC-Lagrangian loss -----
        loss = loss_r + (penalty * beta_used * loss_c)
        loss = loss / (1.0 + penalty)

        # ----- logging -----
        # If we didn't recompute norms this step, log last known values if available.
        if not do_recompute and hasattr(self, "_last_norm_r") and hasattr(self, "_last_norm_c"):
            norm_r = self._last_norm_r
            norm_c = self._last_norm_c
        else:
            # Store latest norms if they are finite
            self._last_norm_r = norm_r.detach()
            self._last_norm_c = norm_c.detach()

        self._logger.store(
            {
                'Misc/BetaGrad': float(beta_used.item()),
                'Misc/GradNormR': float(norm_r.item()) if torch.isfinite(norm_r) else float("nan"),
                'Misc/GradNormC': float(norm_c.item()) if torch.isfinite(norm_c) else float("nan"),
                'Misc/Penalty': float(penalty.item()),
            },
        )

        return loss

    def _log_when_not_update(self) -> None:
        """Keep logging Lagrange multiplier even during warmup/non-update epochs (same as SACLag)."""
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )