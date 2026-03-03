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
        """Compute SAC actor loss with RSI beta-grad scaling.

        Base SACLagRSI actor loss:
            L = alpha * log_pi(a|s) - Q_r(s,a) + lambda * beta * Q_c(s,a)

        and keeps the normalization / (1 + lambda) like PPOLag does.
        """
        # ----- standard SAC forward -----
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)

        q_r_1, q_r_2 = self._actor_critic.reward_critic(obs, action)
        q_r = torch.min(q_r_1, q_r_2)

        q_c = self._actor_critic.cost_critic(obs, action)[0]

        # penalty (lambda)
        penalty = self._lagrange.lagrangian_multiplier.detach()

        # ----- reward-only and cost-only losses (for beta-grad computation) -----
        # Reward part matches the original SACLag reward term:
        #   loss_r = alpha*log_pi - Q_r
        loss_r = (self._alpha * log_prob - q_r).mean()

        # Cost-only part should induce gradients proportional to Q_c term.
        # We use mean(Q_c) (NOT multiplied by lambda) so beta captures relative scales.
        loss_c = q_c.mean()

        # ----- compute beta_grad from reward-only vs cost-only grads -----
        actor_params = [p for p in self._actor_critic.actor.parameters() if p.requires_grad]

        g_r = torch.autograd.grad(loss_r, actor_params, retain_graph=True, create_graph=False)
        g_c = torch.autograd.grad(loss_c, actor_params, retain_graph=True, create_graph=False)

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
        # Base cost term is + lambda * Q_c (since we minimize loss)
        loss = loss_r + (penalty * beta_used * loss_c)
        loss = loss / (1.0 + penalty)

        # ----- logging  -----
        self._logger.store(
            {
                'Misc/BetaGrad': beta_used.item(),
                'Misc/GradNormR': norm_r.item(),
                'Misc/GradNormC': norm_c.item(),
                'Misc/Penalty': penalty.item(),
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