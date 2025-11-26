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
"""Implementation of the Lagrange version of the PPO algorithm (MODIFIED WITH REWARD SCALE INVARIANCE)."""

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange

from typing import Optional

def _global_grad_norm(grads):
    # grads: list[Tensor]
    return torch.sqrt(
        torch.sum(torch.stack([g.detach().pow(2).sum() for g in grads]))
    )



@registry.register
class PPOLagRSI(PPO):
    """The Lagrange version of the PPO algorithm. MODIFIED WITH REWARD SCALE INVARIANCE 

    A simple combination of the Lagrange method and the Proximal Policy Optimization algorithm.
    """

    def _init(self) -> None:
        """Initialize the PPOLag specific model.

        The PPOLag algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._last_adv_r = None
        self._last_adv_c = None
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        # ---- RSI beta-grad settings (add to yaml or hardcode) ----
        self._beta_grad_ema = getattr(self._cfgs.algo_cfgs, "beta_grad_ema", 0.9)
        self._beta_grad_eps = getattr(self._cfgs.algo_cfgs, "beta_grad_eps", 1e-8)
        self._beta_grad_clip = getattr(self._cfgs.algo_cfgs, "beta_grad_clip", None)
        self._beta_grad = None  # running value

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        """Signature-compatible PPO actor loss + beta-grad RSI."""

        # ----- standard PPO forward -----
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std

        ratio = torch.exp(logp_ - logp)
        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )

        # ----- retrieve cached reward/cost advantages -----
        adv_r = self._last_adv_r
        adv_c = self._last_adv_c
        if adv_r is None or adv_c is None:
            raise RuntimeError(
                "adv_r/adv_c not cached. "
                "Make sure _compute_adv_surrogate() runs before _loss_pi()."
            )

        # ----- compute beta_grad from reward-only vs cost-only grads -----
        loss_r = -(ratio * adv_r).mean()
        loss_c = -(ratio * adv_c).mean()

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
        if self._beta_grad_clip is not None:
            beta_used = torch.clamp(beta_used, 1.0 / self._beta_grad_clip, self._beta_grad_clip)

        # ----- re-build RSI mixed advantage and overwrite adv -----
        penalty = self._lagrange.lagrangian_multiplier.detach()
        adv_mix = (adv_r - penalty * beta_used * adv_c) / (1.0 + penalty)

        # ----- PPO-Clip loss on adv_mix -----
        loss = -torch.min(ratio * adv_mix, ratio_cliped * adv_mix).mean()

        # entropy bonus (same as base PPO)
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        # ----- logging (same keys style as base PPO + RSI extras) -----
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
                'Misc/BetaGrad': beta_used.item(),
                'Misc/GradNormR': norm_r.item(),
                'Misc/GradNormC': norm_c.item(),
                'Misc/Penalty': penalty.item(),
            },
        )

        return loss

    def _init_log(self) -> None:
        """Log the PPOLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
        self._logger.register_key('Misc/BetaGrad', min_and_max=True)
        self._logger.register_key('Misc/GradNormR', min_and_max=True)
        self._logger.register_key('Misc/GradNormC', min_and_max=True)
        self._logger.register_key('Misc/Penalty', min_and_max=True)

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        PPOLag uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        self._last_adv_r = adv_r
        self._last_adv_c = adv_c

        penalty = self._lagrange.lagrangian_multiplier.detach()
        # IMPORTANT: don't apply beta here anymore; _loss_pi will.
        return (adv_r - penalty * adv_c) / (1.0 + penalty)
