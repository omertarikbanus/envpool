"""PPO policy whose critic receives simulator-only privileged observations."""

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


DEFAULT_ACTOR_OBSERVATION_DIM = 54


class AsymmetricMlpExtractor(nn.Module):
    """Separate actor and critic MLPs with different input widths."""

    def __init__(self, feature_dim, actor_feature_dim, net_arch,
                 activation_fn, device):
        super().__init__()
        if not 0 < actor_feature_dim <= feature_dim:
            raise ValueError(
                "actor_feature_dim must be positive and no larger than the "
                "full observation dimension")
        if (isinstance(net_arch, list) and len(net_arch) == 1
                and isinstance(net_arch[0], dict)):
            net_arch = net_arch[0]
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get("pi", [])
            vf_layers = net_arch.get("vf", [])
        else:
            pi_layers = vf_layers = net_arch

        self.policy_net, self.latent_dim_pi = self._build(
            actor_feature_dim, pi_layers, activation_fn, device)
        self.value_net, self.latent_dim_vf = self._build(
            feature_dim, vf_layers, activation_fn, device)
        self.actor_feature_dim = actor_feature_dim

    @staticmethod
    def _build(input_dim, layer_dims, activation_fn, device):
        layers = []
        last_dim = input_dim
        for width in layer_dims:
            layers.extend((nn.Linear(last_dim, width), activation_fn()))
            last_dim = width
        return nn.Sequential(*layers).to(device), last_dim

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_feature_dim])

    def forward_critic(self, features):
        return self.value_net(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """Keep privileged trailing observations out of every actor code path."""

    def __init__(self, *args,
                 actor_obs_dim=DEFAULT_ACTOR_OBSERVATION_DIM, **kwargs):
        self.actor_obs_dim = actor_obs_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = AsymmetricMlpExtractor(
            self.features_dim,
            self.actor_obs_dim,
            self.net_arch,
            self.activation_fn,
            self.device,
        )
