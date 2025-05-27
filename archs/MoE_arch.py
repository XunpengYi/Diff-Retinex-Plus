import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from basicsr.utils import get_root_logger, imwrite, tensor2img
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates.view(self._nonzero_gates.size(0), 1, 1, 1)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3), requires_grad=True, device=stitched.device)

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
class Conv_Expert(nn.Module):
    def __init__(self, dim=36, ffn_expansion_factor=2):
        super(Conv_Expert, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim * ffn_expansion_factor, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim * ffn_expansion_factor, dim, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.gelu(out)
        out = self.conv2(out)
        return out

class MoE_layer(nn.Module):
    def __init__(self, num_experts=6, expert_dim=36, ffn_expansion_factor=2, noisy_gating=True, k=2):
        super(MoE_layer, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.k = k

        # instantiate experts
        self.experts = nn.ModuleList([Conv_Expert(dim=self.expert_dim, ffn_expansion_factor=ffn_expansion_factor) for i in range(self.num_experts)])
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.meanpool = nn.AdaptiveMaxPool2d(1)
        self.gate = nn.Sequential(nn.Linear(expert_dim * 2, num_experts))
        self.noise = nn.Linear(expert_dim * 2, num_experts)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        b, _, _, _ = x.shape
        max_vector = self.maxpool(x).view(b, -1)
        mean_vector = self.meanpool(x).view(b, -1)
        x_vector = torch.cat((max_vector, mean_vector), dim=1)

        clean_logits = self.gate(x_vector)
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x_vector)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits


        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)

        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):

        gates, load = self.noisy_top_k_gating(x, self.training)

        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # 2. batch dispatcher
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        return y, loss

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            noise_feature = self.noise_func(noise_embed).view(batch, -1)
            x = x + noise_feature
        return x

class MoE_layer_time(nn.Module):
    def __init__(self, num_experts=6, expert_dim=36, time_dim=32, ffn_expansion_factor=2, noisy_gating=True, k=2):
        super(MoE_layer_time, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.k = k

        self.time_embedding = FeatureWiseAffine(time_dim, expert_dim * 2)
        # instantiate experts
        self.experts = nn.ModuleList([Conv_Expert(dim=self.expert_dim, ffn_expansion_factor=ffn_expansion_factor) for i in range(self.num_experts)])
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.meanpool = nn.AdaptiveMaxPool2d(1)
        self.gate = nn.Sequential(nn.Linear(expert_dim * 2, expert_dim * 2),
                                  nn.LeakyReLU(),
                                  nn.Linear(expert_dim * 2, num_experts))

        self.noise = nn.Linear(expert_dim * 2, num_experts)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, time, train, noise_epsilon=1e-2):
        b, _, _, _ = x.shape
        max_vector = self.maxpool(x).view(b, -1)
        mean_vector = self.meanpool(x).view(b, -1)
        x_vector = torch.cat((max_vector, mean_vector), dim=1)

        x_vector = self.time_embedding(x_vector, time)

        clean_logits = self.gate(x_vector)
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x_vector)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits


        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)

        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, time, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, time, self.training)

        # 1. calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # 2. batch dispatcher
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        return y, loss