import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool

from . import neuron


class EntropyDrivenPartitioner(nn.Module):
    def __init__(
        self,
        max_segments,
        segment_length,
        window_size=25,
        min_segment_length=8,
        alpha_init=1.0,
        lambda_entropy=1.0,
        lambda_domain=0.5,
    ):
        super(EntropyDrivenPartitioner, self).__init__()
        self.max_segments = max_segments
        self.segment_length = segment_length
        self.window_size = window_size
        self.min_segment_length = min_segment_length
        self.lambda_entropy = lambda_entropy
        self.lambda_domain = lambda_domain
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, target, pfam_domains=None):
        if target.dim() == 3 and target.size(1) == 1:
            target = target.squeeze(1)

        segments = []
        segment_masks = []
        batch_size = target.size(0)

        for batch_idx in range(batch_size):
            domains = self._get_domains_for_sample(pfam_domains, batch_idx)
            sample_segments = self._partition_one(target[batch_idx], domains)
            packed_segments, packed_mask = self._pack_segments(sample_segments, target.device)
            segments.append(packed_segments)
            segment_masks.append(packed_mask)

        return torch.stack(segments, dim=0), torch.stack(segment_masks, dim=0)

    def _partition_one(self, target, domains):
        valid_length = int(target.gt(0).sum().item())
        if valid_length == 0:
            return [target.new_zeros(self.segment_length)]

        sequence = target[:valid_length]
        boundaries = self._select_boundaries(sequence, domains)
        starts = [0] + boundaries
        ends = boundaries + [valid_length]

        segments = []
        for start, end in zip(starts, ends):
            if end > start:
                segments.append(self._resample_segment(sequence[start:end]))

        return segments[: self.max_segments] or [self._resample_segment(sequence)]

    def _select_boundaries(self, sequence, domains):
        valid_length = sequence.size(0)
        gradients, positions = self._entropy_gradients(sequence)
        candidates = []

        if gradients.numel() > 0:
            grad_mean = gradients.mean()
            grad_std = gradients.std(unbiased=False)
            threshold = grad_mean + self.alpha.clamp_min(0.0) * grad_std
            threshold_mask = gradients >= threshold

            if threshold_mask.any():
                boundary_positions = positions[threshold_mask]
                boundary_scores = gradients[threshold_mask]
            else:
                fallback_count = min(
                    self.max_segments - 1,
                    max(1, math.ceil(valid_length / self.segment_length) - 1),
                    gradients.numel(),
                )
                boundary_scores, top_indices = torch.topk(gradients, k=fallback_count)
                boundary_positions = positions[top_indices]

            for boundary, score in zip(boundary_positions, boundary_scores):
                candidates.append((int(boundary.item()), float(score.detach().item())))

        for left, right in self._normalize_domains(domains, valid_length):
            for boundary in (left, right):
                if 0 < boundary < valid_length:
                    entropy_score = self._nearest_entropy_score(boundary, gradients, positions)
                    domain_score = self._length_consistency(boundary, left, right, valid_length)
                    score = self.lambda_entropy * entropy_score + self.lambda_domain * domain_score
                    candidates.append((boundary, score))

        return self._choose_boundaries(candidates, valid_length)

    def _entropy_gradients(self, sequence):
        valid_length = sequence.size(0)
        window_size = min(self.window_size, valid_length)
        if valid_length <= window_size:
            empty_float = sequence.new_empty(0, dtype=torch.float32)
            empty_long = sequence.new_empty(0, dtype=torch.long)
            return empty_float, empty_long

        entropy_values = []
        for start in range(valid_length - window_size + 1):
            window = sequence[start : start + window_size]
            entropy_values.append(self._shannon_entropy(window))

        entropies = torch.stack(entropy_values)
        gradients = torch.abs(entropies[1:] - entropies[:-1])
        positions = torch.arange(
            1,
            gradients.numel() + 1,
            device=sequence.device,
            dtype=torch.long,
        )
        positions = (positions + window_size // 2).clamp(1, valid_length - 1)
        return gradients, positions

    def _shannon_entropy(self, window):
        residues = window[window > 0]
        if residues.numel() == 0:
            return window.new_tensor(0.0, dtype=torch.float32)

        counts = torch.bincount(residues.long()).float()
        probabilities = counts[counts > 0] / residues.numel()
        return -(probabilities * torch.log(probabilities + 1e-8)).sum()

    def _nearest_entropy_score(self, boundary, gradients, positions):
        if gradients.numel() == 0:
            return 0.0

        nearest_index = torch.argmin(torch.abs(positions - boundary))
        return float(gradients[nearest_index].detach().item())

    def _length_consistency(self, boundary, domain_left, domain_right, valid_length):
        domain_length = max(1, domain_right - domain_left)
        local_left = max(0, boundary - self.segment_length // 2)
        local_right = min(valid_length, boundary + self.segment_length // 2)
        local_length = max(1, local_right - local_left)
        return 1.0 - abs(domain_length - local_length) / max(domain_length, local_length)

    def _choose_boundaries(self, candidates, valid_length):
        selected = []
        ordered_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)

        for boundary, _ in ordered_candidates:
            if len(selected) >= self.max_segments - 1:
                break
            if boundary < self.min_segment_length or valid_length - boundary < self.min_segment_length:
                continue
            if any(abs(boundary - chosen) < self.min_segment_length for chosen in selected):
                continue
            selected.append(boundary)

        return sorted(set(selected))

    def _resample_segment(self, segment):
        if segment.numel() == 0:
            return segment.new_zeros(self.segment_length)

        if segment.numel() >= self.segment_length:
            indices = torch.linspace(
                0,
                segment.numel() - 1,
                steps=self.segment_length,
                device=segment.device,
            ).round().long()
            return segment[indices]

        padded_segment = segment.new_zeros(self.segment_length)
        padded_segment[: segment.numel()] = segment
        return padded_segment

    def _pack_segments(self, segments, device):
        packed = torch.zeros(
            self.max_segments,
            self.segment_length,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros(self.max_segments, dtype=torch.bool, device=device)

        for index, segment in enumerate(segments[: self.max_segments]):
            packed[index] = segment
            mask[index] = True

        return packed, mask

    def _get_domains_for_sample(self, pfam_domains, batch_idx):
        if pfam_domains is None:
            return None
        if torch.is_tensor(pfam_domains):
            return pfam_domains[batch_idx]
        return pfam_domains[batch_idx]

    def _normalize_domains(self, domains, valid_length):
        if domains is None:
            return []
        if torch.is_tensor(domains):
            domains = domains.detach().cpu().tolist()

        normalized_domains = []
        for domain in domains:
            if len(domain) < 2:
                continue
            left = max(0, min(valid_length, int(domain[0])))
            right = max(0, min(valid_length, int(domain[1])))
            if right > left:
                normalized_domains.append((left, right))
        return normalized_domains


class GINConvNet(torch.nn.Module):
    def __init__(
        self,
        n_output=1,
        num_features_xd=78,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.2,
        time_steps=40,
        chunk_size=25,
        conv_kernel_size=8,
    ):
        super(GINConvNet, self).__init__()

        self.n_output = n_output
        self.time_steps = time_steps
        self.chunk_size = chunk_size
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.dps_partitioner = EntropyDrivenPartitioner(
            max_segments=time_steps,
            segment_length=chunk_size,
            window_size=chunk_size,
        )
        self.lif_neuron = neuron.LIF(tau=1.0, alpha=1.0, surrogate="triangle")

        graph_hidden_dim = 32
        self.gin_conv1 = GINConv(
            Sequential(
                Linear(num_features_xd, graph_hidden_dim),
                ReLU(),
                Linear(graph_hidden_dim, graph_hidden_dim),
            )
        )
        self.graph_bn1 = nn.BatchNorm1d(graph_hidden_dim)

        self.gin_conv2 = GINConv(
            Sequential(
                Linear(graph_hidden_dim, graph_hidden_dim),
                ReLU(),
                Linear(graph_hidden_dim, graph_hidden_dim),
            )
        )
        self.graph_bn2 = nn.BatchNorm1d(graph_hidden_dim)

        self.gin_conv3 = GINConv(
            Sequential(
                Linear(graph_hidden_dim, graph_hidden_dim),
                ReLU(),
                Linear(graph_hidden_dim, graph_hidden_dim),
            )
        )
        self.graph_bn3 = nn.BatchNorm1d(graph_hidden_dim)

        self.gin_conv4 = GINConv(
            Sequential(
                Linear(graph_hidden_dim, graph_hidden_dim),
                ReLU(),
                Linear(graph_hidden_dim, graph_hidden_dim),
            )
        )
        self.graph_bn4 = nn.BatchNorm1d(graph_hidden_dim)

        self.gin_conv5 = GINConv(
            Sequential(
                Linear(graph_hidden_dim, graph_hidden_dim),
                ReLU(),
                Linear(graph_hidden_dim, graph_hidden_dim),
            )
        )
        self.graph_bn5 = nn.BatchNorm1d(graph_hidden_dim)

        self.graph_fc = Linear(graph_hidden_dim, output_dim)

        self.register_buffer(
            "canonical_token_map",
            self._build_canonical_token_map(num_features_xt),
        )
        self.residue_projection = nn.Linear(20, embed_dim, bias=False)
        self.protein_conv = nn.Conv1d(
            in_channels=chunk_size,
            out_channels=n_filters,
            kernel_size=conv_kernel_size,
        )
        protein_conv_dim = n_filters * (embed_dim - conv_kernel_size + 1)
        self.spike_pooling = nn.Linear(time_steps * protein_conv_dim, protein_conv_dim)
        self.protein_fc = nn.Linear(protein_conv_dim, output_dim)

        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        drug_features = self.encode_drug(data.x, data.edge_index, data.batch)
        protein_features = self.encode_protein(
            data.target,
            getattr(data, "pfam_domains", None),
        )

        combined = torch.cat((drug_features, protein_features), dim=1)
        combined = self.dropout(self.relu(self.fc1(combined)))
        combined = self.dropout(self.relu(self.fc2(combined)))
        return self.out(combined)

    def encode_drug(self, x, edge_index, batch):
        x = self.relu(self.gin_conv1(x, edge_index))
        x = self.graph_bn1(x)
        x = self.relu(self.gin_conv2(x, edge_index))
        x = self.graph_bn2(x)
        x = self.relu(self.gin_conv3(x, edge_index))
        x = self.graph_bn3(x)
        x = self.relu(self.gin_conv4(x, edge_index))
        x = self.graph_bn4(x)
        x = self.relu(self.gin_conv5(x, edge_index))
        x = self.graph_bn5(x)

        x = global_add_pool(x, batch)
        x = self.relu(self.graph_fc(x))
        return F.dropout(x, p=0.2, training=self.training)

    def encode_protein(self, target, pfam_domains=None):
        dps_segments, segment_mask = self.dps_partitioner(target, pfam_domains)
        residue_one_hot = self._encode_residues(dps_segments)
        embedded_segments = self.residue_projection(residue_one_hot)
        protein_spikes = []

        for segment_index in range(self.time_steps):
            segment_embedding = embedded_segments[:, segment_index]
            active_segment = segment_mask[:, segment_index].view(-1, 1, 1).float()
            conv_output = self.protein_conv(segment_embedding) * active_segment
            spike_output = self.lif_neuron(conv_output)
            spike_output = spike_output * active_segment
            protein_spikes.append(spike_output.reshape(spike_output.size(0), -1))

        protein_spikes = torch.cat(protein_spikes, dim=1)
        protein_spikes = self.spike_pooling(protein_spikes)
        protein_spikes = self.dropout(self.relu(protein_spikes))
        protein_features = self.protein_fc(protein_spikes)

        neuron.reset_net(self)
        return protein_features

    def _encode_residues(self, token_segments):
        token_ids = token_segments.clamp(0, self.canonical_token_map.numel() - 1)
        canonical_ids = self.canonical_token_map[token_ids]
        known_residue_mask = canonical_ids.ge(0).unsqueeze(-1)
        one_hot = F.one_hot(canonical_ids.clamp_min(0), num_classes=20).float()
        return one_hot * known_residue_mask.float()

    @staticmethod
    def _build_canonical_token_map(num_features_xt):
        seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        canonical_residues = "ACDEFGHIKLMNPQRSTVWY"
        token_map = torch.full((num_features_xt + 1,), -1, dtype=torch.long)

        for token_id, residue in enumerate(seq_voc[:num_features_xt], start=1):
            if residue in canonical_residues:
                token_map[token_id] = canonical_residues.index(residue)

        return token_map
