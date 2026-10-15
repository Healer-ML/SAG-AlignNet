import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_mechanism_variant(spec):
    """Parse the single CLI setting controlling both module families."""
    cprod = "native"
    masalign = "native"
    seen = set()
    value = (spec or "native").strip().lower()

    if value in ("", "native", "default"):
        return cprod, masalign

    allowed = {
        "cprod": {"native", "plot", "prior"},
        "masalign": {"native", "irra", "rasa"},
    }
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(
                "mechanism_variant must be 'native' or comma-separated "
                "'cprod=plot|prior,masalign=irra|rasa' settings"
            )
        key, selected = [part.strip() for part in item.split("=", 1)]
        if key not in allowed:
            raise ValueError(
                "unknown mechanism_variant key {!r}; use cprod or masalign".format(key)
            )
        if key in seen:
            raise ValueError("mechanism_variant repeats {!r}".format(key))
        if selected not in allowed[key]:
            raise ValueError(
                "unsupported {} variant {!r}; choose from {}".format(
                    key, selected, ", ".join(sorted(allowed[key]))
                )
            )
        seen.add(key)
        if key == "cprod":
            cprod = selected
        else:
            masalign = selected
    return cprod, masalign


class PLOTSlotAttention(nn.Module):
    """Iterative learnable-slot extractor following the local PLOT slot design."""

    def __init__(self, dim, iters=3, eps=1e-8):
        super().__init__()
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, max(dim, 128)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim, 128), dim),
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        for module in (self.to_q, self.to_k, self.to_v, self.mlp[0], self.mlp[2]):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs, initial_slots, padding_mask=None):
        batch, _, dim = inputs.shape
        slots = initial_slots.expand(batch, -1, -1).float()

        norm_inputs = self.norm_input(inputs.float()).to(self.to_k.weight.dtype)
        keys = self.to_k(norm_inputs)
        values = self.to_v(norm_inputs)

        for _ in range(self.iters):
            slots_prev = slots
            query_inputs = self.norm_slots(slots).to(self.to_q.weight.dtype)
            queries = self.to_q(query_inputs)
            logits = torch.bmm(queries.float(), keys.float().transpose(1, 2))
            attention = (logits * self.scale).softmax(dim=1)

            if padding_mask is not None:
                attention = attention.masked_fill(
                    padding_mask.unsqueeze(1), 0.0
                )

            attention = attention / (
                attention.sum(dim=-1, keepdim=True) + self.eps
            )
            updates = torch.bmm(attention, values.float())
            slots = self.gru(
                updates.reshape(-1, dim),
                slots_prev.reshape(-1, dim),
            ).reshape(batch, -1, dim)

            mlp_inputs = self.norm_pre_ff(slots).to(self.mlp[0].weight.dtype)
            slots = slots + self.mlp(mlp_inputs).float()

        return slots.to(inputs.dtype)


class PRIORPrototypeBank(nn.Module):
    """Gumbel vector-quantized prototypes following PRIOR's SPB mechanism."""

    def __init__(self, dim, num_prototypes, temperature=0.9):
        super().__init__()
        self.codebook = nn.Embedding(num_prototypes, dim)
        self.temperature = temperature

    def forward(self, inputs, valid_mask=None):
        features = inputs.float()
        codebook = self.codebook.weight.float()
        logits = torch.matmul(features, codebook.t())
        assignments = F.gumbel_softmax(
            logits, tau=self.temperature, hard=False, dim=-1
        )

        if valid_mask is None:
            valid = torch.ones(
                inputs.shape[:2], dtype=features.dtype, device=features.device
            )
        else:
            valid = valid_mask.to(device=features.device, dtype=features.dtype)

        assignments = assignments * valid.unsqueeze(-1)
        quantized = torch.matmul(assignments, codebook)
        reconstruction = (quantized - features).abs()
        reconstruction_loss = (
            (reconstruction * valid.unsqueeze(-1)).sum()
            / (valid.sum() * features.size(-1)).clamp_min(1.0)
        )

        counts = assignments.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        prototypes = torch.bmm(assignments.transpose(1, 2), quantized)
        prototypes = prototypes / counts
        return prototypes.to(inputs.dtype), reconstruction_loss
