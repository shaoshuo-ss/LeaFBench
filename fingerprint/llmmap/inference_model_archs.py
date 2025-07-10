import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HP = {
    'num_blocks': 3,
    'feature_size': 384,
    'num_heads': 4,
    'activation': 'gelu',
    'with_add_dense_class': False,
    'emb_size': 1024,
    'num_queries': 8,
    'num_classes': 42,
}


def get_activation(name):
    if name == 'gelu':
        return F.gelu
    elif name == 'relu':
        return F.relu
    else:
        raise ValueError(f"Unsupported activation: {name}")


class ClassTokenLayer(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(1, 1, feature_size))

    def forward(self, x):
        batch_size = x.size(0)
        return self.class_token.expand(batch_size, -1, -1)


class TransformerBlock(nn.Module):
    def __init__(self, feature_size, num_heads, activation):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_size)
        self.attn = nn.MultiheadAttention(feature_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_size)
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, feature_size),
        )
        self.activation = get_activation(activation)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x_norm = self.norm2(x)
        m = self.mlp(x_norm)
        m = self.activation(m)
        x = x + m
        return x


class InferenceModelLLMmap(nn.Module):
    def __init__(self, hparams=DEFAULT_HP):
        super().__init__()
        self.hparams = hparams
        feature_size = hparams['feature_size']
        activation = hparams['activation']
        self.class_token_layer = ClassTokenLayer(feature_size)
        self.input_proj = nn.Linear(hparams['emb_size'] * 2, feature_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_size, hparams['num_heads'], activation)
            for _ in range(hparams['num_blocks'])
        ])
        self.activation = get_activation(activation)

    def forward(self, traces):
        # traces: (batch, num_queries, emb_size*2)
        class_token = self.class_token_layer(traces)  # (batch, 1, feature_size)
        traces_emb = self.input_proj(traces)  # (batch, num_queries, feature_size)
        x = torch.cat([class_token, traces_emb], dim=1)  # (batch, num_queries+1, feature_size)
        for block in self.transformer_blocks:
            x = block(x)
        x_cls = x[:, 0]  # (batch, feature_size)
        output = x_cls
        return output

