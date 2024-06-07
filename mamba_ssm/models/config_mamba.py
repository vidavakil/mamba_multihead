from dataclasses import dataclass, field


@dataclass
class MambaConfig:

    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=lambda: {
        'convolved_v': True,
        'n_heads': 1,
        'd_state': 16,
        'dt_rank': 16,
        'scalar_dt': False,
        'dense_matrices': True,
        'complementary_b': False,
        'multi_head_proj': False
    })
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
