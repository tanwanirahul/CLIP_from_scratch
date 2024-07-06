'''
    Contains all the configurations for CLIP model implementation.
'''
from dataclasses import dataclass

@dataclass
class CLIPEncoderConfig:
    '''
        The common encoder configurations.
    '''
    projection_dims: int = 512
    n_embed: int = 768
    inter_dims: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-5

@dataclass
class CLIPTextConfig(CLIPEncoderConfig):
    '''
        Configuration for CLIP text encoder model.
    '''
    vocab_size: int = 49408
    max_length: int = 77        # Max position embeddings = 77
    n_embed: int = 512          # Embedding dimension - 512
    inter_dims: int = 2048
    n_heads: int = 8
    n_layers: int = 12
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407

@dataclass
class CLIPVisionConfig(CLIPEncoderConfig):
    '''
        Configuration for CLIP vision encoder model.
    '''
    projection_dims: int = 512
    n_embed: int = 768
    inter_dims: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    patch_size: int = 32        #Square Patch: 32 * 32
    image_size: int = 224       #Square Image: 224 * 224
    num_channels: int = 3

@dataclass
class CLIPConfig:
    '''
        Configurations for the CLIP model.
    '''
    text_config: CLIPTextConfig = CLIPTextConfig()
    vision_config: CLIPVisionConfig = CLIPVisionConfig()
    projection_dims: int = 512
    # The logit temperature parameter to scale logits as mentioned in the paper. 
    # The paper mentions the init value of 0.07. (1/np.exp(2.6592) => 0.07)
    logit_scale_init_value: float = 2.6592 