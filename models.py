'''
    Contians implememntation for the CLIP model along with all the modules
    needed to make it work end to end.
'''
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from activation import QuickGELU
from config import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPEncoderConfig
from utils import contrastive_loss

@dataclass
class CLIPOutput:
    loss: float
    logits_per_text: torch.FloatTensor
    logits_per_image: torch.FloatTensor

class CLIPSelfAttention(nn.Module):
    '''
        Implements self atttention mechanism of the encoder module.
    '''
    def __init__(self, config:CLIPEncoderConfig):
        super().__init__()
        self.config = config

        self.head_dim = self.config.n_embed // self.config.n_heads

        assert self.head_dim * self.config.n_heads == self.config.n_embed,\
            f"Embedding dimensions [{self.config.n_embed}] are not divisble by n_heads [{self.config.n_heads}]."

        self.k_proj = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.v_proj = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.q_proj = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.out_proj = nn.Linear(self.config.n_embed, self.config.n_embed)

    def forward(self, hidden_state, attention_mask=None, apply_causal_mask=False):
        '''
            Forward pass for the self-attention of the transformer encoder.
        '''
        batch_size, seq_len, emb_dim = hidden_state.shape
        assert emb_dim == self.config.n_embed, f"Embedding dimensions mismatch! Expecting {self.config.n_embed}; Found: {emb_dim}."

        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)

        # reshape from (b, seq_len, emb_dim) => (b, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.config.n_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.config.n_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.config.n_heads, self.head_dim).transpose(1,2)

        # Create masked attention mask if the attention mask is specified.
        if attention_mask is not None:
            # Convert attention_mask from shape (B, seq_len) to (B, 1, seq_len, seq_len)
            attention_mask = attention_mask[:,None,None,:].expand(batch_size, 1, seq_len, seq_len).to(hidden_state.dtype)
            attention_mask = 1.0 - attention_mask
            attention_mask = attention_mask.masked_fill(attention_mask==1, torch.finfo(hidden_state.dtype).min)

        # Perform self attention using torch's scaled_dot_product_attention!
        attention_output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=apply_causal_mask)

        # convert the output shapre from (batch, n_heads, seq_len, head_dim) => (batch, seq_len, emd_dim)
        attention_output = attention_output.transpose(1,2).view(batch_size, seq_len, emb_dim)
        # final output projection.
        attention_output = self.out_proj(attention_output)
        return attention_output

class CLIPEncodeLayer(nn.Module):
    '''
        A single layer of the Encoder module. Encapsulates 
        self attention and feed forward modules of the encoder.
    '''
    def __init__(self, config:CLIPEncoderConfig):
        super().__init__()
        self.config = config

        self.self_attn = CLIPSelfAttention(config=config)
        self.layer_norm1 = nn.LayerNorm(self.config.n_embed, eps=self.config.layer_norm_eps)
        self.mlp = nn.ModuleDict(dict(
            fc1 = nn.Linear(self.config.n_embed, self.config.inter_dims),
            fc2 = nn.Linear(self.config.inter_dims, self.config.n_embed),
            activation = QuickGELU()
        ))
        self.layer_norm2 = nn.LayerNorm(self.config.n_embed, eps=self.config.layer_norm_eps)

    def forward(self, hidden_state, attention_mask=None, apply_causal_mask=False):
        '''
            Forward pass for the encoder later. Performs self attention followed by feedforward.
        '''
        # Layer norm followed by self attention. 
        attn_ln = self.layer_norm1(hidden_state)
        attn_output = self.self_attn(attn_ln, attention_mask, apply_causal_mask)

        # First residual connection for self attention.
        attn_output = (attn_output + hidden_state)
        
        # Second layer norm followed by Feed Forward.
        attn_ln2 = self.layer_norm2(attn_output)
        mlp_output = self.mlp.fc1(attn_ln2)
        mlp_output = self.mlp.activation(mlp_output)
        mlp_output = self.mlp.fc2(mlp_output)
        
        # Second residual for feed forward.
        mlp_output = (mlp_output + attn_output)

        return mlp_output
        

class CLIPEncoder(nn.Module):
    '''
        CLIPEncoder is shared between the TextTransformer and 
        VisionTransformer as the encoder remains the same once
        the text and images are converted into embeddings.
    '''
    def __init__(self, config:CLIPEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncodeLayer(config=config) for _ in range(self.config.n_layers)])
        
    def forward(self, hidden_state, attention_mask=None, apply_causal_mask=False):
        '''
            Forward pass for the Transformer Encoder. This is being used as both
            as Vision Encoder and Text Encoder.
        '''
        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask, apply_causal_mask)

        return hidden_state


class TextEmbeddings(nn.Module):
    '''
        Implementation of the text embedding including token 
        and position embeddings.
    '''
    def __init__(self, config:CLIPTextConfig):
        super().__init__()
        self.config = config

        # Token and Position embeddings. 
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        self.position_embedding = nn.Embedding(self.config.max_length, self.config.n_embed)

        self.register_buffer("position_ids",
            torch.arange(self.config.max_length).expand((1,-1)), persistent=False)

    def forward(self, input_ids):
        '''
            Forward pass for the text embeddings -> token_embeddings + position_embedding.
            `input_ids`: input tensor with the shape (batch_size, seq_len)
        '''
        batch_size, seq_len = input_ids.shape
        position_ids = self.position_ids[ : , :seq_len]

        tokn_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        text_embeddings = tokn_embeddings + position_embeddings
        return text_embeddings

class VisionEmbeddings(nn.Module):
    '''
        Implementation of vision embeddings that converts an image
        into patches and runs through the linear projection before
        adding positional embeddings.
    '''
    def __init__(self, config:CLIPVisionConfig):
        super().__init__()
        self.config = config

        # determine no. of patches.
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # Define patch embeddings, class embedding and position embeddings.
        self.class_embedding = nn.Parameter(torch.randn(self.config.n_embed))
        self.patch_embedding = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.config.n_embed,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size, 
            bias=False)
        self.position_embedding = nn.Embedding(self.num_positions, self.config.n_embed)

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values):
        '''
            Forward pass for vision embeddings.
            `pixel_values` are the batch images tensor of the shape (batch_size, channels, img_size, img_size)
        '''
        batch_size = pixel_values.shape[0]
        pixel_values = pixel_values.to(self.patch_embedding.weight.dtype)

        # Perform convolution to extract patches.
        patch_embeddings = self.patch_embedding(pixel_values)

        # reshape from (batch_size, n_embed, num_patches, num_patches) to
        # (batch_size, num_patches * num_patches, n_embed)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        # class embeddings.
        class_embeddings = self.class_embedding.expand(batch_size, 1, -1)

        # Position embeddings.
        position_embeddings = self.position_embedding(self.position_ids)
        
        # Combine patch embeddings with position embeddings to create vision embeddings.
        img_embeddings = torch.cat([class_embeddings, patch_embeddings], dim=1) + position_embeddings
        
        return img_embeddings

class TextTransformer(nn.Module):
    '''
        Text transformer to generate text embedding of the CLIP
        model architecture.
    '''
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config

        self.embeddings = TextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(self.config.n_embed, eps=self.config.layer_norm_eps)
    
    def forward(self, input_ids, attention_mask):
        '''
            forward pass for the text transformer.
        '''
        batch_size = input_ids.shape[0]

        emd_output = self.embeddings(input_ids)
        enc_output = self.encoder(emd_output, attention_mask, apply_causal_mask=True)
        encoder_output = self.final_layer_norm(enc_output)

        # We need to capture the eos token which semantically captures the 
        # embeddings for the entire sequence.
        encoder_output = encoder_output[torch.arange(batch_size), 
                    (input_ids.to(torch.int) == self.config.eos_token_id).int().argmax(dim=-1),
                    ]
        return encoder_output

class VisionTransformer(nn.Module):
    '''
        Vision transformer to generate image embedding of the CLIP
        model architecture.
    '''
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
    
        self.embeddings = VisionEmbeddings(config)
        # HF model has a type here. pre_layrnorm instead of pre_layernorm. 
        # Keeping this similar to HF's name as we will load the weights from HF model.
        self.pre_layrnorm = nn.LayerNorm(self.config.n_embed, eps=self.config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.config.n_embed, eps=self.config.layer_norm_eps)

    def forward(self, pixel_values):
        '''
            forward pass for the vision transformer.
        '''
        emd_output = self.embeddings(pixel_values)
        enc_output = self.pre_layrnorm(emd_output)
        enc_output = self.encoder(enc_output)
        # We need to grab the embeddings belonging to the first class 
        # token. Semantically, the first token captures the essence of the 
        # entire sequence.
        # HF refers to this as the pooled output.
        enc_output = enc_output[:, 0, :]
        enc_output = self.post_layernorm(enc_output)

        return enc_output


class CLIP(nn.Module):
    '''
        The main CLIP model class that encapsulates the entire model 
        architecture.
    '''
    def __init__(self, config:CLIPConfig):
        super().__init__()
        self.config = config
        self.text_config = config.text_config
        self.vision_config = config.vision_config
        
        # CLIP architecture modules for text and vision embeddings.
        self.text_model = TextTransformer(self.text_config)
        self.vision_model = VisionTransformer(self.vision_config)

        # Projection modules.
        self.visual_projection = nn.Linear(self.vision_config.n_embed, self.config.projection_dims, bias=False)
        self.text_projection = nn.Linear(self.text_config.n_embed, self.config.projection_dims, bias=False)

        # The learnable temperature parameter used to scale the logits.  Refer CLIP paper for refenrence.
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
    
    def forward(self, input_ids, pixel_values, attention_mask, return_loss=False):
        '''
            Forward pass for the CLIP model.
            `input_ids`: encoded text tokens of the shape (batch_size, seq_len).
            `pixel_values`: pixel values for the input image (batch_size, channel, img_size, img_size)
            `attention_mask`: attention mask for the input tokens of the shape (batch_size, seq_len)
            
        '''
        # Get the output of the Image and Text encoder models.
        vision_output = self.vision_model(pixel_values)
        text_output = self.text_model(input_ids, attention_mask)

        # Project the output of text and image encoder models into projection_dims.
        vision_output = self.visual_projection(vision_output)
        text_output = self.text_projection(text_output)
        

        # To compute the cosine similarity using dot product, we will normalize the outputs.
        text_output = text_output / text_output.norm(p=2, dim=-1, keepdim=True)
        vision_output = vision_output / vision_output.norm(p=2, dim=-1, keepdim=True)

        # compute the consine similarity using dot product.
        text_output = torch.matmul(text_output, vision_output.t()) * self.logit_scale.exp()
        vision_output = text_output.t()

        loss = None
        if return_loss:
            text_loss = contrastive_loss(text_output)
            vision_loss = contrastive_loss(vision_output)
            loss = (text_loss + vision_loss) / 2.0

        return CLIPOutput(loss=loss, logits_per_text=text_output, logits_per_image=vision_output)

    @classmethod
    def from_pretrained(cls, hf_model):
        '''
            Loads the weights from HF's pretrained models.
        '''
        # create an instance of out CLIP model.
        model = CLIP(CLIPConfig()).to(hf_model.dtype)

        # get the model's state / parameters.
        sd = model.state_dict()
        hf_sd = hf_model.state_dict()
        
        assert len(sd.keys()) == len(hf_sd.keys()), f"mismatch in model keys! expected: {len(sd.keys())}; found: {len(hf_sd.keys())}"

        # Copy the state from hf model to our model.
        for k in sd.keys():
            assert sd[k].shape == hf_sd[k].shape, f"Shape of the key: {k} didn't match!"
            assert sd[k].dtype == hf_sd[k].dtype, f"Type for the key: {k} didn't match!"

            with torch.no_grad():
                sd[k].copy_(hf_sd[k])
        
        return model