# multimodal_utils.py

import torch
import torch.nn as nn

def project_pooled_embed(source_embed: torch.Tensor, projection_layer: nn.Linear) -> torch.Tensor:
    if source_embed.dim() != 2:
        raise ValueError(f"Expected a 2D tensor (B, D_in), but got {source_embed.dim()}D tensor.")
    return projection_layer(source_embed)

def project_sequence_embed(source_embed: torch.Tensor, projection_layer: nn.Linear) -> torch.Tensor:
    if source_embed.dim() != 3:
        raise ValueError(f"Expected a 3D tensor (B, L, D_in), but got {source_embed.dim()}D tensor.")
    return projection_layer(source_embed)

def create_attention_mask_from_embed(embed_sequence: torch.Tensor) -> torch.Tensor:
    if embed_sequence.dim() != 3:
        raise ValueError(f"Expected a 3D tensor (B, L, D), but got {embed_sequence.dim()}D tensor.")
    batch_size, seq_length, _ = embed_sequence.shape
    return torch.ones(batch_size, seq_length, dtype=torch.long, device=embed_sequence.device)

if __name__ == '__main__':
    BATCH_SIZE, SEQ_LENGTH, VISION_DIM, LLM_DIM = 4, 50, 768, 4096
    print("--- Testing Pooled Embed Projection ---")
    dummy_pooled_embed = torch.randn(BATCH_SIZE, VISION_DIM)
    pooled_projector = nn.Linear(VISION_DIM, LLM_DIM)
    projected_pooled = project_pooled_embed(dummy_pooled_embed, pooled_projector)
    print(f"Source shape: {dummy_pooled_embed.shape}, Projected shape: {projected_pooled.shape}\n")

    print("--- Testing Sequence Embed Projection ---")
    dummy_sequence_embed = torch.randn(BATCH_SIZE, SEQ_LENGTH, VISION_DIM)
    sequence_projector = nn.Linear(VISION_DIM, LLM_DIM)
    projected_sequence = project_sequence_embed(dummy_sequence_embed, sequence_projector)
    print(f"Source shape: {dummy_sequence_embed.shape}, Projected shape: {projected_sequence.shape}\n")

    print("--- Testing Attention Mask Creation ---")
    attention_mask = create_attention_mask_from_embed(projected_sequence)
    print(f"Source embed shape: {projected_sequence.shape}, Created mask shape: {attention_mask.shape}")