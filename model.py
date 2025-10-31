import torch
import torch.nn as nn
import timm
import os
from torch.nn import functional as F
from pytorch_wavelets import DWTForward

class Adapter(nn.Module):
    """Lightweight adapter for Vision Transformer blocks with prompt learning"""
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        reduction = 16
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.GELU()
        )
    
    def forward(self, x, rope=None):
        prompt = self.prompt_learn(x)
        prompted = x + prompt
        return self.block(prompted, rope=rope)


class UpBlock(nn.Module):
    """Basic upsampling block with double convolution"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class SimpleDecoder(nn.Module):
    """Simple U-Net style decoder with progressive upsampling"""
    def __init__(self, in_dim=1024): 
        super().__init__()
        self.up1 = UpBlock(in_dim, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.out_conv(x)
        return x



class BoundaryGuidedDecoder(nn.Module):
    """Boundary-guided dual-head decoder: boundary predicts first to enhance mask features"""
    def __init__(self, in_dim=1024):
        super().__init__()

        self.up1 = UpBlock(in_dim, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)

        self.boundary_head = nn.Conv2d(32, 1, 1)
        self.boundary_enhance = nn.Conv2d(1, 32, 3, padding=1)

        self.mask_head = nn.Conv2d(64, 1, 1)  # 32+32=64
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        shared = self.up4(x)  # [B, 32, 512, 512]
        
        boundary = self.boundary_head(shared)
        boundary_feat = self.boundary_enhance(torch.sigmoid(boundary))
        enhanced = torch.cat([shared, boundary_feat], dim=1)

        mask = self.mask_head(enhanced)
        
        return mask, boundary

class MultiScaleFrequencyExtraction(nn.Module):
    """Multi-Scale Frequency Extraction using wavelet decomposition"""
    def __init__(self, in_dim=1024, out_dim=512, wavelet='haar'):
        super().__init__()
        
        self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
        self.high_encoder = nn.Conv2d(in_dim * 3, out_dim, 1)
        self.low_encoder = nn.Conv2d(in_dim, out_dim, 1)
    
    def extract_freq_single_scale(self, feat):
        B, C, H, W = feat.shape
        
        # Wavelet decomposition
        with torch.amp.autocast('cuda', enabled=False): 
            feat_fp32 = feat.float()
            yL, yH = self.dwt(feat_fp32)
            
            H_dir = yH[0][:, :, 0, :, :]  # Horizontal
            V_dir = yH[0][:, :, 1, :, :]  # Vertical
            D_dir = yH[0][:, :, 2, :, :]  # Diagonal
            
            # Concatenate three directions
            freq_high = torch.cat([H_dir, V_dir, D_dir], dim=1)
        
        # Upsample back to original size
        freq_high_up = F.interpolate(freq_high, size=(H, W), mode='bilinear', align_corners=True)
        freq_low_up = F.interpolate(yL, size=(H, W), mode='bilinear', align_corners=True)
        
        high_encoded = self.high_encoder(freq_high_up)
        low_encoded = self.low_encoder(freq_low_up)
        
        return high_encoded, low_encoded
    
    def forward(self, feat):
        # Scale 1: Original scale 32×32
        high_32, low_32 = self.extract_freq_single_scale(feat)
        
        # Scale 2: Downsampled scale 16×16
        feat_16 = F.avg_pool2d(feat, kernel_size=2)
        high_16, _ = self.extract_freq_single_scale(feat_16)
        high_16_up = F.interpolate(high_16, size=feat.shape[-2:], mode='bilinear', align_corners=True)
        
        return {
            'high_fine': high_32,
            'high_coarse': high_16_up,
            'low_global': low_32
        }

class FrequencyBoundaryAlignment(nn.Module):
    """Frequency-Boundary Alignment Attention"""
    def __init__(self, freq_dim=512):
        super().__init__()
        
        # Boundary attention generator (using high-freq)
        self.boundary_attn_gen = nn.Sequential(
            nn.Conv2d(freq_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        # Structure attention generator (using low-freq)
        self.structure_attn_gen = nn.Sequential(
            nn.Conv2d(freq_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, freq_dict, spatial_feat):


        attn_boundary = self.boundary_attn_gen(freq_dict['high_fine'])
        attn_structure = self.structure_attn_gen(freq_dict['low_global'])
        
        combined_attn = self.alpha * attn_boundary + self.beta * attn_structure
        
        enhanced_feat = spatial_feat + 0.3 * spatial_feat * combined_attn
        
        return enhanced_feat
    

class BoundaryPrototypeGenerator(nn.Module):
    """Frequency-Guided Boundary Prototype generator"""
    def __init__(self, freq_dim=512, proto_dim=64):
        super().__init__()
        
        # Extract boundary features from high_fine and high_coarse
        self.boundary_extractor = nn.Sequential(
            nn.Conv2d(freq_dim * 2, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Generate prototype vectors
        self.proto_generator = nn.Sequential(
            nn.Conv2d(128, proto_dim, 1),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, freq_dict):

        high_concat = torch.cat([
            freq_dict['high_fine'],
            freq_dict['high_coarse']
        ], dim=1)
        
        # Extract boundary features
        boundary_feat = self.boundary_extractor(high_concat)
        
        # Generate prototypes
        boundary_proto = self.proto_generator(boundary_feat)
        
        return boundary_proto


class FrequencySpatialCrossAttention(nn.Module):
    """Cross-modal attention: use boundary prototypes to guide spatial features"""
    def __init__(self, spatial_dim=1024, proto_dim=64, num_heads=8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        
        # Query from spatial features
        self.query_proj = nn.Conv2d(spatial_dim, spatial_dim, 1)
        
        # Key/Value from prototypes
        self.key_proj = nn.Conv2d(proto_dim, spatial_dim, 1)
        self.value_proj = nn.Conv2d(proto_dim, spatial_dim, 1)
        
        self.out_proj = nn.Conv2d(spatial_dim, spatial_dim, 1)
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, spatial_feat, boundary_proto):

        B, C, H, W = spatial_feat.shape
        
        # Generate Q, K, V
        Q = self.query_proj(spatial_feat)  # [B, 1024, 32, 32]
        K = self.key_proj(boundary_proto)  # [B, 1024, 32, 32]
        V = self.value_proj(boundary_proto)  # [B, 1024, 32, 32]
        
        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, 8, 1024, 128]
        K = K.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        V = V.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        
        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V)  # [B, 8, 1024, 128]
        
        # Reshape back
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        
        out = self.out_proj(out)
        
        enhanced = spatial_feat + self.fusion_weight * out
        
        return enhanced



class BaselineModel(nn.Module): 
    def __init__(self, img_size=512, pretrained=True, freeze_backbone=True, use_msfe=False, use_fbaa=False, use_fgbp=False, use_cross_attn=False, wavelet='haar', decoder_type='simple'):   
        super().__init__()
        self.img_size = img_size
        self.use_msfe = use_msfe 
        self.use_fbaa = use_fbaa 
        self.use_fgbp = use_fgbp
        self.use_cross_attn = use_cross_attn
        self.decoder_type = decoder_type
        model_name = "hf_hub:timm/vit_large_patch16_dinov3.lvd1689m"
        print(f"Building {model_name} encoder (img_size={img_size})...")
        
        self.encoder = timm.create_model(
            model_name,
            img_size=img_size,
            pretrained=pretrained, 
            num_classes=0
        )
        if pretrained:
            self._verify_weights_loaded()
        
        blocks = []
        for block in self.encoder.blocks:  
            blocks.append(Adapter(block)) 
        self.encoder.blocks = nn.ModuleList(blocks)

        # Freeze backbone 
        if freeze_backbone:
            for name, param in self.encoder.named_parameters():
                if 'prompt_learn' not in name:
                    param.requires_grad = False

        feat_dim = self.encoder.blocks[0].block.attn.qkv.in_features
        print(f"Feature dimension: {feat_dim}")
        
        # Build frequency modules if enabled
        if use_msfe:
            self.msfe = MultiScaleFrequencyExtraction(feat_dim, 512, wavelet)
            self.freq_proj = nn.Conv2d(512, feat_dim, 1)
            print(f" MSFE enabled with wavelet={wavelet}")
            
            if use_fbaa:
                self.fbaa = FrequencyBoundaryAlignment(freq_dim=512)
                print(f" FBAA enabled")
            else:
                print(" FBAA disabled")
            if use_fgbp and use_cross_attn:
                self.fgbp = BoundaryPrototypeGenerator(freq_dim=512, proto_dim=64)
                self.cross_attn = FrequencySpatialCrossAttention(
                    spatial_dim=feat_dim, 
                    proto_dim=64, 
                    num_heads=8
                )
                print(f" FGBP + CrossAttention enabled")
            elif use_fgbp or use_cross_attn:
                print(" !!! FGBP and CrossAttn must be used together !!!")
        else:
            print(" MSFE disabled")
            if use_fbaa:
                print("!!! FBAA requires MSFE !!!")
            if use_fgbp or use_cross_attn:
                print("!!! FGBP/CrossAttn requires MSFE !!!")

        # Build decoder
        if decoder_type == 'simple':
            self.decoder = SimpleDecoder(in_dim=feat_dim)
            print(f" Using SimpleDecoder")
        elif decoder_type == 'boundary_guided':
            self.decoder = BoundaryGuidedDecoder(in_dim=feat_dim)
            print(f" Using BoundaryGuidedDecoder")
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        self._print_params()

    def _verify_weights_loaded(self):
        """Verify pretrained weights are correctly loaded"""
        print(f"\n🔍 Verifying pretrained weights...")
        last_block_idx = len(self.encoder.blocks) - 1
        key_params = {
            'patch_embed': None,
            'first_attn': None,
            'last_attn': None,
        }
        for name, param in self.encoder.named_parameters():
            if 'patch_embed.proj.weight' in name:
                key_params['patch_embed'] = param.std().item()
            elif 'blocks.0.attn.qkv.weight' in name:
                key_params['first_attn'] = param.std().item()
            elif f'blocks.{last_block_idx}.attn.qkv.weight' in name:
                key_params['last_attn'] = param.std().item()
        all_good = True
        for layer, std in key_params.items():
            if std is None:
                print(f" !!! {layer}: not found")
                all_good = False
            elif std < 0.01 or std > 0.3:
                print(f" !!! {layer}: std={std:.4f} (abnormal)")
                all_good = False
            else:
                print(f" ok!!! {layer}: std={std:.4f}")
        if all_good:
            print("Weights loaded successfully!")
        else:
            print("!!! Weights may have issues !!!")
        print()
    
    def _print_params(self):
        """Print parameter statistics for each module"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        msfe_params = 0
        if self.use_msfe:
            msfe_params = sum(p.numel() for p in self.msfe.parameters())
            msfe_params += sum(p.numel() for p in self.freq_proj.parameters())
            if self.use_fbaa and hasattr(self, 'fbaa'):
                fbaa_params = sum(p.numel() for p in self.fbaa.parameters())
            else:
                fbaa_params = 0
            fgbp_params = 0
            cross_attn_params = 0
            if self.use_fgbp and hasattr(self, 'fgbp'):
                fgbp_params = sum(p.numel() for p in self.fgbp.parameters())
            if self.use_cross_attn and hasattr(self, 'cross_attn'):
                cross_attn_params = sum(p.numel() for p in self.cross_attn.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print(f"Encoder: {encoder_params / 1e6:.2f}M")
        print(f"Decoder: {decoder_params / 1e6:.2f}M")
        if self.use_msfe:
            print(f"MSFE:    {msfe_params / 1e6:.2f}M")  
        if self.use_fbaa and fbaa_params > 0:  
            print(f"FBAA:    {fbaa_params / 1e6:.2f}M")
        if self.use_fgbp and fgbp_params > 0:
            print(f"FGBP:    {fgbp_params / 1e6:.2f}M")
        if self.use_cross_attn and cross_attn_params > 0:
            print(f"CrossAttn: {cross_attn_params / 1e6:.2f}M")
        print(f"Sub-total: {(encoder_params + decoder_params + msfe_params) / 1e6:.2f}M")
        print(f"Total: {total_params / 1e6:.2f}M")
        print(f"Trainable: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.1f}%)")
        print(f"Non-trainable: {non_trainable_params / 1e6:.2f}M ({100 * non_trainable_params / total_params:.1f}%)")

    def _tokens_to_map(self, x, grid_size):
        """Convert token sequence to 2D feature map"""
        B = x.shape[0]
        num_total_tokens = x.shape[1]
        expected_patch_tokens = grid_size ** 2
        num_special_tokens = num_total_tokens - expected_patch_tokens
        if not hasattr(self, '_token_info_printed'):
            print(f"\n🔍 Token info (first call):")
            print(f"  - Total tokens: {num_total_tokens}")
            print(f"  - Special tokens: {num_special_tokens}")
            print(f"  - Patch tokens: {expected_patch_tokens} ({grid_size}×{grid_size})")
            print(f"  - Input shape: {x.shape}")
            self._token_info_printed = True
        patch_tokens = x[:, num_special_tokens:, :]
        if not hasattr(self, '_patch_token_printed'):
            print(f"  - After removing special tokens: {patch_tokens.shape}")
            self._patch_token_printed = True
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, grid_size, grid_size)
        if not hasattr(self, '_feat_map_printed'):
            print(f"  - After converting to 2D: {feat_map.shape}")
            print()
            self._feat_map_printed = True
        return feat_map

    def forward(self, x):
        grid_size = self.img_size // 16
        final_tokens = self.encoder.forward_features(x)
        final_feat_map = self._tokens_to_map(final_tokens, grid_size)
        
        if self.use_msfe:
            # Step 1: MSFE extracts frequency features
            freq_dict = self.msfe(final_feat_map)
            
            # Step 2: If FBAA exists, use attention alignment
            if self.use_fbaa and hasattr(self, 'fbaa'):
                enhanced_feat = self.fbaa(freq_dict, final_feat_map)
            else:
                # Simple fusion without FBAA
                freq_enhanced = freq_dict['high_fine'] + freq_dict['high_coarse']
                freq_enhanced = self.freq_proj(freq_enhanced)
                enhanced_feat = final_feat_map + freq_enhanced
            
            # Step 3: If FGBP+CrossAttn exists, further refine
            if self.use_fgbp and self.use_cross_attn and hasattr(self, 'fgbp') and hasattr(self, 'cross_attn'):
                # Generate boundary prototypes
                boundary_proto = self.fgbp(freq_dict)
                # Refine features with CrossAttention
                enhanced_feat = self.cross_attn(enhanced_feat, boundary_proto)
            
            decoder_output = self.decoder(enhanced_feat)
        else:
            # Original flow without frequency modules
            decoder_output = self.decoder(final_feat_map)
        
        # Return based on decoder type
        if self.decoder_type == 'boundary_guided':
            mask, boundary = decoder_output
            return mask, boundary
        else:
            return decoder_output
