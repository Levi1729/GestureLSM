import os
import pdb
import math
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from models.layers.layer import BasicBlock
from models.wavlm.WavLM import WavLM, WavLMConfig


class ExactLengthAdjuster(nn.Module):
    """
    Layer that ensures the output has exactly the target length along the time dimension.
    It either adds or removes frames as needed.
    """
    def __init__(self, target_length=196):
        super(ExactLengthAdjuster, self).__init__()
        self.target_length = target_length
    
    def forward(self, x):
        # x is expected to be [batch, channels, time]
        current_length = x.shape[2]
        
        if current_length == self.target_length:
            return x
        elif current_length < self.target_length:
            # Need to add frames
            frames_to_add = self.target_length - current_length
            
            # Duplicate the last frame as many times as needed
            last_frame = x[:, :, -1:]
            extra_frames = last_frame.repeat(1, 1, frames_to_add)
            
            return torch.cat([x, extra_frames], dim=2)
        else:
            # Need to remove frames
            # Just truncate to the target length
            return x[:, :, :self.target_length]


class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=2, target_length=256):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1700, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
        self.length_adjuster = ExactLengthAdjuster(target_length=target_length)
    
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        out = self.length_adjuster(out)
        
        return out.transpose(1, 2)


class ModalityEncoder(nn.Module):
    def __init__(self, 
                 data_path, 
                 t_fix_pre, 
                 audio_dim, 
                 audio_in=2,
                 raw_audio=False,
                 latent_dim=256,
                 audio_fps=30,
                 use_exp=False,
                 target_length=256,
                 spatial_temporal=False
                 ):
        super().__init__()
        
        self.raw_audio = raw_audio
        self.latent_dim = latent_dim
        self.audio_fps = audio_fps
        

        self.WavEncoder = WavEncoder(audio_dim, audio_in=audio_in, target_length=target_length)
        self.text_encoder_body = nn.Linear(300, audio_dim) 

        vocab_path = f"{data_path}weights/vocab.pkl"
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        else:
            logger.warning(f"vocab.pkl not found at {vocab_path}, using zeroed fallback embedding")
            fallback_weights = np.zeros((2, 300), dtype=np.float32)
            self.lang_model = SimpleNamespace(
                PAD_token=0,
                UNK_token=1,
                word_embedding_weights=fallback_weights,
            )
            pre_trained_embedding = fallback_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=t_fix_pre)
        word_dim = pre_trained_embedding.shape[1]

        if self.raw_audio:
            # load the pre-trained wavlm model
            # self.load_and_freeze_wavlm()
            self.audio_projection = nn.Linear(1024, audio_dim)

        joint_multiplier = 4 if use_exp else 3
        self.context_dim = self.latent_dim * joint_multiplier

        mix_input_dim = audio_dim * 3 if self.raw_audio else audio_dim * 2
        self.mix_audio_text = nn.Linear(mix_input_dim, self.context_dim)

    
    def forward(self, audio, word, raw_audio=None, squeeze_scale=4):
        # Initial features extraction - single transpose each
        # [B, T, D] -> [T, B, D]
        audio_feat = self.WavEncoder(audio)
        text_emb = self.text_pre_encoder_body(word)
        text_feat = self.text_encoder_body(text_emb)

        audio_len = audio_feat.shape[1]
        text_len = text_feat.shape[1]

        if audio_len != text_len:
            target_len = text_len if text_len > 0 else audio_len
            if target_len == 0:
                logger.warning("Both audio and text sequences are empty; inserting single-frame zeros")
                audio_feat = audio_feat.new_zeros(audio_feat.shape[0], 1, audio_feat.shape[2])
                text_feat = text_feat.new_zeros(text_feat.shape[0], 1, text_feat.shape[2])
            else:
                if audio_len == 0:
                    audio_feat = audio_feat.new_zeros(text_feat.shape[0], target_len, audio_feat.shape[2])
                else:
                    audio_feat = F.interpolate(
                        audio_feat.transpose(1, 2),
                        size=target_len,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)

                if text_len == 0:
                    text_feat = text_feat.new_zeros(audio_feat.shape[0], target_len, text_feat.shape[2])
                else:
                    text_feat = F.interpolate(
                        text_feat.transpose(1, 2),
                        size=target_len,
                        mode="nearest",
                    ).transpose(1, 2)

                logger.warning(
                    "Resampled modality features for length mismatch (audio=%d, text=%d -> %d)",
                    audio_len,
                    text_len,
                    target_len,
                )
        if raw_audio is not None and self.raw_audio:
            # Keep the same transpose pattern for consistency
            # raw_feat = self.extract_wavlm_feats(raw_audio)
            raw_feat = self.audio_projection(raw_audio)
            
            at_feat = torch.cat([audio_feat, raw_feat, text_feat], dim=2)
        else:
            at_feat = torch.cat([audio_feat, text_feat], dim=2)  # [B, T, D]
        
        at_feat = self.mix_audio_text(at_feat)  # [B, T, D']
        
        at_feat = F.avg_pool1d(at_feat.transpose(1, 2), squeeze_scale)
        at_feat = at_feat.transpose(1, 2) # [B, T/scale, D']
        return at_feat

    @torch.no_grad()
    def load_and_freeze_wavlm(self, wavlm_path='./dataloaders/wavlm/WavLM-Base+.pt'):
        checkpoint = torch.load(wavlm_path)
        self.wavlm_cfg = WavLMConfig(checkpoint['cfg'])
        self.audio_encoder = WavLM(self.wavlm_cfg)
        self.audio_encoder.load_state_dict(checkpoint['model'])
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    

    def extract_wavlm_feats(self, wav_input_16khz):
        assert self.audio_encoder is not None, "Please load the wavlm model first"
        # check the input type
        if isinstance(wav_input_16khz, np.ndarray):
            wav_input_16khz = torch.from_numpy(wav_input_16khz)
        if wav_input_16khz.dim() == 1:
            wav_input_16khz = wav_input_16khz.unsqueeze(0)
        device = next(self.audio_encoder.parameters()).device
        wav_input_16khz = wav_input_16khz.to(device)

        if self.wavlm_cfg.normalize:
            wav_input_16khz = F.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        
        wavlm_feats = self.audio_encoder.extract_features(wav_input_16khz)[0]
        wavlm_feats = wavlm_feats.detach() # (bs, seq_len, dim)
        
        target_size = math.ceil(wavlm_feats.shape[1] / 50 * self.audio_fps)
        wavlm_feats = F.interpolate(
            wavlm_feats.transpose(1, 2),
            size=target_size,
            align_corners=True,
            mode='linear'
        ).transpose(1, 2)
        return wavlm_feats
        
