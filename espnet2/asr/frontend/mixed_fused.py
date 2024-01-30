from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
import logging

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend


class MixedFusedFrontends(AbsFrontend):
    def __init__(
        self, frontends=None, align_method="linear", proj_dim=100, prediction=False, pred_head_num = 0, fs=16000
    ):
        assert check_argument_types()
        super().__init__()
        self.align_method = (
            align_method
        )
        self.proj_dim = proj_dim  # dim of the projection done on each frontend
        self.fusion = not prediction
        self.pred_head_num = pred_head_num
        self.frontends = []  # list of the frontends to combine
        self.coattention_flag = False
        self.coattention_layers = []
        self.n_mels = 80

        for i, frontend in enumerate(frontends):
            frontend_type = frontend["frontend_type"]
            if frontend_type == "default":
                n_mels, fs, n_fft, win_length, hop_length = (
                    frontend.get("n_mels", 80),

                    fs,
                    frontend.get("n_fft", 512),
                    frontend.get("win_length"),
                    frontend.get("hop_length", 128),
                )
                self.n_mels = n_mels
                window, center, normalized, onesided = (
                    frontend.get("window", "hann"),
                    frontend.get("center", True),
                    frontend.get("normalized", False),
                    frontend.get("onesided", True),
                )
                fmin, fmax, htk, apply_stft = (
                    frontend.get("fmin", None),
                    frontend.get("fmax", None),
                    frontend.get("htk", False),
                    frontend.get("apply_stft", True),
                )

                self.frontends.append(
                    DefaultFrontend(
                        n_mels=n_mels,
                        n_fft=n_fft,
                        fs=fs,
                        win_length=win_length,
                        hop_length=hop_length,
                        window=window,
                        center=center,
                        normalized=normalized,
                        onesided=onesided,
                        fmin=fmin,
                        fmax=fmax,
                        htk=htk,
                        apply_stft=apply_stft,
                    )
                )
            elif frontend_type == "s3prl":
                frontend_conf, download_dir, multilayer_feature = (
                    frontend.get("frontend_conf"),
                    frontend.get("download_dir"),
                    frontend.get("multilayer_feature"),
                )
                self.frontends.append(
                    S3prlFrontend(
                        fs=fs,
                        frontend_conf=frontend_conf,
                        download_dir=download_dir,
                        multilayer_feature=multilayer_feature,
                    )
                )

            else:
                raise NotImplementedError  # frontends are only default or s3prl

        self.frontends = torch.nn.ModuleList(self.frontends)

        self.gcd = np.gcd.reduce([frontend.hop_length for frontend in self.frontends])
        self.factors = [frontend.hop_length // self.gcd for frontend in self.frontends]
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        self.dev = dev
        self.projection_layers = []
        for i, align in enumerate(self.align_method):
            if align == "linear":
                self.projection_layers.append(
                    torch.nn.Linear(
                        in_features=self.frontends[i].output_size(),
                        out_features=self.factors[i] * self.proj_dim,
                    )
                )              
            elif align == "conv":
                #logging.info("conv model number {}", int(i))
                self.projection_layers.append(
                    torch.nn.Conv1d(
                        self.frontends[i].output_size(),
                        self.factors[i] * self.proj_dim,
                        kernel_size=self.factors[i] * 2 + 1,
                        padding=self.factors[i],
                    )
                )            
            elif align == "coattention":
                self.coattention_flag = True
                #logging.info("coattention model number {}", int(i))
                self.coattention_layers.append(i)
                self.projection_layers.append(
                    torch.nn.Linear(
                        in_features=self.frontends[i].output_size(),
                        out_features=self.factors[i] * self.proj_dim,
                    )
                )
            
                self.sqrt_dim = np.sqrt(self.n_mels)
                self.dropout = torch.nn.Dropout(0.2)

        self.frontend_num = len(self.frontends)
        self.projection_layers = torch.nn.ModuleList(self.projection_layers)
        self.projection_layers = self.projection_layers.to(torch.device(dev))
        self.frontend_type = "mixed"

    def output_size(self) -> int:
        return len(self.frontends) * self.proj_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor, training=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # step 0 : get all frontends features
        self.feats = []
        if self.fusion or (not self.fusion and training):
            for i, frontend in enumerate(self.frontends):
                input_feats, feats_lens = frontend.forward(input, input_lengths)
                self.feats.append([input_feats, feats_lens])
                
        elif not self.fusion and not training:
            input_feats, feats_lens = self.frontends[self.pred_head_num].forward(input, input_lengths)
            self.feats.append([input_feats, feats_lens])
        
        pred_model = False
        if len(self.frontends) > 1 or (not self.fusion and not training):
            # only want to perform fusion if there is more than one 
            # if it is inference stage of prediction model, then we also need to project features
            self.feats_proj = []
            self.feats_reshaped = []
            for i, align in enumerate(self.align_method):
                if len(self.feats) == 1 and i != self.pred_head_num:
                    # if self.feats has length 1 then it is in inference stage for prediction
                    pred_model = True
                    continue
                if align == "linear": 
                    # first step : projections
                    k = 0 if pred_model else i
                    input_feats = self.feats[k][0] # only one of the SSL weighted features is pushed into the list
                    self.feats_proj.append(self.projection_layers[i](input_feats))
                    # 2nd step : reshape 
                    input_feats_proj = self.feats_proj[k]

                    bs, nf, dim = input_feats_proj.shape
                    input_feats_reshaped = torch.reshape(
                        input_feats_proj, (bs, nf * self.factors[i], dim // self.factors[i])
                    )
                    self.feats_reshaped.append(input_feats_reshaped)                    

                elif align == "conv":
                    #logging.info("layer {} got to conv", int(i))
                    input_feats = self.feats[i][0]
                    self.feats_proj.append(self.projection_layers[i](input_feats.permute(0, 2, 1)))


                    # 2nd step : reshape

                    input_feats_proj = self.feats_proj[i]
                    bs, dim, nf = input_feats_proj.shape
                    input_feats_reshaped = torch.reshape(
                        input_feats_proj.permute(0, 2, 1), (bs, nf * self.factors[i], dim // self.factors[i])
                    )
                    self.feats_reshaped.append(input_feats_reshaped)


                elif align == "coattention":
                    # first step : projections
                    #logging.info("layer {} got to coattention", int(i))
                    input_feats = self.feats[i][0]
                    self.feats_proj.append(self.projection_layers[i](input_feats))

                    # 2nd step : reshape
                    input_feats_proj = self.feats_proj[i]
                    bs, nf, dim = input_feats_proj.shape
                    input_feats_reshaped = torch.reshape(
                        input_feats_proj, (bs, nf * self.factors[i], dim // self.factors[i])
                    )
                    self.feats_reshaped.append(input_feats_reshaped)
                    

                else:
                    raise NotImplementedError

            # 3rd step : drop the few last frames
            m = min([x.shape[1] for x in self.feats_reshaped])
            self.feats_final = [x[:, :m, :] for x in self.feats_reshaped]

            # coattention block
            if self.coattention_flag:
                for i, layer_num in enumerate(self.coattention_layers):
                    #logging.info("coattention layer num: {}", int(layer_num))
                    query  = self.feats_final[layer_num]
                    key = self.feats_final[self.coattention_layers[1-i]] 
                    score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
                    attn = torch.nn.functional.softmax(score, -1)
                    self.feats_final[layer_num] = torch.bmm(attn, key)

            # 4th step : concatenate all of the features
            input_feats = torch.cat(
                self.feats_final, dim=-1
            )  # change the input size of the preencoder : proj_dim * n_frontends
            feats_lens = torch.ones_like(self.feats[0][1]) * (m)
        return input_feats, feats_lens
