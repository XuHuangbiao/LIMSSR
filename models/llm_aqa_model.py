import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM_AQA(nn.Module):
    def __init__(
        self,
        in_dim,
        clip_num,
        dropout,
        llm_path,
        use_lora=True,
        lora_r=8,
        num_fusion_tokens=4,
    ):
        super().__init__()
        self.clip_num = clip_num
        self.num_fusion_tokens = num_fusion_tokens

        # ========== LLM and tokenizer ==========
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "pretrained_model_name_or_path": llm_path,
            "torch_dtype": torch.float32,
            "device_map": "auto",
        }
        pretrained_llm = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.llm_hidden_dim = pretrained_llm.config.hidden_size

        special_tokens = [
            "<|video_start|>", "<|video_end|>",
            "<|audio_start|>", "<|audio_end|>",
            "<|flow_start|>", "<|flow_end|>",
            "<|missing_video|>", "<|missing_audio|>", "<|missing_flow|>",
        ]
        for i in range(1, num_fusion_tokens + 1):
            special_tokens.append(f"<|feat_dim_{i}|>")

        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        if use_lora:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                inference_mode=False,
                r=lora_r,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.language_model = get_peft_model(pretrained_llm, lora_config)
            # Keep token embeddings and LM head trainable.
            self.language_model.lm_head.weight.requires_grad = True
            self.language_model.base_model.model.model.embed_tokens.weight.requires_grad = True
        else:
            self.language_model = pretrained_llm

        self.language_model.resize_token_embeddings(len(self.tokenizer))

        # ========== Feature projection into the LLM hidden space ==========
        self.feature_to_llm = nn.ModuleDict({
            "v": nn.Sequential(
                nn.Conv1d(in_dim, self.llm_hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.llm_hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Conv1d(self.llm_hidden_dim, self.llm_hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.llm_hidden_dim),
            ),
            "a": nn.Sequential(
                nn.Conv1d(768, self.llm_hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.llm_hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Conv1d(self.llm_hidden_dim, self.llm_hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.llm_hidden_dim),
            ),
            "f": nn.Sequential(
                nn.Conv1d(1024, self.llm_hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.llm_hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Conv1d(self.llm_hidden_dim, self.llm_hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.llm_hidden_dim),
            ),
        })

        self.LN = nn.LayerNorm(self.llm_hidden_dim)

        # ========== Multimodal fusion modules ==========
        # Cross-modal attention over modality-level representations.
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=self.llm_hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Normalize each modality representation before fusion.
        self.modal_norms = nn.ModuleDict({
            "v": nn.LayerNorm(self.llm_hidden_dim),
            "a": nn.LayerNorm(self.llm_hidden_dim),
            "f": nn.LayerNorm(self.llm_hidden_dim),
        })

        # Learn modality weights conditioned on the concatenated features.
        self.modal_gate = nn.Sequential(
            nn.Linear(self.llm_hidden_dim * 3, self.llm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.llm_hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

        # ========== Main regression head ==========
        self.regressor = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim // 2),
            nn.LayerNorm(self.llm_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_hidden_dim // 2, 1),
        )

        # ========== Auxiliary regression head over modal features ==========
        self.auxiliary_regressor = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim // 2),
            nn.LayerNorm(self.llm_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_hidden_dim // 2, 1),
        )

        # Learn fusion-token importance.
        self.token_role_weight = nn.Parameter(torch.ones(num_fusion_tokens))

        # ========== Mask-aware conditional refinement head ==========
        self.cond_head = nn.Sequential(
            nn.Linear(self.llm_hidden_dim + 3, self.llm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
        )
        self.cond_gate = nn.Sequential(
            nn.Linear(self.llm_hidden_dim + 3, self.llm_hidden_dim),
            nn.Sigmoid(),
        )

        # Learn missing-modality confidence instead of hard-coding it.
        self.missing_confidence = nn.Parameter(torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32))
        self.w1 = nn.Parameter(torch.tensor([0.7], dtype=torch.float32))

    def get_embedding_layer(self):
        from peft import PeftModel

        if isinstance(self.language_model, PeftModel):
            return self.language_model.base_model.model.model.embed_tokens
        return self.language_model.model.embed_tokens

    def construct_llm_input(self, v_feat, a_feat, f_feat, mask):
        """
        Build the LLM input sequence and track token spans for each modality.

        Returns:
            full_embeds: Input embeddings fed to the LLM.
            full_mask: Attention mask for the input sequence.
            feature_positions: Token indices of the fusion tokens.
            modal_positions: Start/end index pair for each modality block.
        """
        v_mask, a_mask, f_mask = mask

        if v_feat is not None:
            b, t, _ = v_feat.shape
            device = v_feat.device
        elif a_feat is not None:
            b, t, _ = a_feat.shape
            device = a_feat.device
        else:
            b, t, _ = f_feat.shape
            device = f_feat.device

        assert t == self.clip_num, f"Expected t==clip_num=={self.clip_num}, got t={t}"

        embedding_layer = self.get_embedding_layer()

        video_start_id = self.tokenizer.convert_tokens_to_ids("<|video_start|>")
        video_end_id = self.tokenizer.convert_tokens_to_ids("<|video_end|>")
        audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        audio_end_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        flow_start_id = self.tokenizer.convert_tokens_to_ids("<|flow_start|>")
        flow_end_id = self.tokenizer.convert_tokens_to_ids("<|flow_end|>")
        missing_video_id = self.tokenizer.convert_tokens_to_ids("<|missing_video|>")
        missing_audio_id = self.tokenizer.convert_tokens_to_ids("<|missing_audio|>")
        missing_flow_id = self.tokenizer.convert_tokens_to_ids("<|missing_flow|>")

        feat_dim_ids = [
            self.tokenizer.convert_tokens_to_ids(f"<|feat_dim_{i}|>")
            for i in range(1, self.num_fusion_tokens + 1)
        ]

        # Describe which modalities are observed and which are missing.
        prompt_texts = []
        for _ in range(b):
            available_modalities = []
            missing_modalities = []
            if v_mask == 1:
                available_modalities.append("visual")
            else:
                missing_modalities.append("visual")
            if a_mask == 1:
                available_modalities.append("audio")
            else:
                missing_modalities.append("audio")
            if f_mask == 1:
                available_modalities.append("flow")
            else:
                missing_modalities.append("flow")

            avail_str = ", ".join(available_modalities) if available_modalities else "no available modality"
            miss_str = ", ".join(missing_modalities) if missing_modalities else "none"

            prompt = (f"Given the available {avail_str} features from an action video. "
                     f"The {miss_str} modality is missing. "
                     f"Based on the available modalities, please infer and reconstruct the useful latent representations for the missing {miss_str} modalities at the designated positions. "
                     f"Then integrate and enhance all multimodal features for action quality assessment. "
                     f"Output the fused multi-dimensional feature representations at the designated feature dimension positions: ")
            prompt_texts.append(prompt)

        prompt_encoding = self.tokenizer(
            prompt_texts,
            padding=True,
            return_tensors="pt",
        ).to(device)

        prompt_embeds = embedding_layer(prompt_encoding["input_ids"])
        prompt_mask = prompt_encoding["attention_mask"]

        modal_embeds_list = [prompt_embeds]
        modal_mask_list = [prompt_mask]

        # Track the span occupied by each modality block in the final sequence.
        modal_positions = {"v": None, "a": None, "f": None}
        current_pos = prompt_embeds.size(1)

        ones1 = torch.ones(b, 1, device=device, dtype=torch.long)
        ones_t = torch.ones(b, t, device=device, dtype=torch.long)

        def add_modal_block(modal_key, start_id, end_id, missing_token_id, feat_or_none, is_present):
            """Append one modality block and record its token span."""
            nonlocal current_pos

            modal_embeds_list.append(embedding_layer(torch.tensor([[start_id]] * b, device=device)))
            modal_mask_list.append(ones1)
            current_pos += 1

            start_pos = current_pos
            if is_present and feat_or_none is not None:
                modal_embeds_list.append(feat_or_none)
                modal_mask_list.append(ones_t)
            else:
                missing_token_embed = embedding_layer(torch.tensor([[missing_token_id]], device=device))
                missing_token_embed = missing_token_embed.expand(b, t, -1)
                modal_embeds_list.append(missing_token_embed)
                modal_mask_list.append(ones_t)

            modal_positions[modal_key] = (start_pos, start_pos + t)
            current_pos += t

            modal_embeds_list.append(embedding_layer(torch.tensor([[end_id]] * b, device=device)))
            modal_mask_list.append(ones1)
            current_pos += 1

        # Append available modalities first.
        if v_mask == 1:
            add_modal_block("v", video_start_id, video_end_id, missing_video_id, v_feat, True)
        if a_mask == 1:
            add_modal_block("a", audio_start_id, audio_end_id, missing_audio_id, a_feat, True)
        if f_mask == 1:
            add_modal_block("f", flow_start_id, flow_end_id, missing_flow_id, f_feat, True)

        # Append missing modalities afterwards using placeholder tokens.
        if v_mask == 0:
            add_modal_block("v", video_start_id, video_end_id, missing_video_id, v_feat, False)
        if a_mask == 0:
            add_modal_block("a", audio_start_id, audio_end_id, missing_audio_id, a_feat, False)
        if f_mask == 0:
            add_modal_block("f", flow_start_id, flow_end_id, missing_flow_id, f_feat, False)

        # Append fusion-token positions at the end of the sequence.
        for feat_id in feat_dim_ids:
            modal_embeds_list.append(embedding_layer(torch.tensor([[feat_id]] * b, device=device)))
            modal_mask_list.append(ones1)

        full_embeds = torch.cat(modal_embeds_list, dim=1)
        full_mask = torch.cat(modal_mask_list, dim=1)

        feature_positions_start = full_embeds.size(1) - self.num_fusion_tokens
        feature_positions = list(range(feature_positions_start, full_embeds.size(1)))

        return full_embeds, full_mask, feature_positions, modal_positions

    def extract_and_fuse_modal_features(self, llm_hidden, modal_positions, mask):
        """
        Extract modality-specific features from the LLM output and fuse them.

        Args:
            llm_hidden: Hidden states with shape (b, seq_len, llm_dim).
            modal_positions: Maps each modality to its token span.
            mask: Tuple of modality indicators (v_mask, a_mask, f_mask).

        Returns:
            Fused modality feature of shape (b, llm_dim).
        """
        v_mask, a_mask, f_mask = mask
        b = llm_hidden.size(0)
        device = llm_hidden.device

        # Average temporal tokens inside each modality span.
        modal_feats = {}
        for modal_key in ["v", "a", "f"]:
            if modal_positions[modal_key] is not None:
                start, end = modal_positions[modal_key]
                modal_feat = llm_hidden[:, start:end, :].mean(dim=1)
                modal_feats[modal_key] = self.modal_norms[modal_key](modal_feat)
            else:
                modal_feats[modal_key] = torch.zeros(b, self.llm_hidden_dim, device=device)

        v_feat = modal_feats["v"]
        a_feat = modal_feats["a"]
        f_feat = modal_feats["f"]

        # Self-attention over modality-level features.
        modal_seq = torch.stack([v_feat, a_feat, f_feat], dim=1)
        attn_out, _ = self.cross_modal_attn(modal_seq, modal_seq, modal_seq)

        # Adaptive gating over modality-level features.
        concat_feats = torch.cat([v_feat, a_feat, f_feat], dim=-1)
        weights = self.modal_gate(concat_feats)
        weighted_feats = attn_out * weights.unsqueeze(-1)

        # Down-weight missing modalities using a learned confidence prior.
        mask_tensor = torch.tensor([v_mask, a_mask, f_mask], device=device, dtype=torch.float32)
        mask_tensor = mask_tensor.view(1, 3, 1).expand(b, -1, -1)
        miss_conf = torch.sigmoid(self.missing_confidence).view(1, 3, 1).expand(b, -1, -1)
        confidence = mask_tensor + (1.0 - mask_tensor) * miss_conf
        weighted_feats = weighted_feats * confidence

        return weighted_feats.sum(dim=1)

    def forward(self, video, audio, flow, mask):
        v_mask, a_mask, f_mask = mask

        # Project only the available modalities.
        v = self.feature_to_llm["v"](video.transpose(1, 2)).transpose(1, 2) if v_mask == 1 else None
        a = self.feature_to_llm["a"](audio.transpose(1, 2)).transpose(1, 2) if a_mask == 1 else None
        f = self.feature_to_llm["f"](flow.transpose(1, 2)).transpose(1, 2) if f_mask == 1 else None

        llm_inputs, llm_mask, feature_positions, modal_positions = self.construct_llm_input(v, a, f, mask)
        llm_outputs = self.language_model(
            inputs_embeds=llm_inputs,
            attention_mask=llm_mask,
            output_hidden_states=True,
        )
        llm_hidden = llm_outputs.hidden_states[-1]

        # Path 1: fusion-token-based representation.
        multi_dim_features = llm_hidden[:, feature_positions, :]
        weights = F.softmax(self.token_role_weight, dim=0)
        fused_feature = (multi_dim_features * weights[None, :, None]).sum(dim=1)
        fused_feature = self.LN(fused_feature)

        # Refine the fused feature with a mask-aware conditional head.
        mask_tensor = torch.tensor(mask, device=fused_feature.device, dtype=fused_feature.dtype).view(1, 3)
        mask_tensor = mask_tensor.expand(fused_feature.size(0), -1)
        cond_in = torch.cat([fused_feature, mask_tensor], dim=-1)
        delta = self.cond_head(cond_in)
        gate = self.cond_gate(cond_in)
        fused_feature = fused_feature + gate * delta

        out_main = torch.sigmoid(self.regressor(fused_feature).squeeze(-1))

        # Path 2: modality-span-based representation.
        fused_modal_feat = self.extract_and_fuse_modal_features(llm_hidden, modal_positions, mask)
        out_aux = torch.sigmoid(self.auxiliary_regressor(fused_modal_feat).squeeze(-1))

        # Final prediction fusion.
        out_final = (self.w1 * out_main) + ((1.0 - self.w1) * out_aux)

        return {
            "output": out_final,
            "output_main": out_main,
            "output_aux": out_aux,
            "embed": multi_dim_features,
            "fused_modal_feat": fused_modal_feat,
        }
