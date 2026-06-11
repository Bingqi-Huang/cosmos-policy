"""SCVC trainer subclass for Cosmos Policy Video2World.

This file implements the paired forward/loss path only.  It does not change
the inference contract and it does not run unless an experiment config selects
``SCVCPolicyVideo2WorldModel`` and the dataloader provides ``video_pair``.
"""

from __future__ import annotations

import attrs
import torch

from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldConfig, CosmosPolicyVideo2WorldModel


@attrs.define(slots=False)
class SCVCPolicyVideo2WorldConfig(CosmosPolicyVideo2WorldConfig):
    lambda_cv: float = 0.1
    cv_warmup_start_fraction: float = 0.0
    cv_warmup_end_fraction: float = 0.1
    cv_frame_set: str = "action+value+fproprio"
    cv_noise_shared: bool = True
    cv_pair_mode: str = "matched"  # matched | derangement
    cv_num_samples: int = 2
    cv_total_steps: int = 10000

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        valid_frame_sets = {"action", "action+value", "action+value+fproprio", "full"}
        if self.cv_frame_set not in valid_frame_sets:
            raise ValueError(f"cv_frame_set must be one of {sorted(valid_frame_sets)}, got {self.cv_frame_set!r}")
        if self.cv_pair_mode not in {"matched", "derangement"}:
            raise ValueError("cv_pair_mode must be 'matched' or 'derangement'")
        if self.cv_num_samples < 1:
            raise ValueError("cv_num_samples must be >= 1")
        if self.cv_total_steps < 1:
            raise ValueError("cv_total_steps must be >= 1")


class SCVCPolicyVideo2WorldModel(CosmosPolicyVideo2WorldModel):
    def __init__(self, config: SCVCPolicyVideo2WorldConfig):
        super().__init__(config)
        self.config: SCVCPolicyVideo2WorldConfig = config

    def _lambda_cv_for_iteration(self, iteration: int) -> float:
        progress = float(iteration) / float(self.config.cv_total_steps)
        start = float(self.config.cv_warmup_start_fraction)
        end = float(self.config.cv_warmup_end_fraction)
        if progress <= start:
            return 0.0
        if progress >= end:
            return float(self.config.lambda_cv)
        if end <= start:
            return float(self.config.lambda_cv)
        return float(self.config.lambda_cv) * (progress - start) / (end - start)

    @staticmethod
    def _derangement(batch_size: int, device: torch.device) -> torch.Tensor:
        if batch_size < 2:
            raise ValueError("cv_pair_mode='derangement' requires batch size >= 2")
        identity = torch.arange(batch_size, device=device)
        # Rejection sampling: a uniform random permutation is a derangement with prob ~1/e,
        # so a handful of tries almost always succeeds (CoRL `_random_derangement_permutation` semantics).
        for _ in range(16):
            perm = torch.randperm(batch_size, device=device)
            if not torch.any(perm == identity):
                return perm
        # Fallback: conjugate a cyclic shift by a random permutation — always a derangement.
        sigma = torch.randperm(batch_size, device=device)
        perm = torch.empty_like(sigma)
        perm[sigma] = sigma[(identity + 1) % batch_size]
        return perm

    @staticmethod
    def _permute_batch_dim(value, perm: torch.Tensor, batch_size: int):
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            return value[perm]
        if isinstance(value, list) and len(value) == batch_size:
            perm_cpu = perm.detach().cpu().tolist()
            return [value[i] for i in perm_cpu]
        return value

    def _make_paired_batch(self, data_batch: dict, perm: torch.Tensor | None = None) -> dict:
        paired_batch = dict(data_batch)
        paired_batch["video"] = data_batch["video_pair"]
        if perm is None:
            return paired_batch
        batch_size = int(data_batch["video"].shape[0])
        return {key: self._permute_batch_dim(value, perm, batch_size) for key, value in paired_batch.items()}

    def _reduce_like_base(self, loss: torch.Tensor) -> torch.Tensor:
        if self.loss_reduce == "mean":
            return loss.mean()
        if self.loss_reduce == "sum":
            return loss.sum(dim=1).mean()
        raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

    def _branch_loss(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
        data_batch: dict,
    ):
        return self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W,
            condition,
            epsilon_B_C_T_H_W,
            sigma_B_T,
            action_chunk=data_batch["actions"],
            action_indices=data_batch["action_latent_idx"],
            proprio=data_batch["proprio"],
            current_proprio_indices=data_batch["current_proprio_latent_idx"],
            future_proprio=data_batch["future_proprio"],
            future_proprio_indices=data_batch["future_proprio_latent_idx"],
            future_wrist_image_indices=data_batch["future_wrist_image_latent_idx"],
            future_wrist_image2_indices=(
                data_batch["future_wrist_image2_latent_idx"] if "future_wrist_image2_latent_idx" in data_batch else None
            ),
            future_image_indices=data_batch["future_image_latent_idx"],
            future_image2_indices=(
                data_batch["future_image2_latent_idx"] if "future_image2_latent_idx" in data_batch else None
            ),
            rollout_data_mask=data_batch["rollout_data_mask"],
            world_model_sample_mask=data_batch["world_model_sample_mask"],
            value_function_sample_mask=data_batch["value_function_sample_mask"],
            value_function_return=data_batch["value_function_return"],
            value_indices=data_batch["value_latent_idx"],
        )

    def _cv_frame_indices(self, data_batch: dict) -> list[torch.Tensor]:
        frame_set = self.config.cv_frame_set
        indices = [data_batch["action_latent_idx"]]
        if frame_set in {"action+value", "action+value+fproprio", "full"}:
            indices.append(data_batch["value_latent_idx"])
        if frame_set in {"action+value+fproprio", "full"}:
            indices.append(data_batch["future_proprio_latent_idx"])
        if frame_set == "full":
            indices.append(data_batch["future_image_latent_idx"])
        return indices

    @staticmethod
    def _pair_valid_mask(batch: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        demo_mask = (batch["rollout_data_mask"].to(device) == 0).to(dtype)
        pair_valid = batch.get("pair_valid", torch.ones_like(batch["rollout_data_mask"]))
        return demo_mask * pair_valid.to(device).to(dtype)

    def _cv_loss_unscaled(
        self,
        pred0: torch.Tensor,
        predp: torch.Tensor,
        weights_B_T: torch.Tensor,
        data_batch: dict,
        paired_batch: dict,
    ) -> torch.Tensor:
        # λ-bookkeeping contract (execution_plan §0.1, binding): this term must use the same per-frame
        # w(σ) and the same per-element normalization as the FM loss, and be added BEFORE loss_scale,
        # so that the nominal lambda_cv is exactly the λ of Lemma 1 / Prop. 2.
        # FM ("mean"):  kendall.mean()            => per-element coeff 1/(B·C·T·H·W)
        # FM ("sum"):   kendall.sum(1).mean()     => per-(C-summed)-element coeff 1/(B·T·H·W)
        # CV below mirrors both conventions and divides by (B·T), never by the number of CV frames
        # or the number of valid pairs.
        batch_size, _, num_frames, _, _ = pred0.shape
        batch_indices = torch.arange(batch_size, device=pred0.device)
        # Both comparison arms must be valid demo pairs (matters under derangement and for
        # pass-through rollout samples, which carry pair_valid=0).
        valid_mask = self._pair_valid_mask(data_batch, pred0.device, pred0.dtype) * self._pair_valid_mask(
            paired_batch, pred0.device, pred0.dtype
        )

        total = torch.zeros((), device=pred0.device, dtype=torch.float32)
        for idx in self._cv_frame_indices(data_batch):
            idx = idx.to(pred0.device).long()
            good = (idx != -1).to(pred0.dtype)
            if not torch.any(good > 0):
                continue
            safe_idx = idx.clamp_min(0)
            diff = pred0[batch_indices, :, safe_idx, :, :] - predp[batch_indices, :, safe_idx, :, :]
            if self.loss_reduce == "mean":
                per_sample = (diff**2).mean(dim=(1, 2, 3))
            elif self.loss_reduce == "sum":
                per_sample = (diff**2).sum(dim=1).mean(dim=(1, 2))
            else:
                raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")
            per_sample = per_sample * weights_B_T[batch_indices, safe_idx] * valid_mask * good
            total = total + per_sample.sum().float()
        return total / float(batch_size * num_frames)

    def _covariant_ratio(self, out0: dict, outp: dict, data_batch: dict, paired_batch: dict) -> torch.Tensor:
        # Monitoring-only shrinkage signal (the binding A2 measurement is the held-out per-checkpoint
        # protocol of LOCKED DECISION 8). Restricted to valid matched demo pairs: pass-through rollout
        # samples have video_pair == video, so their target diff is ~0 and would corrupt the ratio.
        pred0 = out0["model_pred"].x0
        predp = outp["model_pred"].x0
        target0 = out0["x0"]
        targetp = outp["x0"]
        idx = data_batch["future_image_latent_idx"].to(pred0.device).long()
        valid = (
            self._pair_valid_mask(data_batch, pred0.device, pred0.dtype)
            * self._pair_valid_mask(paired_batch, pred0.device, pred0.dtype)
        ) > 0
        good = (idx != -1) & valid
        if not torch.any(good):
            return torch.tensor(float("nan"), device=pred0.device)
        batch_indices = torch.arange(pred0.shape[0], device=pred0.device)
        safe_idx = idx.clamp_min(0)
        pred_diff = pred0[batch_indices, :, safe_idx, :, :] - predp[batch_indices, :, safe_idx, :, :]
        target_diff = target0[batch_indices, :, safe_idx, :, :] - targetp[batch_indices, :, safe_idx, :, :]
        numerator = torch.linalg.vector_norm(pred_diff[good].float())
        denominator = torch.linalg.vector_norm(target_diff[good].float()).clamp_min(1e-12)
        return numerator / denominator

    def training_step(self, data_batch: dict, iteration: int):
        if "video_pair" not in data_batch:
            raise KeyError("SCVC training requires data_batch['video_pair']; use LIBEROPairDataset.")

        self._update_train_stats(data_batch)
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        derange_perm = None
        if self.config.cv_pair_mode == "derangement":
            derange_perm = self._derangement(int(data_batch["video"].shape[0]), data_batch["video"].device)
        paired_batch = self._make_paired_batch(data_batch, derange_perm)

        total_fm = None
        total_cv = None
        last_out0 = None
        last_outp = None
        lambda_cv = self._lambda_cv_for_iteration(iteration)

        # VAE encoding (get_data_and_condition) is hoisted out of the noise-draw loop: the latents and
        # conditioning are draw-independent, and re-encoding per draw would double the tokenizer cost.
        # NOTE on memory: the graphs of 2 branches × cv_num_samples draws stay alive until the single
        # backward — residency ≈ 2·K_s·bs sample-forwards. The two-branch memory smoke must run with
        # the real (bs, cv_num_samples) before any pilot (execution_plan §1 locked conventions).
        _, x0_raw, condition_raw = self.get_data_and_condition(data_batch)
        _, x0p_raw, conditionp_raw = self.get_data_and_condition(paired_batch)

        for _ in range(int(self.config.cv_num_samples)):
            sigma, epsilon = self.draw_training_sigma_and_epsilon(x0_raw.size(), condition_raw)
            if self.config.cv_noise_shared:
                sigmap, epsilonp = sigma, epsilon
            else:
                sigmap, epsilonp = self.draw_training_sigma_and_epsilon(x0p_raw.size(), conditionp_raw)

            x0, condition, epsilon, sigma = self.broadcast_split_for_model_parallelsim(
                x0_raw, condition_raw, epsilon, sigma
            )
            x0p, conditionp, epsilonp, sigmap = self.broadcast_split_for_model_parallelsim(
                x0p_raw, conditionp_raw, epsilonp, sigmap
            )

            out0, loss0, _, _ = self._branch_loss(x0, condition, epsilon, sigma, data_batch)
            outp, lossp, _, _ = self._branch_loss(x0p, conditionp, epsilonp, sigmap, paired_batch)

            fm_unscaled = 0.5 * (self._reduce_like_base(loss0) + self._reduce_like_base(lossp))
            cv_unscaled = self._cv_loss_unscaled(
                out0["model_pred"].x0, outp["model_pred"].x0, out0["weights_per_sigma"], data_batch, paired_batch
            )
            total_fm = fm_unscaled if total_fm is None else total_fm + fm_unscaled
            total_cv = cv_unscaled if total_cv is None else total_cv + cv_unscaled
            last_out0, last_outp = out0, outp

        assert total_fm is not None and total_cv is not None and last_out0 is not None and last_outp is not None
        total_fm = total_fm / float(self.config.cv_num_samples)
        total_cv = total_cv / float(self.config.cv_num_samples)
        # CV is injected BEFORE loss_scale so the FM:CV ratio (= lambda_cv) is preserved (λ bookkeeping).
        loss = (total_fm + lambda_cv * total_cv) * self.loss_scale

        output_batch = dict(last_out0)
        output_batch.update(
            {
                "scvc_fm_loss_unscaled": total_fm.detach(),
                "scvc_cv_loss_unscaled": total_cv.detach(),
                "scvc_lambda_cv": torch.tensor(lambda_cv, device=loss.device),
                "scvc_cv_frame_set_full": torch.tensor(1 if self.config.cv_frame_set == "full" else 0, device=loss.device),
                "scvc_covariant_future_image_ratio": self._covariant_ratio(
                    last_out0, last_outp, data_batch, paired_batch
                ).detach(),
            }
        )
        return output_batch, loss
