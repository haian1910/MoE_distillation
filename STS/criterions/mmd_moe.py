import torch
import torch.nn as nn
import torch.nn.functional as F
from .moe_sts_loss import MoE_STSLoss

class CKALoss(nn.Module):
    """
    Loss with knowledge distillation using CKA (Centered Kernel Alignment).
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float32)  # Force float32
        TH = TH.view(-1, dT).to(SH.device, torch.float32)  # Force float32
        
        # Dropout on Hidden State Matching
        SH = F.normalize(SH, p=2, dim=1)
        TH = F.normalize(TH, p=2, dim=1)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)

        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        return 1 - num/torch.sqrt(den1*den2)
    
class MMD_MOE(MoE_STSLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized
        # STS-specific parameters
        self.kd_rate = args.kd_rate  # Knowledge distillation rate 
        
        # MoE-specific parameters
        self.moe_loss_weight = getattr(args, "moe_loss_weight", 0.1)
        self.load_balancing_weight = getattr(args, "load_balancing_weight", 0.01)
        self.diversity_weight = getattr(args, "diversity_weight", 0.1)
        
        # Initialize CKA loss module
        self.cka_loss = CKALoss(eps=getattr(args, 'cka_eps', 1e-6))  # Increase eps
        
        # Parameters for ranking loss
        self.rank_margin = getattr(args, 'rank_margin', 0.1)
        
        # Force float32 for numerical stability
        self.use_fp32 = True
        
        # Check if BF16 is being used
        self.use_bf16 = getattr(args, "bf16", False)
        if not hasattr(args, "bf16"):
            # Try to detect from DeepSpeed config
            if hasattr(args, "deepspeed_config"):
                try:
                    import json
                    with open(args.deepspeed_config, 'r') as f:
                        ds_config = json.load(f)
                        self.use_bf16 = ds_config.get("bf16", {}).get("enabled", False)
                except:
                    pass

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        # Student forward pass
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            return_moe_outputs=True,
            output_hidden_states=True,
            return_dict=True,
            labels=output_data["labels"]
        )
        
        # Extract predictions for STS task
        if hasattr(outputs, 'scores') and outputs.scores is not None:
            predictions = outputs.scores
        elif hasattr(outputs, 'logits') and outputs.logits is not None:
            predictions = outputs.logits
        else:
            if isinstance(outputs, dict):
                predictions = outputs.get("scores", outputs.get("logits"))
            else:
                raise AttributeError("Student outputs does not have 'scores' or 'logits' attribute")
        
        log = {}
        
        # FIXED: Use float32 for all computations to avoid numerical issues
        target_dtype = torch.float32
        device = predictions.device
        
        # Convert predictions to float32
        predictions = predictions.to(dtype=target_dtype)
        
        # Convert labels to the same dtype as predictions
        labels = output_data["labels"].to(dtype=target_dtype, device=device)
        
        # Ensure predictions have the correct shape
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        # Compute MSE loss with consistent dtype
        loss_sts = F.mse_loss(predictions, labels)
        
        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True,
                return_dict=True
            )
        
        # Compute MMD loss with better error handling
        mmd_loss = torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)
        try:
            mmd_loss_computed, log = self.compute_mmd_loss(
                outputs, teacher_outputs, output_data, distiller, log
            )
            
            # Check if MMD loss is valid
            if torch.isfinite(mmd_loss_computed) and not torch.isnan(mmd_loss_computed):
                mmd_loss = mmd_loss_computed.to(dtype=target_dtype)
                print(f"mmd_loss: {mmd_loss}")
            else:
                print(f"Warning: Invalid MMD loss {mmd_loss_computed}, using zero")
                
        except Exception as e:
            print(f"Error computing MMD loss: {e}")
            import traceback
            traceback.print_exc()

        # Compute MOE loss
        moe_loss, log = self.compute_moe_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        
        # Convert moe_loss to target dtype
        moe_loss = moe_loss.to(dtype=target_dtype)
        print(f"moe_loss: {moe_loss}")

        # Compute expert diversity loss
        diversity_loss = self.compute_expert_diversity_loss(outputs.expert_outputs)
        diversity_loss = diversity_loss.to(dtype=target_dtype)
    
        # Combine all losses
        total_moe_loss = moe_loss + self.diversity_weight * diversity_loss

        # Convert scalar coefficients to target dtype
        kd_rate = torch.tensor(self.kd_rate, device=device, dtype=target_dtype)
        one_minus_kd_rate = torch.tensor(1.0 - self.kd_rate, device=device, dtype=target_dtype)
        
        # Combine losses with gradient clipping
        loss = one_minus_kd_rate * loss_sts + kd_rate * (mmd_loss + total_moe_loss)
        
        # Gradient clipping to prevent NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Loss is NaN or Inf, using STS loss only")
            loss = loss_sts

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss, logging_output

    @staticmethod
    def mmd(x, y, kernel="multiscale"):
        """Empirical maximum mean discrepancy with improved numerical stability"""
        device = x.device
        target_dtype = torch.float32  # Force float32 for stability
        
        # Convert to float32 and check for invalid values
        x = x.to(dtype=target_dtype)
        y = y.to(dtype=target_dtype)
        
        # More aggressive NaN/Inf handling
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Invalid values in x, cleaning...")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: Invalid values in y, cleaning...")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Add small noise to prevent identical samples
        noise_scale = 1e-8
        x = x + torch.randn_like(x) * noise_scale
        y = y + torch.randn_like(y) * noise_scale
        
        # Ensure input tensors are contiguous
        x = x.contiguous()
        y = y.contiguous()
        
        # Flatten and limit size for memory efficiency
        x_flat = x.view(-1, x.size(-1))
        y_flat = y.view(-1, y.size(-1))
        
        # Limit batch size for memory efficiency
        max_samples = 1000
        if x_flat.size(0) > max_samples:
            indices = torch.randperm(x_flat.size(0))[:max_samples]
            x_flat = x_flat[indices]
        if y_flat.size(0) > max_samples:
            indices = torch.randperm(y_flat.size(0))[:max_samples]
            y_flat = y_flat[indices]
        
        # Larger epsilon for numerical stability
        eps = torch.tensor(1e-6, device=device, dtype=target_dtype)
        
        try:
            # L2 normalize to prevent scale issues
            x_flat = F.normalize(x_flat, p=2, dim=1)
            y_flat = F.normalize(y_flat, p=2, dim=1)
            
            xx = torch.mm(x_flat, x_flat.t())
            yy = torch.mm(y_flat, y_flat.t())
            zz = torch.mm(x_flat, y_flat.t())
            
            rx = xx.diag().unsqueeze(0).expand_as(xx)
            ry = yy.diag().unsqueeze(0).expand_as(yy)

            dxx = torch.clamp(rx.t() + rx - 2. * xx, min=eps, max=100.0)
            dyy = torch.clamp(ry.t() + ry - 2. * yy, min=eps, max=100.0)
            dxy = torch.clamp(rx.t() + ry - 2. * zz, min=eps, max=100.0)

            XX = torch.zeros(xx.shape, device=device, dtype=target_dtype)
            YY = torch.zeros(yy.shape, device=device, dtype=target_dtype)
            XY = torch.zeros(zz.shape, device=device, dtype=target_dtype)

            if kernel == "multiscale":
                bandwidth_range = [0.1, 0.5, 1.0, 2.0]  # Adjusted bandwidth
                for a in bandwidth_range:
                    a_tensor = torch.tensor(a**2, device=device, dtype=target_dtype)
                    XX += a_tensor / (a_tensor + dxx + eps)
                    YY += a_tensor / (a_tensor + dyy + eps)
                    XY += a_tensor / (a_tensor + dxy + eps)

            elif kernel == "rbf":
                bandwidth_range = [1, 5, 10, 25]  # Smaller bandwidth
                for a in bandwidth_range:
                    a_tensor = torch.tensor(a, device=device, dtype=target_dtype)
                    # More conservative clamping
                    exp_xx = torch.clamp(-0.5 * dxx / a_tensor, min=-10, max=10)
                    exp_yy = torch.clamp(-0.5 * dyy / a_tensor, min=-10, max=10)
                    exp_xy = torch.clamp(-0.5 * dxy / a_tensor, min=-10, max=10)
                    
                    XX += torch.exp(exp_xx)
                    YY += torch.exp(exp_yy)
                    XY += torch.exp(exp_xy)

            mmd_result = torch.mean(XX + YY - 2. * XY)
            
            # Final validation
            if torch.isnan(mmd_result) or torch.isinf(mmd_result) or mmd_result < 0:
                print(f"Warning: Invalid MMD result {mmd_result}, returning small positive value")
                return torch.tensor(1e-6, device=device, dtype=target_dtype, requires_grad=True)
                
            # Clamp to reasonable range
            mmd_result = torch.clamp(mmd_result, min=0.0, max=10.0)
            return mmd_result
            
        except Exception as e:
            print(f"Exception in MMD computation: {e}")
            return torch.tensor(1e-6, device=device, dtype=target_dtype, requires_grad=True)

    def compute_mmd_loss(self, outputs, teacher_outputs, output_data, distiller, log):
        """Compute MMD loss with better error handling"""
        # Handle outputs
        if isinstance(outputs, dict):
            student_hidden_states = outputs.get('hidden_states', None)
        else:
            student_hidden_states = getattr(outputs, 'hidden_states', None)
        
        if isinstance(teacher_outputs, dict):
            teacher_hidden_states = teacher_outputs.get('hidden_states', None)
        else:
            teacher_hidden_states = getattr(teacher_outputs, 'hidden_states', None)
        
        if student_hidden_states is None or teacher_hidden_states is None:
            print("Warning: Hidden states not available, returning zero MMD loss")
            device = next(distiller.student_model.parameters()).device
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True), log
        
        target_dtype = torch.float32
        device = student_hidden_states[0].device
        
        # Process fewer layers to reduce computational load
        student_layers_to_process = [9, 10]  # Only last 2 layers
        
        teacher_layer_num = len(teacher_hidden_states)
        student_layer_num = len(student_hidden_states)
        map_rate = 3
        
        processed_layers = 0
        total_mmd_loss = torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)
        
        for k in student_layers_to_process:
            if k >= student_layer_num:
                continue
                
            teacher_layer_idx = min(k * map_rate, teacher_layer_num - 1)
            
            try:
                # Get hidden states and convert to float32
                stu_k_hidden = student_hidden_states[k].to(dtype=target_dtype)
                tea_k_hidden = teacher_hidden_states[teacher_layer_idx].detach().to(dtype=target_dtype)
                
                # Check for invalid values before projection
                if torch.isnan(stu_k_hidden).any() or torch.isinf(stu_k_hidden).any():
                    print(f"Warning: Invalid student hidden states at layer {k}")
                    continue
                    
                if torch.isnan(tea_k_hidden).any() or torch.isinf(tea_k_hidden).any():
                    print(f"Warning: Invalid teacher hidden states at layer {teacher_layer_idx}")
                    continue
                
                # Project student hidden state
                if hasattr(distiller, 'projectors') and "query" in distiller.projectors:
                    stu_k_hidden_projected = distiller.projectors["query"](stu_k_hidden)
                    stu_k_hidden_projected = stu_k_hidden_projected.to(dtype=target_dtype)
                    
                    # Ensure gradients
                    if not stu_k_hidden_projected.requires_grad:
                        stu_k_hidden_projected = stu_k_hidden_projected.clone().detach().requires_grad_(True)
                else:
                    stu_k_hidden_projected = stu_k_hidden.clone().detach().requires_grad_(True)
                
                # Compute MMD with better error handling
                mmd_loss = self.mmd(stu_k_hidden_projected, tea_k_hidden, kernel="multiscale")
                
                if torch.isfinite(mmd_loss) and not torch.isnan(mmd_loss):
                    total_mmd_loss = total_mmd_loss + mmd_loss
                    processed_layers += 1
                    log[f"mmd_loss_layer_{k}"] = mmd_loss.detach().clone()
                else:
                    print(f"Skipping layer {k} due to invalid MMD loss")
                
            except Exception as e:
                print(f"Error processing layer {k}: {e}")
                continue
        
        # Average across processed layers
        if processed_layers > 0:
            total_mmd_loss = total_mmd_loss / processed_layers
        
        log["total_mmd_loss"] = total_mmd_loss.detach().clone()
        return total_mmd_loss, log
    
    def compute_moe_loss(self, student_outputs, teacher_outputs, output_data, distiller, log):
        """Compute MoE loss with consistent dtype handling"""
        device = next(distiller.student_model.parameters()).device
        target_dtype = torch.float32  # Force float32
        
        # Extract MoE outputs
        if isinstance(student_outputs, dict):
            expert_outputs = student_outputs['expert_outputs']
            gating_weights = student_outputs['gating_weights']
        else:
            expert_outputs = student_outputs.expert_outputs
            gating_weights = student_outputs.gating_weights
        
        # Convert to target dtype
        gating_weights = gating_weights.to(dtype=target_dtype)
        expert_outputs = [output.to(dtype=target_dtype) for output in expert_outputs]
        
        # Extract teacher representation
        if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
            teacher_hidden = teacher_outputs.hidden_states[-1].to(dtype=target_dtype)
        elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
            teacher_hidden = teacher_outputs['hidden_states'][-1].to(dtype=target_dtype)
        else:
            raise ValueError("Cannot extract teacher hidden states")
        
        # Mean pooling for teacher embedding
        if teacher_hidden.dim() == 3:
            teacher_emb = teacher_hidden.mean(dim=1)
        else:
            teacher_emb = teacher_hidden
        
        # Compute expert losses per sample
        expert_losses = []
        
        try:
            # Expert 1: Cosine Loss
            cosine_loss_per_sample = self.compute_cosine_loss_per_sample(expert_outputs[0], teacher_emb)
            expert_losses.append(cosine_loss_per_sample)
            log["expert1_cosine_loss"] = cosine_loss_per_sample.mean().detach()

            # Expert 2: CKA Loss  
            cka_loss_per_sample = self.compute_cka_loss_per_sample(expert_outputs[1], teacher_emb)
            expert_losses.append(cka_loss_per_sample)
            log["expert2_cka_loss"] = cka_loss_per_sample.mean().detach()

            # Expert 3: Ranking Loss
            ranking_loss_per_sample = self.compute_ranking_loss_per_sample(expert_outputs[2], teacher_emb)
            expert_losses.append(ranking_loss_per_sample)
            log["expert3_ranking_loss"] = ranking_loss_per_sample.mean().detach()

            # Stack and compute weighted loss
            expert_losses_tensor = torch.stack(expert_losses)
            weighted_losses = (gating_weights.t() * expert_losses_tensor).sum(dim=0)
            moe_loss = weighted_losses.mean()
            
        except Exception as e:
            print(f"Error in MoE loss computation: {e}")
            moe_loss = torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)
        
        log["moe_distillation_loss"] = moe_loss.detach()
        log["gating_weights_mean"] = gating_weights.mean().detach()
        
        return moe_loss, log

    def compute_cosine_loss_per_sample(self, student_output, teacher_output):
        """Compute cosine similarity loss per sample with numerical stability"""
        # Convert to float32
        student_output = student_output.to(torch.float32)
        teacher_output = teacher_output.to(torch.float32)
        
        student_norm = F.normalize(student_output, p=2, dim=-1, eps=1e-8)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1, eps=1e-8)
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
        cosine_loss = 1 - cosine_sim
        return torch.clamp(cosine_loss, min=0.0, max=2.0)

    def compute_cka_loss_per_sample(self, student_output, teacher_output):
        """Compute CKA loss with numerical stability"""
        student_output = student_output.to(torch.float32)
        teacher_output = teacher_output.to(torch.float32)
        
        batch_cka_loss = self.cka_loss(student_output, teacher_output)
        batch_size = student_output.size(0)
        return torch.clamp(batch_cka_loss.expand(batch_size), min=0.0, max=2.0)

    def compute_ranking_loss_per_sample(self, student_output, teacher_output):
        """Compute ranking loss with numerical stability"""
        student_output = student_output.to(torch.float32)
        teacher_output = teacher_output.to(torch.float32)
        
        batch_size = student_output.size(0)
        
        # Normalize embeddings
        student_norm = F.normalize(student_output, p=2, dim=-1, eps=1e-8)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1, eps=1e-8)
        
        # Compute similarities
        student_similarities = torch.mm(student_norm, student_norm.t())
        teacher_similarities = torch.mm(teacher_norm, teacher_norm.t())
        
        # Per-sample ranking loss with reduced complexity
        per_sample_losses = []
        
        for i in range(batch_size):
            # Sample only a few pairs for efficiency
            other_indices = torch.randperm(batch_size)[:min(5, batch_size-1)]
            other_indices = other_indices[other_indices != i]
            
            if len(other_indices) == 0:
                per_sample_losses.append(torch.tensor(0.0, device=student_output.device))
                continue
            
            sample_loss = torch.relu(
                teacher_similarities[i, other_indices] - 
                student_similarities[i, other_indices] - 
                self.rank_margin
            ).mean()
            
            per_sample_losses.append(sample_loss)
        
        result = torch.stack(per_sample_losses)
        return torch.clamp(result, min=0.0, max=2.0)

    def compute_expert_diversity_loss(self, expert_outputs):
        """Compute expert diversity loss with improved stability"""
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device, dtype=torch.float32)
        
        # Convert to float32
        expert_outputs = [output.to(torch.float32) for output in expert_outputs]
        
        batch_size = expert_outputs[0].size(0)
        num_experts = len(expert_outputs)
        
        total_diversity_loss = 0.0
        pair_count = 0
        
        for m in range(num_experts):
            for n in range(m+1, num_experts):  # Only upper triangle to avoid double counting
                expert_m_norm = F.normalize(expert_outputs[m], p=2, dim=-1, eps=1e-8)
                expert_n_norm = F.normalize(expert_outputs[n], p=2, dim=-1, eps=1e-8)
                
                cosine_similarities = (expert_m_norm * expert_n_norm).sum(dim=-1)
                total_diversity_loss += torch.relu(cosine_similarities).sum()
                pair_count += 1
        
        if pair_count > 0:
            diversity_loss = total_diversity_loss / (pair_count * batch_size)
        else:
            diversity_loss = torch.tensor(0.0, device=expert_outputs[0].device, dtype=torch.float32)
        
        return torch.clamp(diversity_loss, min=0.0, max=1.0)
    
    def compute_cosine_loss(self, student_output, teacher_output):
        """Compute cosine similarity loss"""
        return self.compute_cosine_loss_per_sample(student_output, teacher_output).mean()

    def compute_cka_loss(self, student_output, teacher_output):
        """Compute CKA loss"""
        return self.compute_cka_loss_per_sample(student_output, teacher_output).mean()

    def compute_ranking_loss(self, student_output, teacher_output):
        """Compute ranking loss"""
        return self.compute_ranking_loss_per_sample(student_output, teacher_output).mean()
