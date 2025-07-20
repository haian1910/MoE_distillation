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
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
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
        self.cka_loss = CKALoss(eps=getattr(args, 'cka_eps', 1e-8))
        
        # Parameters for ranking loss
        self.rank_margin = getattr(args, 'rank_margin', 0.1)
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
        
        # Student forward pass - now with correct parameters for STS
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            return_moe_outputs=True,
            output_hidden_states=True,
            return_dict=True,
            labels=output_data["labels"]  # Add labels for loss computation
        )
        
        # Extract predictions for STS task
        if hasattr(outputs, 'scores') and outputs.scores is not None:
            predictions = outputs.scores
        elif hasattr(outputs, 'logits') and outputs.logits is not None:
            predictions = outputs.logits
        else:
            # Try to access as dictionary if it's actually a dict
            if isinstance(outputs, dict):
                predictions = outputs.get("scores", outputs.get("logits"))
            else:
                raise AttributeError("Student outputs does not have 'scores' or 'logits' attribute")
        
        log = {}
        
        # FIXED: Ensure consistent dtype - use the same dtype as predictions
        target_dtype = predictions.dtype
        device = predictions.device
        
        # Convert labels to the same dtype as predictions
        labels = output_data["labels"].to(dtype=target_dtype, device=device)
        
        # Ensure predictions have the correct shape
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)  # Remove last dimension if it's 1
        
        # Compute MSE loss with consistent dtype
        loss_sts = F.mse_loss(predictions, labels)
        
        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True,  # This is crucial for getting hidden states
                return_dict=True  # Ensure we get a structured output
            )
        
        # Compute MMD loss
        try:
            mmd_loss, log = self.compute_mmd_loss(
                outputs, teacher_outputs, output_data, distiller, log
            )
            print("mmd_loss:", mmd_loss)

            # FIXED: Ensure mmd_loss has the same dtype as other losses
            if not isinstance(mmd_loss, torch.Tensor):
                print("Warning: mmd_loss is not a tensor, using zero loss")
                mmd_loss = torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)
            else:
                # Convert to target dtype if needed
                mmd_loss = mmd_loss.to(dtype=target_dtype)
                if not mmd_loss.requires_grad:
                    print("Warning: mmd_loss doesn't require gradients")
                    mmd_loss = mmd_loss.clone().detach().requires_grad_(True)
        except Exception as e:
            print(f"Error computing MMD loss: {e}")
            import traceback
            traceback.print_exc()
            # Use zero loss if MMD computation fails
            mmd_loss = torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)

        # Compute MOE loss
        moe_loss, log = self.compute_moe_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("moe_loss:", moe_loss)

        # Compute expert diversity loss - Fixed: use dot notation instead of dictionary indexing
        diversity_loss = self.compute_expert_diversity_loss(outputs.expert_outputs)
    
        # Combine all losses
        total_moe_loss = moe_loss + self.diversity_weight * diversity_loss

        

        # FIXED: Convert scalar coefficients to target dtype for consistent computation
        kd_rate = torch.tensor(self.kd_rate, device=device, dtype=target_dtype)
        one_minus_kd_rate = torch.tensor(1.0 - self.kd_rate, device=device, dtype=target_dtype)
        
        # Combine STS loss with MMD distillation loss - now all in same dtype
        loss = one_minus_kd_rate * loss_sts + kd_rate * (mmd_loss + total_moe_loss)

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss, logging_output

    @staticmethod
    def mmd(x, y, kernel="multiscale"):
        """Empirical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P [batch_size, seq_len, hidden_dim]
            y: second sample, distribution Q [batch_size, seq_len, hidden_dim]
            kernel: kernel type such as "multiscale" or "rbf"
        """
        device = x.device
        target_dtype = x.dtype  # Use input dtype for consistency
        
        # Check for NaN or Inf values
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in x")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: NaN or Inf detected in y")
            y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure both tensors have the same dtype
        y = y.to(dtype=target_dtype)
        
        # Ensure input tensors are contiguous
        x = x.contiguous()
        y = y.contiguous()
        
        # Flatten the sequence dimension: [batch_size * seq_len, hidden_dim]
        x_flat = x.view(-1, x.size(-1))
        y_flat = y.view(-1, y.size(-1))
        
        # Add small epsilon to avoid division by zero - use target dtype
        eps = torch.tensor(1e-8, device=device, dtype=target_dtype)
        
        # Compute pairwise distances with numerical stability
        try:
            xx = torch.mm(x_flat, x_flat.t())
            yy = torch.mm(y_flat, y_flat.t())
            zz = torch.mm(x_flat, y_flat.t())
            
            rx = (xx.diag().unsqueeze(0).expand_as(xx))
            ry = (yy.diag().unsqueeze(0).expand_as(yy))

            dxx = torch.clamp(rx.t() + rx - 2. * xx, min=eps)
            dyy = torch.clamp(ry.t() + ry - 2. * yy, min=eps)
            dxy = torch.clamp(rx.t() + ry - 2. * zz, min=eps)

            XX = torch.zeros(xx.shape, device=device, dtype=target_dtype)
            YY = torch.zeros(yy.shape, device=device, dtype=target_dtype)
            XY = torch.zeros(zz.shape, device=device, dtype=target_dtype)

            if kernel == "multiscale":
                bandwidth_range = [0.2, 0.5, 0.9, 1.3]
                for a in bandwidth_range:
                    # Convert to target dtype
                    a_tensor = torch.tensor(a**2, device=device, dtype=target_dtype)
                    XX += a_tensor / (a_tensor + dxx + eps)
                    YY += a_tensor / (a_tensor + dyy + eps)
                    XY += a_tensor / (a_tensor + dxy + eps)

            elif kernel == "rbf":
                bandwidth_range = [10, 15, 20, 50]
                for a in bandwidth_range:
                    # Convert to target dtype and clamp exponential arguments
                    a_tensor = torch.tensor(a, device=device, dtype=target_dtype)
                    exp_xx = torch.clamp(-0.5 * dxx / a_tensor, min=-50, max=50)
                    exp_yy = torch.clamp(-0.5 * dyy / a_tensor, min=-50, max=50)
                    exp_xy = torch.clamp(-0.5 * dxy / a_tensor, min=-50, max=50)
                    
                    XX += torch.exp(exp_xx)
                    YY += torch.exp(exp_yy)
                    XY += torch.exp(exp_xy)

            mmd_result = torch.mean(XX + YY - 2. * XY)
            
            # Final check for NaN/Inf
            if torch.isnan(mmd_result) or torch.isinf(mmd_result):
                print("Warning: MMD result is NaN or Inf, returning zero")
                return torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)
                
            return mmd_result
            
        except RuntimeError as e:
            print(f"Runtime error in MMD computation: {e}")
            return torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)

    def compute_mmd_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        """
        Compute MMD loss between student and teacher hidden states
        """
        # Handle both dictionary and object outputs for student
        if isinstance(outputs, dict):
            student_hidden_states = outputs.get('hidden_states', None)
        else:
            student_hidden_states = getattr(outputs, 'hidden_states', None)
        
        # Handle both dictionary and object outputs for teacher
        if isinstance(teacher_outputs, dict):
            teacher_hidden_states = teacher_outputs.get('hidden_states', None)
        else:
            teacher_hidden_states = getattr(teacher_outputs, 'hidden_states', None)
        
        # Check if hidden states are available
        if student_hidden_states is None:
            raise ValueError("Student model outputs don't contain hidden_states. Make sure to call model with output_hidden_states=True")
        
        if teacher_hidden_states is None:
            raise ValueError("Teacher model outputs don't contain hidden_states. Make sure to call model with output_hidden_states=True")
        
        # Get target dtype from student outputs
        if isinstance(outputs, dict):
            sample_tensor = next(iter(outputs.values()))
        else:
            sample_tensor = outputs.logits if hasattr(outputs, 'logits') else student_hidden_states[0]
        
        target_dtype = sample_tensor.dtype
        device = sample_tensor.device
        
        # Define the layers to process
        student_layers_to_process = [9, 10]
        
        teacher_layer_num = len(teacher_hidden_states)
        student_layer_num = len(student_hidden_states)
        
        # Calculate mapping rate to align teacher and student layers
        map_rate = 3
        
        processed_layers = 0
        total_mmd_loss = torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)
        
        for k in student_layers_to_process:
            if k >= student_layer_num:
                print(f"Warning: Student layer {k} doesn't exist (max: {student_layer_num-1})")
                continue
                
            # Calculate corresponding teacher layer
            teacher_layer_idx = min(k * map_rate, teacher_layer_num - 1)
            
            try:
                # Get student hidden state and ensure correct dtype
                stu_k_hidden = student_hidden_states[k].to(dtype=target_dtype)
                
                # Ensure student hidden state requires gradients
                if not stu_k_hidden.requires_grad:
                    stu_k_hidden = stu_k_hidden.clone().detach().requires_grad_(True)
                
                # Project student hidden state to teacher's embedding space
                if hasattr(distiller, 'projectors') and "query" in distiller.projectors:
                    stu_k_hidden_projected = distiller.projectors["query"](stu_k_hidden)
                    # Ensure projected tensor has correct dtype
                    stu_k_hidden_projected = stu_k_hidden_projected.to(dtype=target_dtype)
                else:
                    print("Warning: No 'query' projector found. Using original student hidden states.")
                    stu_k_hidden_projected = stu_k_hidden
                
                # Get teacher hidden state and ensure correct dtype
                tea_k_hidden = teacher_hidden_states[teacher_layer_idx].detach().to(dtype=target_dtype)
                
                # Compute MMD loss between the hidden states
                mmd_loss = self.mmd(stu_k_hidden_projected, tea_k_hidden, kernel="multiscale")
                
                # Ensure MMD loss is valid and has correct dtype
                if torch.isnan(mmd_loss) or torch.isinf(mmd_loss):
                    print(f"Warning: MMD loss for layer {k} is NaN or Inf, skipping")
                    continue
                
                mmd_loss = mmd_loss.to(dtype=target_dtype)
                total_mmd_loss = total_mmd_loss + mmd_loss
                processed_layers += 1
                
                # Log individual layer losses
                log[f"mmd_loss_layer_{k}"] = mmd_loss.detach().clone()
                
            except Exception as e:
                print(f"Error processing layer {k}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average the MMD loss across layers
        if processed_layers > 0:
            processed_layers_tensor = torch.tensor(processed_layers, device=device, dtype=target_dtype)
            total_mmd_loss = total_mmd_loss / processed_layers_tensor
        else:
            # If no layers were processed, return zero loss
            total_mmd_loss = torch.tensor(0.0, device=device, dtype=target_dtype, requires_grad=True)
        
        log["total_mmd_loss"] = total_mmd_loss.detach().clone()
        
        return total_mmd_loss, log
    
    def compute_moe_loss(self, student_outputs, teacher_outputs, output_data, distiller, log):
        """
        Compute the Mixture of Experts (MoE) distillation loss for three experts.
        - Expert 1: Cosine Loss
        - Expert 2: CKA Loss 
        - Expert 3: Ranking Loss
        The final MoE loss is weighted by the gating network outputs.
        """
        # Get device for tensor creation
        device = next(distiller.student_model.parameters()).device
        
        # Get MoE outputs from student model - Fixed: use dot notation consistently
        if isinstance(student_outputs, dict):
            expert_outputs = student_outputs['expert_outputs']
            gating_weights = student_outputs['gating_weights']
            student_cls = student_outputs.get('cls_representation', None)
        else:
            expert_outputs = student_outputs.expert_outputs
            gating_weights = student_outputs.gating_weights
            student_cls = getattr(student_outputs, 'cls_representation', None)
        
        # Extract teacher hidden states
        if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
            teacher_hidden = teacher_outputs.hidden_states[-1]
        elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
            teacher_hidden = teacher_outputs['hidden_states'][-1]
        else:
            raise ValueError("Cannot extract teacher hidden states")
        
        # Extract representation from teacher (mean pooling for STS)
        if teacher_hidden.dim() == 3:  # [batch_size, sequence_length, hidden_size]
            teacher_emb = teacher_hidden.mean(dim=1)  # Mean pooling across sequence
        elif teacher_hidden.dim() == 2:  # [batch_size, hidden_size]
            teacher_emb = teacher_hidden
        else:
            raise ValueError(f"Unexpected dimension for teacher_hidden: {teacher_hidden.shape}")

        # Compute individual expert losses PER SAMPLE
        expert_losses = []
        
        # Expert 1: Cosine Loss
        expert1_output = expert_outputs[0]
        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(expert1_output, teacher_emb)
        expert_losses.append(cosine_loss_per_sample)
        log["expert1_cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()

        # Expert 2: CKA Loss
        expert2_output = expert_outputs[1]
        cka_loss_per_sample = self.compute_cka_loss_per_sample(expert2_output, teacher_emb)
        expert_losses.append(cka_loss_per_sample)
        log["expert2_cka_loss"] = cka_loss_per_sample.mean().detach().clone()

        # Expert 3: Ranking Loss
        expert3_output = expert_outputs[2]
        ranking_loss_per_sample = self.compute_ranking_loss_per_sample(expert3_output, teacher_emb)
        expert_losses.append(ranking_loss_per_sample)
        log["expert3_ranking_loss"] = ranking_loss_per_sample.mean().detach().clone()

        # Compute weighted MoE loss using gating weights
        expert_losses_tensor = torch.stack(expert_losses)  # [num_experts, batch_size]
        
        # Compute per-sample weighted loss
        weighted_losses = (gating_weights.t() * expert_losses_tensor).sum(dim=0)  # [batch_size]
        
        # Take mean across batch to get final scalar loss
        moe_loss = weighted_losses.mean()
        
        log["moe_distillation_loss"] = moe_loss.detach().clone()
        # FIXED: Convert multi-element tensor to scalar by taking mean
        log["gating_weights_mean"] = gating_weights.mean().detach().clone()  # Take overall mean instead of per-expert mean
        
        return moe_loss, log

    def compute_cosine_loss_per_sample(self, student_output, teacher_output):
        """Compute cosine similarity loss per sample"""
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
        cosine_loss = 1 - cosine_sim
        return cosine_loss

    def compute_cka_loss_per_sample(self, student_output, teacher_output):
        """Compute CKA loss for the batch (returns same value for all samples)"""
        batch_cka_loss = self.cka_loss(student_output, teacher_output)
        batch_size = student_output.size(0)
        return batch_cka_loss.expand(batch_size)

    def compute_ranking_loss_per_sample(self, student_output, teacher_output):
        """Compute margin-based ranking loss per sample"""
        batch_size = student_output.size(0)
        
        # Normalize embeddings
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        
        # Compute all pairwise similarities
        student_similarities = torch.mm(student_norm, student_norm.t())
        teacher_similarities = torch.mm(teacher_norm, teacher_norm.t())
        
        # Compute per-sample ranking loss
        per_sample_losses = []
        
        for i in range(batch_size):
            sample_loss = 0.0
            count = 0
            
            for j in range(batch_size):
                if i != j:
                    loss_term = torch.relu(
                        teacher_similarities[i, j] - student_similarities[i, j] + self.rank_margin
                    )
                    sample_loss += loss_term
                    count += 1
            
            if count > 0:
                sample_loss = sample_loss / count
            else:
                sample_loss = torch.tensor(0.0, device=student_output.device)
            
            per_sample_losses.append(sample_loss)
        
        return torch.stack(per_sample_losses)

    def compute_expert_diversity_loss(self, expert_outputs):
        """Compute expert diversity loss"""
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)
        
        batch_size = expert_outputs[0].size(0)
        num_experts = len(expert_outputs)
        
        total_diversity_loss = 0.0
        pair_count = 0
        
        for m in range(num_experts):
            for n in range(num_experts):
                if m != n:
                    expert_m_norm = F.normalize(expert_outputs[m], p=2, dim=-1)
                    expert_n_norm = F.normalize(expert_outputs[n], p=2, dim=-1)
                    
                    cosine_similarities = (expert_m_norm * expert_n_norm).sum(dim=-1)
                    total_diversity_loss += torch.relu(cosine_similarities).sum()
                    pair_count += 1
        
        if pair_count > 0:
            diversity_loss = total_diversity_loss / (pair_count * batch_size)
        else:
            diversity_loss = torch.tensor(0.0, device=expert_outputs[0].device)
        
        return diversity_loss
    
    def compute_cosine_loss(self, student_output, teacher_output):
        """
        Compute cosine similarity loss: L_cosine = sum_x (1 - s_x . t_x)
        """
        return self.compute_cosine_loss_per_sample(student_output, teacher_output).mean()

    def compute_cka_loss(self, student_output, teacher_output):
        """
        Compute CKA loss: L_cka using Centered Kernel Alignment
        """
        return self.compute_cka_loss_per_sample(student_output, teacher_output).mean()

    def compute_ranking_loss(self, student_output, teacher_output):
        """
        Compute ranking loss: L_rank using margin-based ranking loss
        """
        return self.compute_ranking_loss_per_sample(student_output, teacher_output).mean()
