import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
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


class MOE(MoE_STSLoss):
    """
    MoE-based loss for STS (Semantic Textual Similarity) tasks with knowledge distillation.
    """
    def __init__(self, args) -> None:
        super().__init__(args)
        
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
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute loss for STS tasks with MoE model and knowledge distillation.
        """
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        target = output_data["labels"]
        loss_mse = nn.MSELoss()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        # Convert target to BF16 if needed
        if self.use_bf16 and target.dtype == torch.float32:
            target = target.to(torch.bfloat16)

        # Student model forward pass with MoE outputs
        student_outputs = model(
            input_ids=input_data['input_ids'],
            attention_mask=input_data['attention_mask'],
            token_type_ids=input_data['token_type_ids'] if 'token_type_ids' in input_data else None,
            return_moe_outputs=True,
            labels=target
        )
        print(dir(student_outputs))
        
        # FIXED: Use dot notation instead of dictionary indexing
        # Check if student_outputs has scores attribute, otherwise use logits
        if hasattr(student_outputs, 'scores'):
            predictions = student_outputs.scores
        elif hasattr(student_outputs, 'logits'):
            predictions = student_outputs.logits
        else:
            # Try to access as dictionary if it's actually a dict
            if isinstance(student_outputs, dict):
                predictions = student_outputs.get("scores", student_outputs.get("logits"))
            else:
                raise AttributeError("student_outputs does not have 'scores' or 'logits' attribute")
        
        labels = output_data["labels"].to(dtype)
        loss_sts = loss_mse(predictions, labels)
        log = {}

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"] if "teacher_input_ids" in input_data else input_data["input_ids"],
                attention_mask=input_data["teacher_attention_mask"] if "teacher_attention_mask" in input_data else input_data["attention_mask"],
                output_hidden_states=True,
                return_dict=True
            )
        
        # Compute MoE distillation loss
        # Compute distillation loss
        
        kd_loss, log = self.compute_moe_loss(
            student_outputs, teacher_outputs, output_data, distiller, log
        )
        print("moe_loss:", kd_loss)
        
        # Compute expert diversity loss - Fixed: use dot notation instead of dictionary indexing
        diversity_loss = self.compute_expert_diversity_loss(student_outputs.expert_outputs)
        log["diversity_loss"] = diversity_loss.detach().clone()
        print("diversity_loss:", diversity_loss.detach().clone())
        
        # Combine all losses
        total_kd_loss = kd_loss + self.diversity_weight * diversity_loss
      
        loss = (1.0 - self.kd_rate) * loss_sts + self.kd_rate * total_kd_loss
        log["loss"] = loss.detach().clone()  # Store as tensor for distributed logging


        
        return loss, logging_output
        

    def compute_sts_loss(self, predictions, target):
        """Compute the primary STS loss"""
        if self.loss_type == "mse":
            return nn.MSELoss()(predictions, target)
        elif self.loss_type == "mae":
            return nn.L1Loss()(predictions, target)
        elif self.loss_type == "huber":
            return nn.SmoothL1Loss()(predictions, target)
        else:
            return nn.MSELoss()(predictions, target)

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
        log["gating_weights_mean"] = gating_weights.mean(dim=0).detach().clone()
        
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
                        teacher_similarities[i, j] - student_similarities[i, j] - self.rank_margin
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
