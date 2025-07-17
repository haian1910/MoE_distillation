import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss_moe import CrossEntropyLossMoE


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
        print(SH)
        print(TH)
        
        slen = SH.size(0)
        # Dropout on Hidden State Matching
        SH = F.normalize(SH, p=2, dim=1)
        TH = F.normalize(TH, p=2, dim=1)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        print('SH after centering:', SH)
        print('TH after centering:', TH)

        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        print('num:', num)
        print('den1:', den1)
        print('den2:', den2)
        return 1 - num/torch.sqrt(den1*den2)

class MOE(CrossEntropyLossMoE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        # Initialize CKA loss module
        self.cka_loss = CKALoss(eps=getattr(args, 'cka_eps', 1e-8))
        
        # Parameters for ranking loss
        self.rank_margin = getattr(args, 'rank_margin', 0.1)
        
        # Parameters for expert diversity loss
        self.diversity_weight = getattr(args, 'diversity_weight', 1)
    
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
        
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            return_moe_outputs=True
        )
        if isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            raise TypeError("Model outputs must be a dictionary with 'logits' or a tensor")
        log = {}
        
        # Compute cross-entropy loss with ground-truth labels
        loss = self.compute_cross_entropy_loss(
            logits, output_data["labels"]
        )[0]

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True,  # This is crucial for getting hidden states
                return_dict=True  # Ensure we get a structured output
            )
        # Compute distillation loss
        kd_loss, log = self.compute_moe_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("moe_loss:", kd_loss)
        
        # Compute expert diversity loss
        diversity_loss = self.compute_expert_diversity_loss(outputs['expert_outputs'])
        log["diversity_loss"] = diversity_loss.detach().clone()
        print("diversity_loss:", diversity_loss.detach().clone())
        
        # Combine all losses
        total_kd_loss = kd_loss + self.diversity_weight * diversity_loss
      
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * total_kd_loss
        log["loss"] = loss.detach().clone()  # Store as tensor for distributed logging

        # Compute accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"]
        )
        log["accuracy"] = accuracy  # This should already be a tensor from compute_accuracy

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss, logging_output

    def compute_moe_loss(
    self, outputs, teacher_outputs, output_data, distiller, log
):
        """
        Compute the Mixture of Experts (MoE) distillation loss for three experts.
        - Expert 1: Cosine Loss
        - Expert 2: CKA Loss 
        - Expert 3: Ranking Loss (replaced Resim Loss)
        The final MoE loss is weighted by the gating network outputs.
        """
        # Get device for tensor creation
        device = next(distiller.student_model.parameters()).device
        
        # Get MoE outputs from student model
        expert_outputs = outputs['expert_outputs']  # List of 3 expert outputs
        gating_weights = outputs['gating_weights']  # [batch_size, num_experts]
        
        # Extract student CLS representation (before MoE)
        student_cls = outputs['cls_representation']  # [batch_size, 768]
        
        # Extract teacher hidden states - Handle different model architectures
        if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
            # For LLM2Vec model loaded with AutoModelForSequenceClassification
            teacher_hidden = teacher_outputs.hidden_states[-1]  # Last layer hidden states
        elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
            # If teacher_outputs is a dict with hidden_states key
            teacher_hidden = teacher_outputs['hidden_states'][-1]
        
        # Extract CLS token representation from teacher
        if teacher_hidden.dim() == 3:  # [batch_size, sequence_length, hidden_size]
            teacher_emb = teacher_hidden.mean(dim=1)  # Take mean across sequence length
        elif teacher_hidden.dim() == 2:  # [batch_size, hidden_size] - already CLS representation
            teacher_emb = teacher_hidden
        else:
            raise ValueError(f"Unexpected dimension for teacher_hidden: {teacher_hidden.shape}")


        projected_teacher = teacher_emb  # [batch_size, teacher_hidden_size]
        print("teacher_embedding:", projected_teacher)

        # Compute individual expert losses PER SAMPLE
        expert_losses = []
        
        # Expert 1: Cosine Loss - compute per sample
        expert1_output = expert_outputs[0]  # [batch_size, teacher_hidden_size]
        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(expert1_output, projected_teacher)
        expert_losses.append(cosine_loss_per_sample)
        log["expert1_cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()
        print("expert1_cosine_loss:", cosine_loss_per_sample.mean().detach().clone())

        # Expert 2: CKA Loss - compute per sample 
        expert2_output = expert_outputs[1]  # [batch_size, teacher_hidden_size]
        cka_loss_per_sample = self.compute_cka_loss_per_sample(expert2_output, projected_teacher)
        
        expert_losses.append(cka_loss_per_sample)
        log["expert2_cka_loss"] = cka_loss_per_sample.mean().detach().clone()
        print("expert2_cka_loss:", cka_loss_per_sample.mean().detach().clone())

        # Expert 3: Ranking Loss - compute per sample (replaced Resim Loss)
        expert3_output = expert_outputs[2]  # [batch_size, teacher_hidden_size]
        ranking_loss_per_sample = self.compute_ranking_loss_per_sample(expert3_output, projected_teacher)
        expert_losses.append(ranking_loss_per_sample)
        log["expert3_ranking_loss"] = ranking_loss_per_sample.mean().detach().clone()
        print("expert3_ranking_loss:", ranking_loss_per_sample.mean().detach().clone())

        # Compute weighted MoE loss using gating weights
        # gating_weights: [batch_size, num_experts]
        # expert_losses: [num_experts, batch_size] (each is per-sample loss)
        
        # Stack expert losses: [num_experts, batch_size]
        expert_losses_tensor = torch.stack(expert_losses)  # [num_experts, batch_size]
        
        # Compute per-sample weighted loss
        # gating_weights.t(): [num_experts, batch_size]
        # Element-wise multiplication and sum over experts
        weighted_losses = (gating_weights.t() * expert_losses_tensor).sum(dim=0)  # [batch_size]
        
        # Take mean across batch to get final scalar loss
        moe_loss = weighted_losses.mean()  # scalar
        
        log["moe_loss"] = moe_loss.detach().clone()
        log["gating_weights_mean"] = gating_weights.mean(dim=0).detach().clone()  # Average gating weights as tensor
        
        return moe_loss, log

    def compute_ranking_loss_per_sample(self, student_output, teacher_output):
        """
        Compute margin-based ranking loss per sample.
        L_rank = sum_i sum_j max(0, sim(z_i^s, z_j^s) - sim(z_i^t, z_j^t) + δ)
        
        Args:
            student_output: [batch_size, hidden_size] student embeddings
            teacher_output: [batch_size, hidden_size] teacher embeddings
            
        Returns:
            [batch_size] tensor of per-sample losses
        """
        batch_size = student_output.size(0)
        
        # Normalize embeddings for similarity computation
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        
        # Compute all pairwise similarities
        student_similarities = torch.mm(student_norm, student_norm.t())  # [batch_size, batch_size]
        teacher_similarities = torch.mm(teacher_norm, teacher_norm.t())  # [batch_size, batch_size]
        
        # Compute per-sample ranking loss
        per_sample_losses = []
        
        for i in range(batch_size):
            sample_loss = 0.0
            count = 0
            
            for j in range(batch_size):
                if i != j:  # Skip self-similarity
                    # Compute ranking loss for pair (i, j)
                    loss_term = torch.relu(
                        teacher_similarities[i, j] - student_similarities[i, j] + self.rank_margin
                    )
                    sample_loss += loss_term
                    count += 1
            
            # Average over all pairs for this sample
            if count > 0:
                sample_loss = sample_loss / count
            else:
                sample_loss = torch.tensor(0.0, device=student_output.device)
                print(f"Warning: No valid pairs for sample {i}, setting loss to zero.")
            
            per_sample_losses.append(sample_loss)
        
        return torch.stack(per_sample_losses)  # [batch_size]

    def compute_expert_diversity_loss(self, expert_outputs):
        """
        Compute expert diversity loss to encourage different experts to produce diverse representations.
        L_div = sum_i sum_m sum_n cos(h_i^(m), h_i^(n)) for m != n
        
        Args:
            expert_outputs: List of [batch_size, hidden_size] tensors, one for each expert
            
        Returns:
            Scalar tensor representing the diversity loss
        """
        batch_size = expert_outputs[0].size(0)
        num_experts = len(expert_outputs)
        
        total_diversity_loss = 0.0
        pair_count = 0
        
        # Iterate over all pairs of experts
        for m in range(num_experts):
            for n in range(num_experts):
                if m != n:  # Only consider different experts
                    # Normalize expert outputs
                    expert_m_norm = F.normalize(expert_outputs[m], p=2, dim=-1)
                    expert_n_norm = F.normalize(expert_outputs[n], p=2, dim=-1)
                    
                    # Compute cosine similarity for each sample
                    cosine_similarities = (expert_m_norm * expert_n_norm).sum(dim=-1)  # [batch_size]
                    
                    # Sum over all samples in the batch
                    total_diversity_loss += torch.relu(cosine_similarities).sum()
                    pair_count += 1
        
        # Average over all expert pairs and all samples
        if pair_count > 0:
            diversity_loss = total_diversity_loss / (pair_count * batch_size)
        else:
            diversity_loss = torch.tensor(0.0, device=expert_outputs[0].device)
        
        return diversity_loss

    def compute_cosine_loss_per_sample(self, student_output, teacher_output):
        """
        Compute cosine similarity loss per sample: L_cosine = 1 - s_x . t_x
        Returns: [batch_size] tensor of per-sample losses
        """
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)  # [batch_size]
        cosine_loss = 1 - cosine_sim  # [batch_size]
        return cosine_loss

    def compute_cka_loss_per_sample(self, student_output, teacher_output):
        """
        Compute CKA loss for the full batch to preserve overall structure
        CKA measures structural similarity across the entire batch
        Returns: [batch_size] tensor where each element is the same batch-level CKA loss
        """
        # Compute CKA loss for the full batch
        batch_cka_loss = self.cka_loss(student_output, teacher_output)
        
        # Return the same loss value for each sample in the batch
        # This ensures that the gating mechanism still works at the sample level
        # while the CKA loss captures batch-level structural similarity
        batch_size = student_output.size(0)
        return batch_cka_loss.expand(batch_size)

    # Keep original methods for backward compatibility
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
