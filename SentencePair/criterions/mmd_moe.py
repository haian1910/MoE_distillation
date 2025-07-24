import torch
import torch.nn as nn
from .cross_entropy_loss_moe import CrossEntropyLossMoE
from .various_divergence import VariousDivergence
import torch.nn.functional as F

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
        
        slen = SH.size(0)
        # Dropout on Hidden State Matching
        SH = F.normalize(SH, p=2, dim=1)
        TH = F.normalize(TH, p=2, dim=1)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
    
        return 1 - num/torch.sqrt(den1*den2)

class MMD_MOE(CrossEntropyLossMoE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized
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
        
        # Student forward pass - now with correct parameters
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            return_moe_outputs=True,
            output_hidden_states=True,  # This should now work
            return_dict=True,  # Ensure structured output
            labels=output_data["labels"]  # Add labels for loss computation
        )
        
        
        # Extract logits from the structured output
        logits = outputs['logits']
        
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
        
        # Compute MMD loss
        mmd_loss, log = self.compute_mmd_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("mmd_loss:", mmd_loss)

        # Compute MOE loss
        moe_loss, log = self.compute_moe_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        diversity_loss = self.compute_expert_diversity_loss(outputs['expert_outputs'])
        total_moe_loss = moe_loss + self.diversity_weight * diversity_loss
        print("moe_loss:", total_moe_loss)

        # Combine all losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * (mmd_loss + total_moe_loss)
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

        # Compute individual expert losses PER SAMPLE
        expert_losses = []
        
        # Expert 1: Cosine Loss - compute per sample
        expert1_output = expert_outputs[0]  # [batch_size, teacher_hidden_size]
        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(expert1_output, projected_teacher)
        #cosine_loss_per_sample = torch.zeros(expert1_output.size(0), device=expert1_output.device)
        expert_losses.append(cosine_loss_per_sample)
        log["expert1_cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()
        print("expert1_cosine_loss:", cosine_loss_per_sample.mean().detach().clone())

        # Expert 2: CKA Loss - compute per sample 
        #expert2_output = expert_outputs[1]  # [batch_size, teacher_hidden_size]
        cka_loss_per_sample = torch.zeros(expert2_output.size(0), device=expert2_output.device)
        cka_loss_per_sample = self.compute_cka_loss_per_sample(expert2_output, projected_teacher)

        expert_losses.append(cka_loss_per_sample)
        log["expert2_cka_loss"] = cka_loss_per_sample.mean().detach().clone()
        print("expert2_cka_loss:", cka_loss_per_sample.mean().detach().clone())

        # Expert 3: Ranking Loss - compute per sample (replaced Resim Loss)
        expert3_output = expert_outputs[2]  # [batch_size, teacher_hidden_size]
        ranking_loss_per_sample = self.compute_ranking_loss_per_sample(expert3_output, projected_teacher)
        #ranking_loss_per_sample = torch.zeros(expert3_output.size(0), device=expert3_output.device)
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
                        teacher_similarities[i, j] - student_similarities[i, j] - self.rank_margin
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
        
        # Flatten the sequence dimension: [batch_size * seq_len, hidden_dim]
        x_flat = x.view(-1, x.size(-1))
        y_flat = y.view(-1, y.size(-1))
        
        # Compute pairwise distances
        xx = torch.mm(x_flat, x_flat.t())
        yy = torch.mm(y_flat, y_flat.t())
        zz = torch.mm(x_flat, y_flat.t())
        
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX = torch.zeros(xx.shape, device=device)
        YY = torch.zeros(yy.shape, device=device)
        XY = torch.zeros(zz.shape, device=device)

        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1

        elif kernel == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)

    def compute_mmd_loss(
    self, outputs, teacher_outputs, output_data, distiller, log
):
        """
        Compute MMD loss between student and teacher hidden states
        
        Args:
            outputs: Student model outputs (dictionary with 'hidden_states' key)
            teacher_outputs: Teacher model outputs (dictionary with 'hidden_states' key)
            output_data: Not used in this function
            distiller: Distiller object containing projectors
            log: Logging dictionary
        """
        total_mmd_loss = 0.0
        
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
        

        
        # Define the layers to process (adjust these based on your student model architecture)
        # For BERT-base, layers are typically 0-11, so using last few layers
        student_layers_to_process = [10, 11]  # Can modify based on your needs

        teacher_layer_num = len(teacher_hidden_states)
        student_layer_num = len(student_hidden_states)
        
        # Calculate mapping rate to align teacher and student layers
        map_rate = 3
        
        
        for k in student_layers_to_process:
            if k >= student_layer_num:
                print(f"Warning: Student layer {k} doesn't exist (max: {student_layer_num-1})")
                continue
                
            # Calculate corresponding teacher layer
            teacher_layer_idx = min(k * map_rate, teacher_layer_num - 1)
            
            try:
                # Get student hidden state: [batch_size, seq_len, hidden_dim]
                stu_k_hidden = student_hidden_states[k]
                
                # Project student hidden state to teacher's embedding space
                if hasattr(distiller, 'projectors') and "query" in distiller.projectors:
                    # Apply projection: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, teacher_hidden_dim]
                    stu_k_hidden_projected = distiller.projectors["query"](stu_k_hidden)
                else:
                    # If no projector available, you might need to add a simple linear projection
                    print("Warning: No 'query' projector found. Using original student hidden states.")
                    stu_k_hidden_projected = stu_k_hidden
                
                # Get teacher hidden state: [batch_size, seq_len, hidden_dim]
                tea_k_hidden = teacher_hidden_states[teacher_layer_idx]
                
       
                # Compute MMD loss between the hidden states
                mmd_loss = self.mmd(stu_k_hidden_projected, tea_k_hidden, kernel="multiscale")
                total_mmd_loss += mmd_loss
                
                
                # Log individual layer losses
                
            except Exception as e:
                print(f"Error processing layer {k}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average the MMD loss across layers
        if len(student_layers_to_process) > 0:
            total_mmd_loss = total_mmd_loss / len(student_layers_to_process)
        
        log["total_mmd_loss"] = total_mmd_loss.detach().clone()
        
        return total_mmd_loss, log
