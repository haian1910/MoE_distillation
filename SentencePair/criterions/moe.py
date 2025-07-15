import torch
import torch.nn.functional as F
from .cross_entropy_loss_moe import CrossEntropyLossMoE

class MOE(CrossEntropyLossMoE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
    
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
                output_hidden_states=True
            )
        
        # Compute distillation loss
        kd_loss, log = self.compute_moe_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
      
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
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
        - Expert 2: Similarity Loss  
        - Expert 3: Resim Loss
        The final MoE loss is weighted by the gating network outputs.
        """
        # Get device for tensor creation
        device = next(distiller.student_model.parameters()).device
        
        # Get MoE outputs from student model
        expert_outputs = outputs['expert_outputs']  # List of 3 expert outputs
        gating_weights = outputs['gating_weights']  # [batch_size, num_experts]
        
        # Extract student CLS representation (before MoE)
        student_cls = outputs['cls_representation']  # [batch_size, 768]
        
        # Extract teacher hidden states
        teacher_hidden = teacher_outputs['hidden_states'][-1] if 'hidden_states' in teacher_outputs else teacher_outputs
        if teacher_hidden.dim() == 3:  # [batch_size, sequence_length, hidden_size]
            teacher_cls = teacher_hidden[:, 0, :]  # Take [CLS] token
        elif teacher_hidden.dim() == 2:  # [batch_size, hidden_size]
            teacher_cls = teacher_hidden  # Already CLS representation
        else:
            raise ValueError("Unexpected dimension for teacher_hidden")

        # Project teacher representation to match student dimension using MoE projections
        # Each expert output should be [batch_size, teacher_hidden_size] to match teacher
        projected_teacher = teacher_cls  # [batch_size, teacher_hidden_size]
        
        # Compute individual expert losses PER SAMPLE
        expert_losses = []
        
        # Expert 1: Cosine Loss - compute per sample
        expert1_output = expert_outputs[0]  # [batch_size, teacher_hidden_size]
        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(expert1_output, projected_teacher)
        expert_losses.append(cosine_loss_per_sample)
        log["expert1_cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()

        # Expert 2: Similarity Loss - compute per sample
        expert2_output = expert_outputs[1]  # [batch_size, teacher_hidden_size]
        sim_loss_per_sample = self.compute_similarity_loss_per_sample(expert2_output, projected_teacher)
        expert_losses.append(sim_loss_per_sample)
        log["expert2_sim_loss"] = sim_loss_per_sample.mean().detach().clone()

        # Expert 3: Resim Loss - compute per sample
        expert3_output = expert_outputs[2]  # [batch_size, teacher_hidden_size]
        resim_loss_per_sample = self.compute_resim_loss_per_sample(expert3_output, projected_teacher)
        expert_losses.append(resim_loss_per_sample)
        log["expert3_resim_loss"] = resim_loss_per_sample.mean().detach().clone()

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

    def compute_similarity_loss_per_sample(self, student_output, teacher_output):
        """
        Compute similarity matrix loss per sample
        For per-sample loss, we compute MSE between each sample's similarity with all other samples
        Returns: [batch_size] tensor of per-sample losses
        """
        batch_size = student_output.size(0)
        
        # Compute similarity matrices
        student_self_sim = torch.mm(student_output, student_output.t())  # [batch_size, batch_size]
        teacher_self_sim = torch.mm(teacher_output, teacher_output.t())  # [batch_size, batch_size]
        
        # Compute per-sample loss as MSE of each row
        per_sample_losses = []
        for i in range(batch_size):
            sample_loss = F.mse_loss(student_self_sim[i], teacher_self_sim[i], reduction='mean')
            per_sample_losses.append(sample_loss)
        
        return torch.stack(per_sample_losses)  # [batch_size]

    def compute_resim_loss_per_sample(self, student_output, teacher_output, margin=0.1, thre_sim=0.5):
        """
        Compute resim loss per sample
        For each sample, compute loss based on its similarity with other samples
        Returns: [batch_size] tensor of per-sample losses
        """
        batch_size = student_output.size(0)
        
        # Normalize outputs for similarity computation
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        
        # Compute all pairwise similarities
        student_similarities = torch.mm(student_norm, student_norm.t())  # [batch_size, batch_size]
        teacher_similarities = torch.mm(teacher_norm, teacher_norm.t())  # [batch_size, batch_size]
        
        # Compute per-sample loss
        per_sample_losses = []
        for i in range(batch_size):
            # For sample i, find valid pairs (j != i and teacher_sim[i,j] > threshold)
            mask = (teacher_similarities[i] > thre_sim) & (torch.arange(batch_size, device=student_output.device) != i)
            
            if mask.sum() > 0:
                # Get similarities for valid pairs
                valid_student_sim = student_similarities[i][mask]
                valid_teacher_sim = teacher_similarities[i][mask]
                
                # Compute loss for valid pairs
                loss_terms = torch.relu(margin + valid_student_sim - valid_teacher_sim)
                sample_loss = loss_terms.mean()
            else:
                sample_loss = torch.tensor(0.0, device=student_output.device)
            
            per_sample_losses.append(sample_loss)
        
        return torch.stack(per_sample_losses)  # [batch_size]

    # Keep original methods for backward compatibility
    def compute_cosine_loss(self, student_output, teacher_output):
        """
        Compute cosine similarity loss: L_cosine = sum_x (1 - s_x . t_x)
        """
        return self.compute_cosine_loss_per_sample(student_output, teacher_output).mean()

    def compute_similarity_loss(self, student_output, teacher_output):
        """
        Compute similarity matrix loss: L_sim = MSE(S_x S_x^T, T_x T_x^T)
        """
        return self.compute_similarity_loss_per_sample(student_output, teacher_output).mean()

    def compute_resim_loss(self, student_output, teacher_output, margin=0.1, thre_sim=0.5):
        """
        Compute resim loss: L_resim = (1/N) sum_{tr_i tr_j > thre_sim} MAX(0, s_i . s_j - t_i . t_j + margin)
        """
        return self.compute_resim_loss_per_sample(student_output, teacher_output, margin, thre_sim).mean()
