import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss_moe import CrossEntropyLossMoE

# orthogonal projection for loss in expert1 and expert 2
class OrthogonalProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=4096):
        super(OrthogonalProjection, self).__init__()
        # Create a regular linear layer first
        self.projector = nn.Linear(in_dim, out_dim, bias=False)
        # Initialize with orthogonal weights (in float32)
        with torch.no_grad():
            nn.init.orthogonal_(self.projector.weight)

    def forward(self, x):
        # Handle dtype conversion for orthogonal constraint if needed
        original_dtype = x.dtype
        
        # If input is bfloat16, we can work directly with it
        # The linear layer will handle the computation
        return self.projector(x)
        
    def orthogonal_regularization_loss(self):
        """
        Optional: Add this to your total loss to maintain orthogonality during training
        L_ortho = ||W^T W - I||_F^2
        """
        W = self.projector.weight  # [out_dim, in_dim]
        if W.shape[0] >= W.shape[1]:  # out_dim >= in_dim
            # W^T W should be identity
            WtW = torch.mm(W.t(), W)  # [in_dim, in_dim]
            I = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
        else:  # out_dim < in_dim  
            # W W^T should be identity
            WWt = torch.mm(W, W.t())  # [out_dim, out_dim]
            I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            WtW = WWt
        
        ortho_loss = torch.norm(WtW - I, p='fro') ** 2
        return ortho_loss

class MOE(CrossEntropyLossMoE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate

        # Parameters for pairwise relation loss
        self.rank_margin = getattr(args, 'rank_margin', 0.1)
        
        # Parameters for expert diversity loss
        self.diversity_weight = getattr(args, 'diversity_weight', 1)

        # Create projections for experts 1 and 2 (expert 3 doesn't need projection)
        self.expert1_projection = OrthogonalProjection(768, 2048)
        self.expert2_projection = OrthogonalProjection(768, 2048)
        self.ortho_weight = getattr(args, 'ortho_weight', 1)  # Add this line
        
        # Flag to track if projections have been moved to device
        self._projections_initialized = False

    def _ensure_projections_on_device(self, device, dtype):
        """Ensure projection layers are on the correct device and dtype"""
        if not self._projections_initialized:
            self.expert1_projection = self.expert1_projection.to(device=device, dtype=dtype)
            self.expert2_projection = self.expert2_projection.to(device=device, dtype=dtype)
            self._projections_initialized = True

    
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
        
        # Get device and dtype from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Ensure projections are on the correct device and dtype
        self._ensure_projections_on_device(device, dtype)
        
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
        
        # Compute orthogonal regularization loss
        ortho_loss_expert1 = self.expert1_projection.orthogonal_regularization_loss()
        ortho_loss_expert2 = self.expert2_projection.orthogonal_regularization_loss()
        ortho_loss = ortho_loss_expert1 + ortho_loss_expert2
        log["ortho_loss"] = ortho_loss.detach().clone()
        print("ortho_loss:", ortho_loss.detach().clone())
        
        # Combine all losses
        total_kd_loss = kd_loss + self.diversity_weight * diversity_loss
    
        # Add orthogonal loss to the final loss
        # You may want to add a weight parameter for the orthogonal loss
        ortho_weight = getattr(self, 'ortho_weight', 1)  # Default weight of 1
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * total_kd_loss + ortho_weight * ortho_loss
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
        - Expert 2: InfoNCE Loss
        - Expert 3: Pairwise Relation loss
        The final MoE loss is weighted by the gating network outputs.
        """
        # Get device for tensor creation
        device = next(distiller.student_model.parameters()).device
        
        # Get MoE outputs from student model
        expert_outputs = outputs['expert_outputs']  # List of 3 expert outputs
        gating_weights = outputs['gating_weights']  # [batch_size, num_experts]
        
        # Extract teacher hidden states - Handle different model architectures
        if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
            # For LLM2Vec model loaded with AutoModelForSequenceClassification
            teacher_hidden = teacher_outputs.hidden_states[-1]  # Last layer hidden states
        elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
            # If teacher_outputs is a dict with hidden_states key
            teacher_hidden = teacher_outputs['hidden_states'][-1]
        else:
            raise ValueError("Cannot extract teacher hidden states")
        
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
        expert1_output = expert_outputs[0]  # [batch_size, student_hidden_size]
        
        # Because of size mismatch between student and teacher, we project student to teacher size using an orthogonal projection
        projected_expert1 = self.expert1_projection(expert1_output)  # [batch_size, teacher_hidden_size]

        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(projected_expert1, projected_teacher) 
        expert_losses.append(cosine_loss_per_sample)
        log["expert1_cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()
        print("expert1_cosine_loss:", cosine_loss_per_sample.mean().detach().clone())

        # Expert 2: InfoNCE Loss - compute per sample 
        expert2_output = expert_outputs[1]  # [batch_size, student_hidden_size]
        # Project expert2 output to teacher size
        projected_expert2 = self.expert2_projection(expert2_output)  # [batch_size, teacher_hidden_size]
        infoNCE_loss_per_sample = self.compute_infoNCE_loss_per_sample(projected_expert2, projected_teacher)

        expert_losses.append(infoNCE_loss_per_sample)
        log["expert2_infonce_loss"] = infoNCE_loss_per_sample.mean().detach().clone()
        print("expert2_infonce_loss:", infoNCE_loss_per_sample.mean().detach().clone())

        # Expert 3: pairwise relation Loss - compute per sample (no projection needed)
        expert3_output = expert_outputs[2]  # [batch_size, student_hidden_size]
        pairwise_relation_loss_per_sample = self.compute_pairwise_relation_loss_per_sample(expert3_output, projected_teacher)
        expert_losses.append(pairwise_relation_loss_per_sample)
        log["expert3_pairwise_relation_loss"] = pairwise_relation_loss_per_sample.mean().detach().clone()
        print("expert3_pairwise_relation_loss:", pairwise_relation_loss_per_sample.mean().detach().clone())

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

    def compute_pairwise_relation_loss_per_sample(self, student_output, teacher_output):
        """
        Compute margin-based pairwise relation loss per sample.
        L_rank = sum_i sum_j max(0, sim(z_i^s, z_j^s) - sim(z_i^t, z_j^t) + δ)
        
        Args:
            student_output: [batch_size, student_hidden_size] student embeddings
            teacher_output: [batch_size, teacher_hidden_size] teacher embeddings

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
        
        # Compute per-sample pairwise relation loss
        per_sample_losses = []
        
        for i in range(batch_size):
            sample_loss = 0.0
            count = 0
            
            for j in range(batch_size):
                if i != j:  # Skip self-similarity
                    # Compute pairwise relation loss for pair (i, j)
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

    def compute_infoNCE_loss_per_sample(self, student_output, teacher_output, temperature=0.1):
        """
        Compute InfoNCE loss per sample with proper contrastive learning.
        For each student sample i, positive is teacher_i, negatives are all other teachers.
        
        Args:
            student_output: [batch_size, hidden_size] - projected student embeddings
            teacher_output: [batch_size, hidden_size] - teacher embeddings  
            temperature: temperature parameter for InfoNCE
            
        Returns:
            [batch_size] tensor of per-sample InfoNCE losses
        """
        batch_size = student_output.size(0)
        
        # Normalize embeddings
        student_norm = F.normalize(student_output, p=2, dim=-1)  # [batch_size, hidden_size]
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)  # [batch_size, hidden_size]
        
        # Compute similarity matrix: student_i vs all teachers
        # similarity_matrix[i,j] = sim(student_i, teacher_j)
        similarity_matrix = torch.mm(student_norm, teacher_norm.t()) / temperature  # [batch_size, batch_size]
        
        # For each sample i, positive is similarity_matrix[i,i], negatives are similarity_matrix[i,j] for j!=i
        # But InfoNCE denominator includes the positive, so we use all similarities
        
        # Get positive similarities (diagonal)
        positive_similarities = torch.diagonal(similarity_matrix)  # [batch_size]
        
        # Compute InfoNCE loss per sample
        # For sample i: -log(exp(pos_i) / sum_j(exp(sim_ij)))
        log_sum_exp = torch.logsumexp(similarity_matrix, dim=1)  # [batch_size]
        infonce_loss_per_sample = log_sum_exp - positive_similarities  # [batch_size]
        
        return infonce_loss_per_sample

    # Keep original methods for backward compatibility
    def compute_cosine_loss(self, student_output, teacher_output):
        """
        Compute cosine similarity loss: L_cosine = sum_x (1 - s_x . t_x)
        """
        return self.compute_cosine_loss_per_sample(student_output, teacher_output).mean()

    def compute_infoNCE_loss(self, student_output, teacher_output):
        """
        Compute InfoNCE loss: L_infoNCE using InfoNCE
        """
        return self.compute_infoNCE_loss_per_sample(student_output, teacher_output).mean()

    def compute_pairwise_relation_loss(self, student_output, teacher_output):
        """
        Compute pairwise relation loss: L_rank using margin-based pairwise relation loss
        """
        return self.compute_pairwise_relation_loss_per_sample(student_output, teacher_output).mean()
