import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss_moe import CrossEntropyLossMoE

# Simple linear projection for loss in expert1 and expert 2
class LinearProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=4096):
        super(LinearProjection, self).__init__()
        self.projector = nn.Linear(in_dim, out_dim, bias=False)
        # Initialize with Xavier uniform
        with torch.no_grad():
            nn.init.xavier_uniform_(self.projector.weight)

    def forward(self, x):
        return self.projector(x)
    
class CKALoss(nn.Module):
    """CKA Loss for measuring similarity between hidden representations"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
        # Center the representations
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        
        # Compute CKA similarity
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        # Return CKA loss
        return 1 - num / torch.sqrt(den1 * den2)

class CKA_MOE(CrossEntropyLossMoE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate

        # Parameters for pairwise relation loss
        self.rank_margin = getattr(args, 'rank_margin', 0.1)
        
        # Parameters for expert diversity loss
        self.diversity_weight = getattr(args, 'diversity_weight', 1)
        
        # Parameter for L_min regularization
        self.min_gate_threshold = getattr(args, 'min_gate_threshold', 0.25)

        # Create projections for experts 1 and 2 (expert 3 doesn't need projection)
        self.projection = LinearProjection(768, 4096)
        
        # Dynamic top-k selection parameters
        self.temperature = getattr(args, 'temperature', 1.0)  # Temperature for softmax
        self.probability_mass_threshold = getattr(args, 'probability_mass_threshold', 0.95)  # t in the algorithm
        self.k_min = getattr(args, 'k_min', 1)  # Minimum number of tokens to select
        self.k_max = getattr(args, 'k_max', 3)  # Maximum number of tokens to select
        self.s_min = getattr(args, 's_min', 0.3)  # Minimum similarity threshold
        
        # Initialize CKA loss
        self.cka_loss = CKALoss()
        
        # Flag to track if projections have been moved to device
        self._projections_initialized = False

    def _ensure_projections_on_device(self, device, dtype):
        """Ensure projection layers are on the correct device and dtype"""
        if not self._projections_initialized:
            self.projection = self.projection.to(device=device, dtype=dtype)
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
        moe_loss, log = self.compute_moe_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("moe_loss:", moe_loss)

        # Compute expert diversity loss
        diversity_loss = self.compute_expert_diversity_loss(outputs['expert_outputs'])
        log["diversity_loss"] = diversity_loss.detach().clone()
        print("diversity_loss:", diversity_loss.detach().clone())
        
        # Combine all losses
        total_moe_loss = moe_loss + self.diversity_weight * diversity_loss

        topk_cka_loss, log = self.compute_topk_cka_loss(
            outputs, teacher_outputs, output_data, input_data, distiller, log
        )
        print("topk_cka_loss:", topk_cka_loss)

        # Final loss combination
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * (0.7*total_moe_loss + 0.3*topk_cka_loss)
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
        
        # Because of size mismatch between student and teacher, we project student to teacher size using linear projection
        projected_expert1 = self.projection(expert1_output)  # [batch_size, teacher_hidden_size]

        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(projected_expert1, projected_teacher) 
        expert_losses.append(cosine_loss_per_sample)
        log["expert1_cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()
        print("expert1_cosine_loss:", cosine_loss_per_sample.mean().detach().clone())

        # Expert 2: InfoNCE Loss - compute per sample 
        expert2_output = expert_outputs[1]  # [batch_size, student_hidden_size]
        # Project expert2 output to teacher size
        projected_expert2 = self.projection(expert2_output)  # [batch_size, teacher_hidden_size]
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
        
        # Compute L_min regularization
        # L_min = (1/N) * sum_{t=1}^N sum_{i=1}^S [max(0, l - pi_t,i)]^2
        # where l is min_gate_threshold, pi_t,i is gating weight for expert i in sample t
        l_min_loss = torch.mean(torch.sum(
            torch.pow(torch.relu(0.25 - gating_weights), 2),
            dim=1
        ))
        print("L_min_loss:", l_min_loss.detach().clone())
        # Add L_min regularization to moe_loss
        moe_loss = moe_loss + 20*l_min_loss
        
        log["moe_loss"] = moe_loss.detach().clone()
        log["l_min_loss"] = l_min_loss.detach().clone()
        log["gating_weights_mean"] = gating_weights.mean(dim=0).detach().clone()  # Average gating weights as tensor
        
        return moe_loss, log

    def compute_pairwise_relation_loss_per_sample(self, student_output, teacher_output):
        """
        Compute margin-based pairwise relation loss per sample.
        L_rank = sum_i sum_j max(0, abs(sim(z_i^s, z_j^s) - sim(z_i^t, z_j^t)) - δ)
        
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
                        torch.abs(teacher_similarities[i, j] - student_similarities[i, j]) - self.rank_margin
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
    

    def compute_topk_cka_loss(
        self, outputs, teacher_outputs, output_data, input_data, distiller, log
    ):
        """
        Compute Top-k Token Transfer + CKA loss as described in the paper
        """
        total_cka_loss = 0.0
        num_aligned_layers = 0
        
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
        
        # Get number of layers
        student_layer_num = len(student_hidden_states)
        teacher_layer_num = len(teacher_hidden_states)
        
        # Define layer mapping - align last few layers
        # For BERT (12 layers) to LLM2Vec-Mistral (32 layers), we align last few layers
        num_layers_to_align = min(2, student_layer_num)  # Adjust based on your needs
        student_layer_indices = list(range(student_layer_num - num_layers_to_align, student_layer_num))
        teacher_layer_indices = list(range(teacher_layer_num - num_layers_to_align, teacher_layer_num))
        
        # Process each layer alignment
        for s_idx, t_idx in zip(student_layer_indices, teacher_layer_indices):
            # Get hidden states for current layers
            student_h = student_hidden_states[s_idx]  # [batch_size, seq_len, student_dim]
            teacher_h = teacher_hidden_states[t_idx]   # [batch_size, seq_len, teacher_dim]
            
            # Apply linear projection to student hidden states
            batch_size, student_seq_len, student_dim = student_h.shape
            teacher_seq_len = teacher_h.size(1)
            
            # Reshape for projection
            student_h_reshaped = student_h.view(-1, student_dim)  # [batch*seq_len, student_dim]
            projected_student_h = self.projection(student_h_reshaped)  # [batch*seq_len, teacher_dim]
            projected_student_h = projected_student_h.view(batch_size, student_seq_len, -1)  # [batch, seq_len, teacher_dim]
            
            # Handle different sequence lengths due to different tokenizers
            if student_seq_len != teacher_seq_len:
                # Create soft representation using dynamic top-k token transfer
                aligned_teacher_h = self.create_soft_representation(
                    projected_student_h, teacher_h, student_seq_len, teacher_seq_len
                )
            else:
                aligned_teacher_h = teacher_h
                
            # Compute CKA loss between projected student and (aligned) teacher representations
            cka_loss = self.cka_loss(projected_student_h.view(-1, projected_student_h.size(-1)), 
                                   aligned_teacher_h.view(-1, aligned_teacher_h.size(-1)))
            total_cka_loss += cka_loss
            num_aligned_layers += 1
            
            # Log individual layer losses for debugging
            log[f"cka_loss_layer_{s_idx}_{t_idx}"] = cka_loss.detach().clone()
        
        # Average CKA loss across aligned layers
        if num_aligned_layers > 0:
            avg_cka_loss = total_cka_loss / num_aligned_layers
        else:
            avg_cka_loss = torch.tensor(0.0, device=student_hidden_states[0].device)
            
        log["avg_cka_loss"] = avg_cka_loss.detach().clone()
        log["num_aligned_layers"] = torch.tensor(num_aligned_layers, device=student_hidden_states[0].device)
        
        return avg_cka_loss, log
    
    def create_soft_representation(self, student_h, teacher_h, student_seq_len, teacher_seq_len):
        """
        Create soft representation for teacher tokens aligned to student tokens using dynamic top-k transfer
        following the algorithm in the paper.
        
        Args:
            student_h: [batch_size, student_seq_len, hidden_dim] - projected student hidden states
            teacher_h: [batch_size, teacher_seq_len, hidden_dim] - teacher hidden states
            student_seq_len: length of student sequence
            teacher_seq_len: length of teacher sequence
            
        Returns:
            aligned_teacher_h: [batch_size, student_seq_len, hidden_dim] - aligned teacher representations
        """
        batch_size, _, hidden_dim = student_h.shape
        device = student_h.device
        
        # Initialize aligned teacher representations
        aligned_teacher_h = torch.zeros(batch_size, student_seq_len, hidden_dim, 
                                      device=device, dtype=student_h.dtype)
        
        # For each student token position p ∈ {1, ..., n_i}
        for p in range(student_seq_len):
            student_token = student_h[:, p, :]  # [batch_size, hidden_dim] - h_{i,p}^(s)
            
            # Normalize for cosine similarity computation
            student_token_norm = F.normalize(student_token, p=2, dim=-1)  # [batch_size, hidden_dim]
            teacher_h_norm = F.normalize(teacher_h, p=2, dim=-1)  # [batch_size, teacher_seq_len, hidden_dim]
            
            # Compute cosine similarities s_{p,q} = sim(h_{i,p}^(s)W, h_{i,q}^(t)) for q = 1, ..., m_i
            similarities = torch.bmm(
                student_token_norm.unsqueeze(1),  # [batch_size, 1, hidden_dim]
                teacher_h_norm.transpose(1, 2)    # [batch_size, hidden_dim, teacher_seq_len]
            ).squeeze(1)  # [batch_size, teacher_seq_len]
            
            # Process each sample in the batch
            for b in range(batch_size):
                batch_similarities = similarities[b]  # [teacher_seq_len]
                
                # Convert similarities to full distribution via softmax: α̃_{p,q}
                alpha_tilde = F.softmax(batch_similarities / self.temperature, dim=0)  # [teacher_seq_len]
                
                # Dynamic top-k selection based on probability mass
                # Sort α̃_{p,q} in descending order
                sorted_probs, sorted_indices = torch.sort(alpha_tilde, descending=True)
                
                # Find smallest set S_{i,p} such that cumulative probability mass ≥ t
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                
                # Find the first position where cumsum >= threshold
                valid_positions = (cumsum_probs >= self.probability_mass_threshold).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    # +1 because we want to include the position where threshold is reached
                    num_selected = min(valid_positions[0].item() + 1, self.k_max)
                else:
                    # If threshold is never reached, select all tokens up to k_max
                    num_selected = min(teacher_seq_len, self.k_max)
                
                # Ensure we select at least k_min tokens
                num_selected = max(num_selected, self.k_min)
                num_selected = min(num_selected, teacher_seq_len)  # Can't select more than available
                
                # Get the selected indices
                selected_indices = sorted_indices[:num_selected]
                selected_probs = sorted_probs[:num_selected]
                
                # Apply similarity threshold constraint: s_{p,q} ≥ s_min for any q ∈ S_{i,p}
                similarity_mask = batch_similarities[selected_indices] >= self.s_min
                if similarity_mask.sum() > 0:
                    # Keep only tokens that meet similarity threshold
                    final_indices = selected_indices[similarity_mask]
                    final_probs = selected_probs[similarity_mask]
                else:
                    # If no tokens meet threshold, keep the top-1 token to avoid empty selection
                    final_indices = selected_indices[:1]
                    final_probs = selected_probs[:1]
                
                # Renormalize probabilities over the selected set: α_{p,q}
                if len(final_indices) > 0:
                    alpha_normalized = final_probs / final_probs.sum()
                    
                    # Aggregate teacher vectors: h̃_{i,p}^(t) = Σ_{q∈S_{i,p}} α_{p,q} h_{i,q}^(t)
                    selected_teacher_tokens = teacher_h[b, final_indices, :]  # [num_final, hidden_dim]
                    aggregated_token = torch.sum(
                        alpha_normalized.unsqueeze(-1) * selected_teacher_tokens, dim=0
                    )  # [hidden_dim]
                    
                    aligned_teacher_h[b, p, :] = aggregated_token
                else:
                    # Fallback: if no valid selection, use the most similar token
                    best_idx = torch.argmax(batch_similarities)
                    aligned_teacher_h[b, p, :] = teacher_h[b, best_idx, :]
        
        return aligned_teacher_h  # [batch_size, student_seq_len, hidden_dim]
