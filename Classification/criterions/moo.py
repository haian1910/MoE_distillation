import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss import CrossEntropyLoss

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

class MOO(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate

        # Parameters for pairwise relation loss
        self.rank_margin = getattr(args, 'rank_margin', 0.1)
        
        self.projection = LinearProjection(768, 4096)
        
        # Dynamic top-k selection parameters
        self.temperature = getattr(args, 'temperature', 1.0)
        self.probability_mass_threshold = getattr(args, 'probability_mass_threshold', 0.9)
        self.k_min = getattr(args, 'k_min', 1)
        self.k_max = getattr(args, 'k_max', 3)
        self.s_min = getattr(args, 's_min', 0.3)
        
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
        
        # Student forward pass with hidden states
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True,  # Important: get hidden states for both losses
            return_dict=True
        )
        
        # Extract logits safely
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            raise TypeError("Model outputs must contain 'logits'")
        
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
                output_hidden_states=True,
                return_dict=True
            )
        
        # Compute distillation loss
        moo_loss, log = self.compute_moo_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("moo_loss:", moo_loss.item())

        topk_cka_loss, log = self.compute_topk_cka_loss(
            outputs, teacher_outputs, output_data, input_data, distiller, log
        )
        print("topk_cka_loss:", topk_cka_loss.item())

        # Final loss combination with proper weighting
        total_distillation_loss = 0.8 * moo_loss + 0.2 * topk_cka_loss
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * total_distillation_loss
        
        log["loss"] = loss.detach().clone()
        log["ce_loss"] = self.compute_cross_entropy_loss(logits, output_data["labels"])[0].detach().clone()
        log["total_distillation_loss"] = total_distillation_loss.detach().clone()

        # Compute accuracy
        accuracy = self.compute_accuracy(logits, output_data["labels"])
        log["accuracy"] = accuracy

        # Update logging output
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output

    def compute_moo_loss(self, outputs, teacher_outputs, output_data, distiller, log):
        """
        Compute the MOO (Multi Objective Optimization) loss.
        Apply 3 loss functions between the student's and teacher's output embeddings:
        1. Cosine Loss  
        2. InfoNCE Loss
        3. Pairwise Relation Loss
        """
        device = next(distiller.student_model.parameters()).device

        # Extract student embeddings
        student_emb = self._extract_student_embeddings(outputs)
        
        # Extract teacher embeddings  
        teacher_emb = self._extract_teacher_embeddings(teacher_outputs)
        
        # Project student embeddings to teacher space
        projected_student_emb = self.projection(student_emb)

        # Compute individual losses (per sample)
        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(
            projected_student_emb, teacher_emb
        )
        print("cosine_loss_per_sample:", cosine_loss_per_sample.mean().item())

        infoNCE_loss_per_sample = self.compute_infoNCE_loss_per_sample(
            projected_student_emb, teacher_emb
        )
        print("infoNCE_loss_per_sample:", infoNCE_loss_per_sample.mean().item())

        pairwise_relation_loss_per_sample = self.compute_pairwise_relation_loss_per_sample(
            student_emb, teacher_emb
        )
        print("pairwise_relation_loss_per_sample:", pairwise_relation_loss_per_sample.mean().item())


        # Combine losses with equal weighting
        moo_loss = (0.2*cosine_loss_per_sample + 0.6*infoNCE_loss_per_sample + 
                   0.2*pairwise_relation_loss_per_sample)
        
        # Take mean across batch
        moo_loss = moo_loss.mean()
        
        # Log individual components
        log["moo_loss"] = moo_loss.detach().clone()
        log["cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()
        log["infoNCE_loss"] = infoNCE_loss_per_sample.mean().detach().clone()
        log["pairwise_relation_loss"] = pairwise_relation_loss_per_sample.mean().detach().clone()

        return moo_loss, log

    def _extract_student_embeddings(self, outputs):
        """Extract student embeddings from model outputs"""
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            student_hidden = outputs.hidden_states[-1]  # Last layer
        elif isinstance(outputs, dict) and 'hidden_states' in outputs:
            student_hidden = outputs['hidden_states'][-1]
        else:
            raise ValueError("No hidden states found in student model outputs")
        
        # Extract CLS token embedding (position 0 for BERT-like models)
        if student_hidden.dim() == 3:  # [batch_size, seq_len, hidden_size]
            student_emb = student_hidden[:, 0, :]  # CLS token
        elif student_hidden.dim() == 2:  # [batch_size, hidden_size]  
            student_emb = student_hidden
        else:
            raise ValueError(f"Unexpected student hidden state shape: {student_hidden.shape}")
            
        return student_emb

    def _extract_teacher_embeddings(self, teacher_outputs):
        """Extract teacher embeddings from model outputs"""
        if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
            teacher_hidden = teacher_outputs.hidden_states[-1]  # Last layer
        elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
            teacher_hidden = teacher_outputs['hidden_states'][-1]
        else:
            raise ValueError("No hidden states found in teacher model outputs")
        
        # For LLM2Vec, use mean pooling across sequence length
        if teacher_hidden.dim() == 3:  # [batch_size, seq_len, hidden_size]
            teacher_emb = teacher_hidden.mean(dim=1)  # Mean pooling
        elif teacher_hidden.dim() == 2:  # [batch_size, hidden_size]
            teacher_emb = teacher_hidden
        else:
            raise ValueError(f"Unexpected teacher hidden state shape: {teacher_hidden.shape}")
            
        return teacher_emb

    def compute_cosine_loss_per_sample(self, student_output, teacher_output):
        """
        Compute cosine similarity loss per sample: L_cosine = 1 - s_x · t_x
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
        
        # Handle single sample case
        if batch_size == 1:
            return torch.zeros(1, device=student_output.device)
        
        # Normalize embeddings
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        
        # Compute similarity matrix: student_i vs all teachers
        similarity_matrix = torch.mm(student_norm, teacher_norm.t()) / temperature
        
        # Get positive similarities (diagonal)
        positive_similarities = torch.diagonal(similarity_matrix)
        
        # Compute InfoNCE loss per sample
        log_sum_exp = torch.logsumexp(similarity_matrix, dim=1)
        infonce_loss_per_sample = log_sum_exp - positive_similarities
        
        return infonce_loss_per_sample

    def compute_pairwise_relation_loss_per_sample(self, student_output, teacher_output):
        """
        Compute margin-based pairwise relation loss per sample.
        L_rank = sum_i sum_j max(0, |sim(z_i^s, z_j^s) - sim(z_i^t, z_j^t)| - δ)
        
        Args:
            student_output: [batch_size, hidden_size] student embeddings
            teacher_output: [batch_size, hidden_size] teacher embeddings

        Returns:
            [batch_size] tensor of per-sample losses
        """
        batch_size = student_output.size(0)
        
        # Handle single sample case
        if batch_size == 1:
            return torch.zeros(1, device=student_output.device)
        
        # Normalize embeddings for similarity computation
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        
        # Compute all pairwise similarities
        student_similarities = torch.mm(student_norm, student_norm.t())
        teacher_similarities = torch.mm(teacher_norm, teacher_norm.t())
        
        # Compute difference matrix
        diff_matrix = torch.abs(student_similarities - teacher_similarities)
        
        # Apply margin and ReLU, then compute per-sample loss
        margin_loss_matrix = torch.relu(diff_matrix - self.rank_margin)
        
        # Create mask to exclude diagonal (self-similarities)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=student_output.device)
        
        # Compute per-sample loss by averaging over valid pairs
        per_sample_losses = []
        for i in range(batch_size):
            valid_losses = margin_loss_matrix[i][mask[i]]
            sample_loss = valid_losses.mean() if valid_losses.numel() > 0 else torch.tensor(0.0, device=student_output.device)
            per_sample_losses.append(sample_loss)
        
        return torch.stack(per_sample_losses)

    def compute_topk_cka_loss(
        self, outputs, teacher_outputs, output_data, input_data, distiller, log
    ):
        """
        Compute Top-k Token Transfer + CKA loss as described in the paper
        """
        total_cka_loss = 0.0
        num_aligned_layers = 0
        
        # Extract hidden states
        student_hidden_states = self._get_hidden_states(outputs, "student")
        teacher_hidden_states = self._get_hidden_states(teacher_outputs, "teacher")
        
        # Get number of layers
        student_layer_num = len(student_hidden_states)
        teacher_layer_num = len(teacher_hidden_states)
        
        # Define layer mapping - align last few layers
        num_layers_to_align = min(2, student_layer_num)
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
            student_h_reshaped = student_h.view(-1, student_dim)
            projected_student_h = self.projection(student_h_reshaped)
            projected_student_h = projected_student_h.view(batch_size, student_seq_len, -1)
            
            # Handle different sequence lengths
            if student_seq_len != teacher_seq_len:
                aligned_teacher_h = self.create_soft_representation(
                    projected_student_h, teacher_h, student_seq_len, teacher_seq_len
                )
            else:
                aligned_teacher_h = teacher_h
                
            # Compute CKA loss
            cka_loss = self.cka_loss(
                projected_student_h.view(-1, projected_student_h.size(-1)), 
                aligned_teacher_h.view(-1, aligned_teacher_h.size(-1))
            )
            total_cka_loss += cka_loss
            num_aligned_layers += 1
            
            # Log individual layer losses
            log[f"cka_loss_layer_{s_idx}_{t_idx}"] = cka_loss.detach().clone()
        
        # Average CKA loss across aligned layers
        if num_aligned_layers > 0:
            avg_cka_loss = total_cka_loss / num_aligned_layers
        else:
            avg_cka_loss = torch.tensor(0.0, device=student_hidden_states[0].device)
            
        log["avg_cka_loss"] = avg_cka_loss.detach().clone()
        log["num_aligned_layers"] = torch.tensor(num_aligned_layers, device=student_hidden_states[0].device)
        
        return avg_cka_loss, log

    def _get_hidden_states(self, outputs, model_type):
        """Extract hidden states from model outputs"""
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            return outputs.hidden_states
        elif isinstance(outputs, dict) and 'hidden_states' in outputs:
            return outputs['hidden_states']
        else:
            raise ValueError(f"Cannot extract {model_type} hidden states. Make sure to call model with output_hidden_states=True")
    
    def create_soft_representation(self, student_h, teacher_h, student_seq_len, teacher_seq_len):
        """
        Create soft representation for teacher tokens aligned to student tokens using dynamic top-k transfer
        
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
        
        # For each student token position
        for p in range(student_seq_len):
            student_token = student_h[:, p, :]  # [batch_size, hidden_dim]
            
            # Normalize for cosine similarity computation
            student_token_norm = F.normalize(student_token, p=2, dim=-1)
            teacher_h_norm = F.normalize(teacher_h, p=2, dim=-1)
            
            # Compute cosine similarities
            similarities = torch.bmm(
                student_token_norm.unsqueeze(1),  # [batch_size, 1, hidden_dim]
                teacher_h_norm.transpose(1, 2)    # [batch_size, hidden_dim, teacher_seq_len]
            ).squeeze(1)  # [batch_size, teacher_seq_len]
            
            # Process each sample in the batch
            for b in range(batch_size):
                batch_similarities = similarities[b]  # [teacher_seq_len]
                
                # Convert to probabilities via softmax
                alpha_tilde = F.softmax(batch_similarities / self.temperature, dim=0)
                
                # Dynamic top-k selection
                sorted_probs, sorted_indices = torch.sort(alpha_tilde, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                
                # Find number of tokens to select based on probability mass
                valid_positions = (cumsum_probs >= self.probability_mass_threshold).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    num_selected = min(valid_positions[0].item() + 1, self.k_max)
                else:
                    num_selected = min(teacher_seq_len, self.k_max)
                
                num_selected = max(num_selected, self.k_min)
                num_selected = min(num_selected, teacher_seq_len)
                
                # Apply similarity threshold constraint
                selected_indices = sorted_indices[:num_selected]
                selected_probs = sorted_probs[:num_selected]
                
                similarity_mask = batch_similarities[selected_indices] >= self.s_min
                if similarity_mask.sum() > 0:
                    final_indices = selected_indices[similarity_mask]
                    final_probs = selected_probs[similarity_mask]
                else:
                    # Keep top-1 if no tokens meet similarity threshold
                    final_indices = selected_indices[:1]
                    final_probs = selected_probs[:1]
                
                # Renormalize and aggregate
                if len(final_indices) > 0:
                    alpha_normalized = final_probs / final_probs.sum()
                    selected_teacher_tokens = teacher_h[b, final_indices, :]
                    aggregated_token = torch.sum(
                        alpha_normalized.unsqueeze(-1) * selected_teacher_tokens, dim=0
                    )
                    aligned_teacher_h[b, p, :] = aggregated_token
                else:
                    # Fallback: use most similar token
                    best_idx = torch.argmax(batch_similarities)
                    aligned_teacher_h[b, p, :] = teacher_h[b, best_idx, :]
        
        return aligned_teacher_h

    # Legacy methods for backward compatibility
    def compute_cosine_loss(self, student_output, teacher_output):
        """Compute average cosine similarity loss across batch"""
        return self.compute_cosine_loss_per_sample(student_output, teacher_output).mean()

    def compute_infoNCE_loss(self, student_output, teacher_output):
        """Compute average InfoNCE loss across batch"""
        return self.compute_infoNCE_loss_per_sample(student_output, teacher_output).mean()

    def compute_pairwise_relation_loss(self, student_output, teacher_output):
        """Compute average pairwise relation loss across batch"""
        return self.compute_pairwise_relation_loss_per_sample(student_output, teacher_output).mean()
