import torch
import torch.nn as nn
import torch.nn.functional as F
from .sts_loss import STSLoss

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
        SH = SH.view(-1, dS).to(TH.device, torch.float64)
        TH = TH.view(-1, dT).to(TH.device, torch.float64)
        
        # Center the representations
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        
        # Compute CKA similarity with numerical stability
        SH_gram = SH.t().matmul(SH)
        TH_gram = TH.t().matmul(TH)
        SH_TH_cross = SH.t().matmul(TH)
        
        num = torch.norm(SH_TH_cross, 'fro')
        den1 = torch.norm(SH_gram, 'fro') + self.eps
        den2 = torch.norm(TH_gram, 'fro') + self.eps
        
        # Return CKA loss (1 - similarity for loss)
        return 1 - num / torch.sqrt(den1 * den2)

class MOO(STSLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate

        # Parameters for pairwise relation loss
        self.rank_margin = getattr(args, 'rank_margin', 0.1)
        
        # Parameters for expert diversity loss
        self.diversity_weight = getattr(args, 'diversity_weight', 1.0)

        # Create projections for experts 1 and 2 (for STS dimension matching)
        # Student BERT: 768 dim -> Teacher LLM2Vec: 4096 dim
        self.projection = LinearProjection(768, 4096)
        
        # Dynamic top-k selection parameters
        self.temperature = getattr(args, 'temperature', 1.0)
        self.probability_mass_threshold = getattr(args, 'probability_mass_threshold', 0.8)
        self.k_min = getattr(args, 'k_min', 1)
        self.k_max = getattr(args, 'k_max', 3)
        self.s_min = getattr(args, 's_min', 0.3)
        
        # Initialize CKA loss
        self.cka_loss = CKALoss()
        
        # STS-specific parameters
        self.sts_loss_weight = getattr(args, 'sts_loss_weight', 0.3)  # Weight for CKA loss
        self.mse_loss_fn = nn.MSELoss()  # For STS regression
        
        # Flag to track if projections have been moved to device
        self._projections_initialized = False

    def _ensure_projections_on_device(self, device, dtype):
        """Ensure projection layers are on the correct device and dtype"""
        if not self._projections_initialized:
            self.projection = self.projection.to(device=device, dtype=dtype)
            self._projections_initialized = True

    def _ensure_shape_consistency(self, predictions, labels):
        """
        Ensure predictions and labels have consistent shapes for STS regression.
        STS expects scalar values for each sample.
        
        Args:
            predictions: Model predictions (can be [batch_size, 1] or [batch_size])
            labels: Target labels (should be [batch_size])
            
        Returns:
            predictions: [batch_size] shaped tensor
            labels: [batch_size] shaped tensor
        """
        # Handle predictions shape
        if predictions.dim() > 1:
            if predictions.size(-1) == 1:
                # If shape is [batch_size, 1], squeeze to [batch_size]
                predictions = predictions.squeeze(-1)
            else:
                # If shape is [batch_size, num_classes] where num_classes > 1
                # For STS, we need a regression head to convert to scalar
                if not hasattr(self, 'regression_head'):
                    # Create regression head if it doesn't exist
                    self.regression_head = nn.Linear(
                        predictions.size(-1), 1, device=predictions.device, dtype=predictions.dtype
                    )
                    # Initialize regression head
                    with torch.no_grad():
                        self.regression_head.weight.normal_(mean=0.0, std=0.02)
                        self.regression_head.bias.zero_()
                
                predictions = self.regression_head(predictions).squeeze(-1)
        
        # Handle labels shape
        if labels.dim() > 1:
            if labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            else:
                # If labels have multiple dimensions, take the first column or mean
                labels = labels[:, 0] if labels.size(-1) > 1 else labels.squeeze()
        
        # Ensure both tensors are 1D with same length
        assert predictions.dim() == 1, f"Predictions should be 1D, got shape {predictions.shape}"
        assert labels.dim() == 1, f"Labels should be 1D, got shape {labels.shape}"
        assert predictions.size(0) == labels.size(0), f"Batch size mismatch: predictions {predictions.size(0)} vs labels {labels.size(0)}"
        
        return predictions, labels
    
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

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Ensure projections are on the correct device and dtype
        self._ensure_projections_on_device(device, dtype)
        
        # Student forward pass
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            labels=output_data["labels"]
        )
        
        # Extract predictions for STS task - handle both scores and logits
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

        # Use float32 for numerical stability in STS regression
        target_dtype = torch.float32
        device = predictions.device

        # Convert predictions to target dtype
        predictions = predictions.to(dtype=target_dtype)
        
        # Convert labels to the same dtype as predictions
        labels = output_data["labels"].to(dtype=target_dtype, device=device)
        
        # Ensure shape consistency before computing loss
        predictions, labels = self._ensure_shape_consistency(predictions, labels)
        
        # Compute MSE loss for STS regression task
        loss_sts = F.mse_loss(predictions, labels)
        log["loss_sts"] = loss_sts.detach().clone()
        
        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True,
                return_dict=True
            )
        
        # Compute distillation losses
        moo_loss, log = self.compute_moo_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("moo_loss:", moo_loss.item())
        
        # Compute top-k CKA loss
        topk_cka_loss, log = self.compute_topk_cka_loss(
            outputs, teacher_outputs, output_data, input_data, distiller, log
        )
        print("topk_cka_loss:", topk_cka_loss.item())

        # Final loss combination - balance STS task loss with distillation losses
        loss = (1.0 - self.kd_rate) * loss_sts + self.kd_rate * (0.8 * moo_loss + 0.2 * topk_cka_loss)
        log["loss"] = loss.detach().clone()

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss, logging_output

    def compute_moo_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        """
        Compute the MOO (Multi Objective Optimization) loss.
        Apply 3 loss functions between the student's and teacher's output embeddings:
        1. Cosine Loss  
        2. InfoNCE Loss
        3. Pairwise Relation Loss
        """
        # Get device for tensor creation
        device = next(distiller.student_model.parameters()).device

        # Extract student embeddings - use the final sentence representation
        student_emb = self.extract_sentence_embedding(outputs, is_teacher=False)
        
        # Extract teacher embeddings - use the final sentence representation
        teacher_emb = self.extract_sentence_embedding(teacher_outputs, is_teacher=True)
        
        # Project student embeddings to match teacher dimensionality
        projected_student_emb = self.projection(student_emb)  # [batch_size, teacher_hidden_size]

        # Compute three types of losses per sample
        cosine_loss_per_sample = self.compute_cosine_loss_per_sample(projected_student_emb, teacher_emb) 
        infoNCE_loss_per_sample = self.compute_infoNCE_loss_per_sample(projected_student_emb, teacher_emb)
        pairwise_relation_loss_per_sample = self.compute_pairwise_relation_loss_per_sample(student_emb, teacher_emb)

        # Combine losses with weights
        weighted_losses = (0.2 * cosine_loss_per_sample + 
                          0.6 * infoNCE_loss_per_sample + 
                          0.2 * pairwise_relation_loss_per_sample)

        # Take mean across batch to get final scalar loss
        moo_loss = weighted_losses.mean()
        
        # Log individual loss components
        log["loss_moo"] = moo_loss.detach().clone()
        log["cosine_loss"] = cosine_loss_per_sample.mean().detach().clone()
        log["infoNCE_loss"] = infoNCE_loss_per_sample.mean().detach().clone()
        log["pairwise_relation_loss"] = pairwise_relation_loss_per_sample.mean().detach().clone()

        return moo_loss, log

    def extract_sentence_embedding(self, model_outputs, is_teacher=False):
        """
        Extract sentence-level embeddings from model outputs.
        For STS tasks, we typically use CLS token or mean pooling.
        
        Args:
            model_outputs: Model outputs containing hidden states
            is_teacher: Whether this is teacher model (affects extraction method)
            
        Returns:
            torch.Tensor: [batch_size, hidden_dim] sentence embeddings
        """
        # Extract hidden states
        if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states is not None:
            hidden_states = model_outputs.hidden_states[-1]  # Last layer
        elif isinstance(model_outputs, dict) and 'hidden_states' in model_outputs:
            hidden_states = model_outputs['hidden_states'][-1]
        elif hasattr(model_outputs, 'last_hidden_state'):
            hidden_states = model_outputs.last_hidden_state
        else:
            raise ValueError("Cannot extract hidden states from model outputs")
        
        # Extract sentence embedding
        if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_size]
            # For STS tasks, use CLS token (first token) or mean pooling
            if is_teacher:  # LLM2Vec might benefit from mean pooling
                sentence_emb = hidden_states.mean(dim=1)
            else:  # BERT typically uses CLS token
                sentence_emb = hidden_states[:, 0]  # CLS token
        elif hidden_states.dim() == 2:  # [batch_size, hidden_size]
            sentence_emb = hidden_states
        else:
            raise ValueError(f"Unexpected dimension for hidden states: {hidden_states.shape}")
            
        return sentence_emb

    def compute_pairwise_relation_loss_per_sample(self, student_output, teacher_output):
        """
        Compute margin-based pairwise relation loss per sample.
        L_rank = sum_i sum_j max(0, |sim(z_i^s, z_j^s) - sim(z_i^t, z_j^t)| - δ)
        
        Args:
            student_output: [batch_size, student_hidden_size] student embeddings
            teacher_output: [batch_size, teacher_hidden_size] teacher embeddings

        Returns:
            [batch_size] tensor of per-sample losses
        """
        batch_size = student_output.size(0)
        
        # Handle edge case: batch size < 2
        if batch_size < 2:
            return torch.zeros(batch_size, device=student_output.device, dtype=student_output.dtype)
        
        # Normalize embeddings for similarity computation
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        
        # Compute all pairwise similarities
        student_similarities = torch.mm(student_norm, student_norm.t())  # [batch_size, batch_size]
        teacher_similarities = torch.mm(teacher_norm, teacher_norm.t())  # [batch_size, batch_size]
        
        # Compute difference matrix
        similarity_diff = torch.abs(student_similarities - teacher_similarities)
        
        # Create mask to exclude diagonal (self-similarities)
        mask = ~torch.eye(batch_size, device=student_output.device, dtype=torch.bool)
        
        # Apply margin and mask
        margin_loss = torch.relu(similarity_diff - self.rank_margin) * mask.float()
        
        # Sum over all pairs for each sample and normalize
        per_sample_losses = margin_loss.sum(dim=1) / (batch_size - 1)
        
        return per_sample_losses  # [batch_size]

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
        
        # Handle edge case: batch size = 1
        if batch_size == 1:
            return torch.zeros(1, device=student_output.device, dtype=student_output.dtype)
        
        # Normalize embeddings
        student_norm = F.normalize(student_output, p=2, dim=-1)  # [batch_size, hidden_size]
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)  # [batch_size, hidden_size]
        
        # Compute similarity matrix: student_i vs all teachers
        # similarity_matrix[i,j] = sim(student_i, teacher_j)
        similarity_matrix = torch.mm(student_norm, teacher_norm.t()) / temperature  # [batch_size, batch_size]
        
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
        num_layers_to_align = min(3, student_layer_num, teacher_layer_num)  # Align up to 3 layers
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
            teacher_dim = teacher_h.size(-1)
            
            # Reshape for projection
            student_h_reshaped = student_h.view(-1, student_dim)  # [batch*seq_len, student_dim]
            projected_student_h = self.projection(student_h_reshaped)  # [batch*seq_len, teacher_dim]
            projected_student_h = projected_student_h.view(batch_size, student_seq_len, teacher_dim)
            
            # Handle different sequence lengths due to different tokenizers
            if student_seq_len != teacher_seq_len:
                # Create soft representation using dynamic top-k token transfer
                try:
                    aligned_teacher_h = self.create_soft_representation(
                        projected_student_h, teacher_h, student_seq_len, teacher_seq_len
                    )
                except Exception as e:
                    print(f"Warning: Soft representation failed for layer {s_idx}->{t_idx}: {e}")
                    # Fallback: simple interpolation or truncation/padding
                    if teacher_seq_len > student_seq_len:
                        # Truncate teacher sequence
                        aligned_teacher_h = teacher_h[:, :student_seq_len, :]
                    else:
                        # Pad teacher sequence
                        padding = torch.zeros(batch_size, student_seq_len - teacher_seq_len, teacher_dim,
                                            device=teacher_h.device, dtype=teacher_h.dtype)
                        aligned_teacher_h = torch.cat([teacher_h, padding], dim=1)
            else:
                aligned_teacher_h = teacher_h
                
            # Compute CKA loss between projected student and aligned teacher representations
            try:
                cka_loss = self.cka_loss(
                    projected_student_h.view(-1, projected_student_h.size(-1)),
                    aligned_teacher_h.view(-1, aligned_teacher_h.size(-1))
                )
                total_cka_loss += cka_loss
                num_aligned_layers += 1
                
                # Log individual layer losses for debugging
                log[f"cka_loss_layer_{s_idx}_{t_idx}"] = cka_loss.detach().clone()
                
            except Exception as e:
                print(f"Warning: CKA loss computation failed for layer {s_idx}->{t_idx}: {e}")
                continue
        
        # Average CKA loss across aligned layers
        if num_aligned_layers > 0:
            avg_cka_loss = total_cka_loss / num_aligned_layers
        else:
            print("Warning: No layers successfully aligned for CKA loss")
            avg_cka_loss = torch.tensor(0.0, device=device)
            
        log["avg_cka_loss"] = avg_cka_loss.detach().clone()
        log["num_aligned_layers"] = torch.tensor(num_aligned_layers, device=device)
        
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
            student_token = student_h[:, p, :]  # [batch_size, hidden_dim]
            
            # Normalize for cosine similarity computation
            student_token_norm = F.normalize(student_token, p=2, dim=-1)  # [batch_size, hidden_dim]
            teacher_h_norm = F.normalize(teacher_h, p=2, dim=-1)  # [batch_size, teacher_seq_len, hidden_dim]
            
            # Compute cosine similarities
            similarities = torch.bmm(
                student_token_norm.unsqueeze(1),  # [batch_size, 1, hidden_dim]
                teacher_h_norm.transpose(1, 2)    # [batch_size, hidden_dim, teacher_seq_len]
            ).squeeze(1)  # [batch_size, teacher_seq_len]
            
            # Process each sample in the batch
            for b in range(batch_size):
                batch_similarities = similarities[b]  # [teacher_seq_len]
                
                # Convert similarities to probability distribution via softmax
                alpha_tilde = F.softmax(batch_similarities / self.temperature, dim=0)
                
                # Dynamic top-k selection based on probability mass
                sorted_probs, sorted_indices = torch.sort(alpha_tilde, descending=True)
                
                # Find smallest set such that cumulative probability mass >= threshold
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                valid_positions = (cumsum_probs >= self.probability_mass_threshold).nonzero(as_tuple=True)[0]
                
                if len(valid_positions) > 0:
                    num_selected = min(valid_positions[0].item() + 1, self.k_max)
                else:
                    num_selected = min(teacher_seq_len, self.k_max)
                
                # Ensure we select at least k_min tokens
                num_selected = max(num_selected, self.k_min)
                num_selected = min(num_selected, teacher_seq_len)
                
                # Get selected indices
                selected_indices = sorted_indices[:num_selected]
                selected_probs = sorted_probs[:num_selected]
                
                # Apply similarity threshold constraint
                similarity_mask = batch_similarities[selected_indices] >= self.s_min
                if similarity_mask.sum() > 0:
                    final_indices = selected_indices[similarity_mask]
                    final_probs = selected_probs[similarity_mask]
                else:
                    # Fallback: keep top-1 token
                    final_indices = selected_indices[:1]
                    final_probs = selected_probs[:1]
                
                # Renormalize probabilities and aggregate
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
