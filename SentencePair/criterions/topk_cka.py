import torch
import torch.nn as nn
from .cross_entropy_loss_moe import CrossEntropyLossMoE
from .various_divergence import VariousDivergence
import torch.nn.functional as F

class OrthogonalProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=4096):
        super(OrthogonalProjection, self).__init__()
        # Create a regular linear layer first
        self.projector = nn.Linear(in_dim, out_dim, bias=False)
        # Initialize with orthogonal weights (in float32)
        with torch.no_grad():
            nn.init.orthogonal_(self.projector.weight)

    def forward(self, x):
        # Handle dtype conversion - ensure projector weights match input dtype
        if x.dtype != self.projector.weight.dtype:
            # Convert projector weights to match input dtype
            self.projector.weight.data = self.projector.weight.data.to(x.dtype)
        
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

class TOPK_CKA(CrossEntropyLossMoE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized
        self.topk = getattr(args, 'topk', 3)  # Default top-k value
        self.temperature = getattr(args, 'temperature', 1.0)  # Temperature for softmax
        self.ortho_reg_weight = getattr(args, 'ortho_reg_weight', 0.1)  # Weight for orthogonal regularization
        
        # Initialize CKA loss
        self.cka_loss = CKALoss()
        
        # Initialize orthogonal projector (will be created dynamically based on dimensions)
        self.projector = None

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
        ce_loss = self.compute_cross_entropy_loss(
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
        kd_loss, log = self.compute_topk_cka_loss(
            outputs, teacher_outputs, output_data, input_data, distiller, log
        )
        print("topk_cka_loss:", kd_loss)
        
        # Add orthogonal regularization loss if projector exists
        ortho_loss = 0
        if self.projector is not None:
            ortho_loss = self.projector.orthogonal_regularization_loss()
            log["ortho_loss"] = ortho_loss.detach().clone()
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * kd_loss + self.ortho_reg_weight * ortho_loss
        
        log["ce_loss"] = ce_loss.detach().clone()
        log["kd_loss"] = kd_loss.detach().clone()
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
        num_layers_to_align = min(3, student_layer_num)  # Adjust based on your needs
        student_layer_indices = list(range(student_layer_num - num_layers_to_align, student_layer_num))
        teacher_layer_indices = list(range(teacher_layer_num - num_layers_to_align, teacher_layer_num))
        
        # Create orthogonal projector if not exists
        if self.projector is None:
            student_dim = student_hidden_states[0].size(-1)
            teacher_dim = teacher_hidden_states[0].size(-1)
            self.projector = OrthogonalProjection(student_dim, teacher_dim)
            self.projector = self.projector.to(student_hidden_states[0].device)
            print(f"Created orthogonal projector: {student_dim} -> {teacher_dim}")
        
        # Process each layer alignment
        for s_idx, t_idx in zip(student_layer_indices, teacher_layer_indices):
            # Get hidden states for current layers
            student_h = student_hidden_states[s_idx]  # [batch_size, seq_len, student_dim]
            teacher_h = teacher_hidden_states[t_idx]   # [batch_size, seq_len, teacher_dim]
            
            # Apply orthogonal projection to student hidden states
            batch_size, student_seq_len, student_dim = student_h.shape
            teacher_seq_len = teacher_h.size(1)
            
            # Reshape for projection
            student_h_reshaped = student_h.view(-1, student_dim)  # [batch*seq_len, student_dim]
            projected_student_h = self.projector(student_h_reshaped)  # [batch*seq_len, teacher_dim]
            projected_student_h = projected_student_h.view(batch_size, student_seq_len, -1)  # [batch, seq_len, teacher_dim]
            
            # Handle different sequence lengths due to different tokenizers
            if student_seq_len != teacher_seq_len:
                # Create soft representation using top-k token transfer
                aligned_teacher_h = self.create_soft_representation(
                    projected_student_h, teacher_h, student_seq_len, teacher_seq_len
                )
            else:
                aligned_teacher_h = teacher_h
                
            # Compute CKA loss between projected student and (aligned) teacher representations
            cka_loss = self.cka_loss(student_h_reshaped, aligned_teacher_h)
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
        Create soft representation for teacher tokens aligned to student tokens using top-k transfer
        
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
        
        # For each student token position, find top-k similar teacher tokens
        for p in range(student_seq_len):  # For each student token position
            student_token = student_h[:, p, :]  # [batch_size, hidden_dim]
            
            # Compute cosine similarities with all teacher tokens
            # student_token: [batch_size, hidden_dim]
            # teacher_h: [batch_size, teacher_seq_len, hidden_dim]
            
            # Normalize for cosine similarity
            student_token_norm = F.normalize(student_token, p=2, dim=-1)  # [batch_size, hidden_dim]
            teacher_h_norm = F.normalize(teacher_h, p=2, dim=-1)  # [batch_size, teacher_seq_len, hidden_dim]
            
            # Compute cosine similarities
            similarities = torch.bmm(
                student_token_norm.unsqueeze(1),  # [batch_size, 1, hidden_dim]
                teacher_h_norm.transpose(1, 2)    # [batch_size, hidden_dim, teacher_seq_len]
            ).squeeze(1)  # [batch_size, teacher_seq_len]
            
            # Get top-k teacher token indices for each sample in batch
            topk_values, topk_indices = torch.topk(similarities, k=min(self.topk, teacher_seq_len), dim=-1)
            # topk_values: [batch_size, k], topk_indices: [batch_size, k]
            
            # Compute normalized weights using temperature
            alpha_weights = F.softmax(topk_values / self.temperature, dim=-1)  # [batch_size, k]
            
            # Aggregate teacher vectors for each sample in batch
            for b in range(batch_size):
                selected_teacher_tokens = teacher_h[b, topk_indices[b], :]  # [k, hidden_dim]
                weighted_teacher_token = torch.sum(
                    alpha_weights[b].unsqueeze(-1) * selected_teacher_tokens, dim=0
                )  # [hidden_dim]
                aligned_teacher_h[b, p, :] = weighted_teacher_token
        
        return aligned_teacher_h  # [batch_size, student_seq_len, hidden_dim]
