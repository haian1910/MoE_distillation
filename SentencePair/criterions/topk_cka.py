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
        self.ortho_reg_weight = getattr(args, 'ortho_reg_weight', 1)  # Weight for orthogonal regularization
        
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
        
        # Define layer mapping (you can adjust this based on your needs)
        # Map last few student layers to last few teacher layers
        num_layers_to_align = [9, 10]  # Adjust based on your model architecture
        
        # Align the last few layers
        for i in range(len(num_layers_to_align)):
            student_layer_idx = num_layers_to_align[i]
            teacher_layer_idx = 3* student_layer_idx  # Example mapping, adjust as needed
            
            if student_layer_idx < 0 or teacher_layer_idx < 0:
                continue
                
            # Get hidden states for this layer
            student_hidden = student_hidden_states[student_layer_idx]  # [batch_size, seq_len_s, hidden_dim_s]
            teacher_hidden = teacher_hidden_states[teacher_layer_idx]  # [batch_size, seq_len_t, hidden_dim_t]
            
            # Get attention masks to handle padding
            student_mask = input_data["attention_mask"]  # [batch_size, seq_len_s]
            teacher_mask = input_data["teacher_attention_mask"]  # [batch_size, seq_len_t]
            
            # Compute layer-wise CKA loss
            layer_cka_loss = self.compute_single_layer_topk_cka(
                student_hidden, 
                teacher_hidden,
                student_mask,
                teacher_mask
            )
            
            total_cka_loss += layer_cka_loss
            num_aligned_layers += 1
            
            # Log individual layer losses for debugging
            log[f"cka_loss_layer_{student_layer_idx}"] = layer_cka_loss.detach().clone()
        
        # Average across aligned layers
        if num_aligned_layers > 0:
            total_cka_loss = total_cka_loss / num_aligned_layers
        
        return total_cka_loss, log
    
    def compute_single_layer_topk_cka(
        self, 
        student_hidden, 
        teacher_hidden,
        student_mask,
        teacher_mask
    ):
        """
        Compute Top-k Token Transfer + CKA loss for a single layer pair
        
        Args:
            student_hidden: [batch_size, seq_len_s, hidden_dim_s]
            teacher_hidden: [batch_size, seq_len_t, hidden_dim_t]
            student_mask: [batch_size, seq_len_s]
            teacher_mask: [batch_size, seq_len_t]
        """
        batch_size, seq_len_s, hidden_dim_s = student_hidden.shape
        _, seq_len_t, hidden_dim_t = teacher_hidden.shape
        
        # Initialize projector if needed (project student to teacher dimension space)
        if hidden_dim_s != hidden_dim_t:
            if self.projector is None or self.projector.projector.in_features != hidden_dim_s:
                self.projector = OrthogonalProjection(hidden_dim_s, hidden_dim_t).to(student_hidden.device)
            
            # Project student hidden states to teacher dimension
            student_hidden_proj = self.projector(student_hidden)  # [batch_size, seq_len_s, hidden_dim_t]
        else:
            student_hidden_proj = student_hidden
        
        # Step 1: Create soft representation
        # Compute cosine similarities between all student-teacher token pairs
        # Normalize hidden states
        student_norm = F.normalize(student_hidden_proj, p=2, dim=-1)  # [batch_size, seq_len_s, hidden_dim_t]
        teacher_norm = F.normalize(teacher_hidden, p=2, dim=-1)  # [batch_size, seq_len_t, hidden_dim_t]
        
        # Compute similarity matrix
        similarity_matrix = torch.bmm(student_norm, teacher_norm.transpose(1, 2))  # [batch_size, seq_len_s, seq_len_t]
        
        # Apply masks to ignore padding tokens
        student_mask_expanded = student_mask.unsqueeze(-1).float()  # [batch_size, seq_len_s, 1]
        teacher_mask_expanded = teacher_mask.unsqueeze(1).float()  # [batch_size, 1, seq_len_t]
        mask_matrix = student_mask_expanded * teacher_mask_expanded  # [batch_size, seq_len_s, seq_len_t]
        
        # Set similarity to -inf for padded positions
        similarity_matrix = similarity_matrix.masked_fill(mask_matrix == 0, -1e9)
        
        # Step 2: Select top-k teacher tokens for each student token
        topk_values, topk_indices = torch.topk(similarity_matrix, k=min(self.topk, seq_len_t), dim=-1)  # [batch_size, seq_len_s, k]
        
        # Step 3: Compute normalized weights using temperature
        topk_weights = F.softmax(topk_values / self.temperature, dim=-1)  # [batch_size, seq_len_s, k]
        
        # Step 4: Aggregate teacher representations
        aggregated_teacher = torch.zeros_like(student_hidden_proj)  # [batch_size, seq_len_s, hidden_dim_t]
        
        for b in range(batch_size):
            for s in range(seq_len_s):
                if student_mask[b, s] == 0:  # Skip padding tokens
                    continue
                    
                # Get top-k teacher indices for this student token
                teacher_indices = topk_indices[b, s]  # [k]
                weights = topk_weights[b, s]  # [k]
                
                # Aggregate teacher representations
                for k_idx in range(len(teacher_indices)):
                    t_idx = teacher_indices[k_idx]
                    if t_idx < seq_len_t and teacher_mask[b, t_idx] == 1:  # Valid teacher token
                        aggregated_teacher[b, s] += weights[k_idx] * teacher_hidden[b, t_idx]
        
        # Step 5: Compute CKA loss between student and aggregated teacher representations
        # Reshape for CKA computation
        student_flat = student_hidden.view(batch_size, -1)  # [batch_size, seq_len_s * hidden_dim_s]
        teacher_flat = aggregated_teacher.view(batch_size, -1)  # [batch_size, seq_len_s * hidden_dim_t]
        
        # Apply CKA loss
        cka_loss = self.cka_loss(student_flat, teacher_flat)
        
        return cka_loss
