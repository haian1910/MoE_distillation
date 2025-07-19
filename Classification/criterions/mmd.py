import torch
import torch.nn as nn
from .cross_entropy_loss_moe import CrossEntropyLossMoE
from .various_divergence import VariousDivergence
import torch.nn.functional as F


class MMD(CrossEntropyLossMoE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized

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
        
        # Compute distillation loss
        kd_loss, log = self.compute_mmd_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("mmd_loss:", kd_loss)
        
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
        student_layers_to_process = [9, 10]  # Can modify based on your needs
        
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
                log[f"mmd_loss_layer_{k}"] = mmd_loss.detach().clone()
                
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
