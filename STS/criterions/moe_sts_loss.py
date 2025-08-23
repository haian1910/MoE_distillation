import torch
import torch.nn as nn
import torch.distributed as dist

class MoE_STSLoss(nn.Module):
    def __init__(self, args) -> None:
        super(MoE_STSLoss, self).__init__()
        self.loss_type = getattr(args, "sts_loss_type", "mse")  # Default to MSE if not specified
        
        # MoE-specific parameters
        self.moe_loss_weight = getattr(args, "moe_loss_weight", 0.1)  # Weight for MoE auxiliary loss
        self.load_balancing_weight = getattr(args, "load_balancing_weight", 0.01)  # Load balancing loss weight
        
        # Check if BF16 is being used
        self.use_bf16 = getattr(args, "bf16", False)
        if not hasattr(args, "bf16"):
            # Try to detect from DeepSpeed config
            if hasattr(args, "deepspeed_config"):
                try:
                    import json
                    with open(args.deepspeed_config, 'r') as f:
                        ds_config = json.load(f)
                        self.use_bf16 = ds_config.get("bf16", {}).get("enabled", False)
                except:
                    pass
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute loss for STS (Semantic Textual Similarity) tasks with MoE model.
        - Expects model output (batch_size, 1), target (batch_size, 1).
        - batch_denom is typically the batch size.
        """
        self.distiller = distiller
        model = distiller.student_model
        target = output_data["labels"]

        # Convert target to BF16 if needed
        if self.use_bf16 and target.dtype == torch.float32:
            target = target.to(torch.bfloat16)

        # Forward pass through the MoE model - request MoE outputs
        model_output = model(
            input_ids=input_data['input_ids'],
            attention_mask=input_data['attention_mask'],
            token_type_ids=input_data['token_type_ids'] if 'token_type_ids' in input_data else None,
            labels=target  # Pass labels to compute loss inside model if needed
        )
        
        # Extract predictions (similarity scores)
        predictions = model_output.scores if hasattr(model_output, 'scores') else model_output.logits
            
        # Ensure predictions are the right shape
        if predictions.shape[-1] != 1:
            # If the model outputs multiple values, use mean pooling or a linear layer
            if hasattr(self.distiller, "regression_head"):
                predictions = self.distiller.regression_head(predictions)
            else:
                # Create a simple regression head on first use
                self.distiller.regression_head = nn.Linear(predictions.size(-1), 1).to(predictions.device)
                if self.use_bf16:
                    self.distiller.regression_head = self.distiller.regression_head.to(torch.bfloat16)
                self.distiller.regression_head.weight.data.normal_(mean=0.0, std=0.02)
                self.distiller.regression_head.bias.data.zero_()
                predictions = self.distiller.regression_head(predictions)

        # Ensure consistent dtype
        if self.use_bf16:
            if predictions.dtype != torch.bfloat16:
                predictions = predictions.to(torch.bfloat16)
            if target.dtype != torch.bfloat16:
                target = target.to(torch.bfloat16)

        # Compute primary STS loss
        sts_loss = self.compute_sts_loss(predictions, target)
        
        # Compute MoE-specific losses
        total_loss = sts_loss
        moe_losses = {}
        
        if hasattr(model_output, 'gating_weights') and model_output.gating_weights is not None:
            # Load balancing loss - encourage balanced usage of experts
            load_balancing_loss = self.compute_load_balancing_loss(model_output.gating_weights)
            moe_losses['load_balancing'] = load_balancing_loss
            total_loss = total_loss + self.load_balancing_weight * load_balancing_loss
            
            # Expert diversity loss (optional) - encourage different experts to specialize
            if hasattr(model_output, 'expert_outputs') and model_output.expert_outputs is not None:
                diversity_loss = self.compute_expert_diversity_loss(model_output.expert_outputs)
                moe_losses['diversity'] = diversity_loss
                total_loss = total_loss + self.moe_loss_weight * diversity_loss
        
        # Compute correlation for evaluation
        pearson, spearman = self.compute_correlations(predictions, target)
        
        # Prepare logging content
        logging_content = {
            "loss": sts_loss,
            "total_loss": total_loss,
            "pearson": pearson,
            "spearman": spearman,
            "predictions": predictions.detach().mean(),
            "target": target.detach().mean()
        }
        
        # Add MoE-specific logging
        if moe_losses:
            logging_content.update({f"moe_{k}": v for k, v in moe_losses.items()})
            
        # Add expert usage statistics
        if hasattr(model_output, 'gating_weights') and model_output.gating_weights is not None:
            expert_usage = self.compute_expert_usage_stats(model_output.gating_weights)
            logging_content.update(expert_usage)
        
        # Update logging output, return to main distillation
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            logging_content
        )
        
        return total_loss, logging_output

    def compute_sts_loss(self, predictions, target):
        """Compute the primary STS loss"""
        if self.loss_type == "mse":
            return nn.MSELoss()(predictions, target)
        elif self.loss_type == "mae":
            return nn.L1Loss()(predictions, target)
        elif self.loss_type == "huber":
            return nn.SmoothL1Loss()(predictions, target)
        else:
            # Default to MSE
            return nn.MSELoss()(predictions, target)

    def compute_load_balancing_loss(self, gating_weights):
        """
        Compute load balancing loss to encourage balanced usage of experts.
        gating_weights: [batch_size, num_experts]
        """
        # Calculate the fraction of tokens assigned to each expert
        expert_usage = gating_weights.mean(dim=0)  # [num_experts]
        
        # Ideal uniform distribution
        num_experts = expert_usage.size(0)
        uniform_distribution = torch.ones_like(expert_usage) / num_experts
        
        # KL divergence between actual and uniform distribution
        kl_div = torch.nn.functional.kl_div(
            torch.log(expert_usage + 1e-8), 
            uniform_distribution, 
            reduction='batchmean'
        )
        
        return kl_div

    def compute_expert_diversity_loss(self, expert_outputs):
        """
        Compute diversity loss to encourage experts to produce different outputs.
        expert_outputs: List of [batch_size, output_dim] tensors
        """
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)
        
        # Stack expert outputs: [batch_size, num_experts, output_dim]
        stacked_outputs = torch.stack(expert_outputs, dim=1)
        
        # Compute pairwise cosine similarities between experts
        similarities = []
        num_experts = len(expert_outputs)
        
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                expert_i = stacked_outputs[:, i, :]  # [batch_size, output_dim]
                expert_j = stacked_outputs[:, j, :]  # [batch_size, output_dim]
                
                # Cosine similarity
                cos_sim = torch.cosine_similarity(expert_i, expert_j, dim=1)  # [batch_size]
                similarities.append(cos_sim.mean())
        
        # Encourage low similarity (high diversity)
        diversity_loss = torch.mean(torch.stack(similarities))
        
        return diversity_loss

    def compute_expert_usage_stats(self, gating_weights):
        """
        Compute statistics about expert usage for logging.
        gating_weights: [batch_size, num_experts]
        """
        # Average usage per expert
        expert_usage = gating_weights.mean(dim=0)  # [num_experts]
        
        # Entropy of expert usage (higher is better for diversity)
        entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
        
        # Most used expert
        most_used_expert = torch.argmax(expert_usage)
        
        # Usage variance (lower is better for balance)
        usage_variance = torch.var(expert_usage)
        
        stats = {
            'expert_entropy': entropy,
            'most_used_expert': most_used_expert.float(),
            'usage_variance': usage_variance
        }
        
        # Add individual expert usage
        for i in range(expert_usage.size(0)):
            stats[f'expert_{i}_usage'] = expert_usage[i]
        
        return stats

    def compute_correlations(self, predictions, target):
        """
        Compute Pearson and Spearman correlations between predictions and targets.
        Returns naive local estimates (proper correlation will be computed at evaluation).
        """
        # Detach and convert to CPU and float32 for correlation calculation
        pred_flat = predictions.detach().to(torch.float32).view(-1).cpu()
        target_flat = target.detach().to(torch.float32).view(-1).cpu()
        
        # Calculate mean and standard deviation for Pearson correlation
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        
        # Simple Pearson correlation estimate (for logging only)
        try:
            # Calculate Pearson correlation coefficient
            numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
            denominator = torch.sqrt(((pred_flat - pred_mean) ** 2).sum() * ((target_flat - target_mean) ** 2).sum())
            pearson = numerator / denominator if denominator != 0 else torch.tensor(0.0)
            
            # Simple placeholder for Spearman (proper calculation requires rank conversion)
            # In practice, this would be calculated during evaluation, not training
            spearman = pearson  # Placeholder
        except:
            # Handle potential numerical issues
            pearson = torch.tensor(0.0)
            spearman = torch.tensor(0.0)
            
        return pearson, spearman

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and correlations for logging, handling distributed training.
        """
        # Get the device of the current process
        if torch.distributed.is_initialized():
            # Get the device used in the current process
            device_params = list(self.parameters())
            if device_params:
                device = device_params[0].device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        for k, v in content.items():
            if k in ["pearson", "spearman", "expert_entropy", "most_used_expert", "usage_variance"] or k.startswith("expert_") and k.endswith("_usage"):
                # For correlations and expert stats, we don't need to divide by batch_denom
                # but we still average across processes
                if isinstance(v, torch.Tensor):
                    # Handle multi-element tensors by taking mean
                    if v.numel() > 1:
                        record_v = v.mean().clone().to(device).to(torch.float32)
                    else:
                        record_v = v.clone().to(device).to(torch.float32)
                        
                    if torch.distributed.is_initialized():
                        try:
                            dist.all_reduce(record_v, dist.ReduceOp.SUM)
                            record_v = record_v.item() / dist.get_world_size()
                        except RuntimeError:
                            # Fallback if all_reduce fails
                            record_v = record_v.item()
                    else:
                        record_v = record_v.item()
                else:
                    record_v = v
                    if torch.distributed.is_initialized():
                        record_v = record_v / dist.get_world_size()
            elif k in ["predictions", "target"]:
                # Just record mean values for monitoring
                if isinstance(v, torch.Tensor):
                    # Handle multi-element tensors by taking mean
                    if v.numel() > 1:
                        record_v = v.mean().item()
                    else:
                        record_v = v.item()
                else:
                    record_v = v
            else:
                # Normalize loss by batch_denom and average across processes
                if isinstance(v, torch.Tensor):
                    # Handle multi-element tensors by taking mean first
                    if v.numel() > 1:
                        v_scalar = v.mean()
                    else:
                        v_scalar = v
                        
                    # Move tensor to correct device before all_reduce
                    # Always use float32 for these statistics
                    record_v = (v_scalar / batch_denom).to(torch.float32)
                    if torch.distributed.is_initialized():
                        record_v = record_v.to(device)
                        try:
                            dist.all_reduce(record_v, dist.ReduceOp.SUM)
                            record_v = record_v.item() / dist.get_world_size()
                        except RuntimeError:
                            # Fallback if all_reduce fails
                            record_v = (v_scalar / batch_denom).item()
                    else:
                        record_v = record_v.item()
                else:
                    record_v = v / batch_denom
                    if torch.distributed.is_initialized():
                        record_v = record_v / dist.get_world_size()
                    
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
                
        return logging_output
