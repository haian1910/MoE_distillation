import torch
import torch.nn as nn
import torch.distributed as dist

class CrossEntropyLossMoE(nn.Module):
    def __init__(self, args) -> None:
        super(CrossEntropyLossMoE, self).__init__()
        self.label_smoothing = args.label_smoothing
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute cross-entropy loss and accuracy for text classification.
        - Expects logits (batch_size, num_classes), target (batch_size,).
        - batch_denom is typically the batch size.
        """
        self.distiller = distiller
        model = distiller.student_model
        target = output_data["labels"]
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            return_moe_outputs=True
        )
        if isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        elif isinstance(outputs, torch.Tensor):
             # If outputs is a tensor, assume it's the logits directly
             # This might be needed if the model doesn't return a dict in some cases
             logits = outputs
        else:
             raise TypeError("Model outputs must be a dictionary with 'logits' or a tensor")



        # Compute loss and accuracy
        loss, nll_loss = self.compute_cross_entropy_loss(logits, target)
        correct = self.compute_accuracy(logits, target)
        
        # Update logging output, return to main distillation
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "nll_loss": nll_loss,
                "correct": correct
            }
        )
        return loss, logging_output

    def compute_cross_entropy_loss(self, logits, target):
        # Tính log softmax trên chiều lớp
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        
        # Tính negative log likelihood loss
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        
        if self.label_smoothing > 0:
            # Tính mất mát mịn (smooth loss)
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss, nll_loss

    def compute_accuracy(self, logits, target):
        # Lấy chỉ số lớp có xác suất cao nhất
        pred = logits.argmax(dim=-1)
        
        # Tính số lượng mẫu dự đoán đúng
        correct = pred.eq(target).sum().float()
        accu = correct / target.size(0)
        return accu

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and accuracy for logging, handling distributed training.
        content = {
                "loss": loss,
                "nll_loss": nll_loss,
                "correct": correct
            }
        """
        
        for k, v in content.items():
            if k == "correct":
                # Sum the correct counts across processes
                record_v = v.clone()
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item()
            else:
                # Normalize loss by batch_denom and average across processes
                record_v = v / batch_denom
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                
                # Check if tensor is scalar before calling .item()
                if record_v.numel() == 1:
                    # Scalar tensor - can use .item()
                    record_v = record_v.item() / dist.get_world_size()
                else:
                    # Non-scalar tensor - handle differently
                    record_v = record_v / dist.get_world_size()
                    # Convert to python list or keep as tensor depending on your logging needs
                    if record_v.numel() <= 10:  # Small tensors can be converted to list
                        record_v = record_v.tolist()
                    else:
                        # For large tensors, you might want to compute some summary statistics
                        record_v = {
                            'mean': record_v.mean().item(),
                            'std': record_v.std().item(),
                            'shape': list(record_v.shape)
                        }
                        
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        return logging_output
