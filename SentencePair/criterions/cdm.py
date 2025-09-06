import logging
import torch
import torch.nn.functional as F
import numpy as np
import transformers
import editdistance
import copy
import math
from typing import Dict, List
from .cross_entropy_loss import CrossEntropyLoss


def calculate_weight(logits):
    """Calculate weight factor based on logits entropy/uncertainty"""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    # Normalize entropy to [0, 1] range and use as weight
    max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
    weight = entropy / max_entropy
    return weight.cpu().numpy()


class CDM(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        # Simplified initialization for classification task
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        # Classification specific parameters
        self.topk = getattr(args, 'topk', 100)
        self.simi_threshold = getattr(args, 'simi_threadshold', 0.1)
        self.kd_alpha = getattr(args, 'kd_alpha', 0.5)
        self.padding_id = -100
        
        # Token mapping for different tokenizers
        self.TOKENIZER_TO_SPECIAL_TOKEN = {
            transformers.models.llama.tokenization_llama.LlamaTokenizer: "<s>",
            transformers.models.bert.tokenization_bert.BertTokenizer: "[CLS]",
            transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer: "<|endoftext|>",
        }
    
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

        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers

        # Get student outputs
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        
        # Compute standard cross-entropy loss for classification
        log = {}
        
        # Compute cross-entropy loss with ground-truth labels
        loss_ce = self.compute_cross_entropy_loss(
            outputs.logits, output_data["labels"]
        )[0]
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
            
        # Compute CDM knowledge distillation loss
        kd_loss, log = self.compute_CDM_loss(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        print("CDM_loss:", kd_loss)
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * kd_loss
        
        # Calculate accuracy 
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"]
        )
        log["accuracy"] = accuracy
        
        # Update logging output
        if logging_output is None:
            logging_output = {}
        logging_output.update(log)
        
        return loss, logging_output

    def compute_CDM_loss(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        """Compute CDM loss adapted for classification tasks"""
        try:
            # For classification, we work with the final logits directly
            student_logits = outputs.logits  # Shape: [batch_size, num_classes]
            teacher_logits = teacher_outputs.logits  # Shape: [batch_size, num_classes]
            
            # Initialize tokenizers
            stu_tokenizer = distiller.student_tokenizer
            tea_tokenizer = distiller.teacher_tokenizers
            
            # Setup token mappings if available
            if hasattr(distiller, 'tea2stu_id_mapping'):
                self.tea2stu_id_mapping = distiller.tea2stu_id_mapping
                self.stu2tea_id_mapping = distiller.stu2tea_id_mapping
                self.em_tea2stu_id_mapping = distiller.em_tea2stu_id_mapping
                self.em_stu2tea_id_mapping = distiller.em_stu2tea_id_mapping
            else:
                # Create dummy mappings for classification
                vocab_size = min(stu_tokenizer.vocab_size, tea_tokenizer.vocab_size)
                device = student_logits.device
                self.tea2stu_id_mapping = torch.arange(vocab_size).to(device)
                self.stu2tea_id_mapping = torch.arange(vocab_size).to(device)
                self.em_tea2stu_id_mapping = torch.arange(vocab_size).to(device)
                self.em_stu2tea_id_mapping = torch.arange(vocab_size).to(device)
            
            batch_size = student_logits.shape[0]
            aligned_tea_logits = []
            aligned_stu_logits = []
            total_unmask_rate = 0.0
            
            # Process each sample in the batch
            for i in range(batch_size):
                # For classification, we can work directly with class logits
                stu_per_sample_logits = student_logits[i:i+1, :]  # Keep batch dimension
                tea_per_sample_logits = teacher_logits[i:i+1, :]
                
                # Get input tokens for alignment (if needed for token-level analysis)
                stu_input_ids = input_data["input_ids"][i]
                tea_input_ids = input_data["teacher_input_ids"][i] if "teacher_input_ids" in input_data else stu_input_ids
                
                try:
                    aligned_tea_sample_logits, aligned_stu_sample_logits, unmask_rate = self.transform_classification_logits(
                        stu_tokenizer,
                        tea_tokenizer,
                        stu_input_ids,
                        stu_per_sample_logits,
                        tea_input_ids,
                        tea_per_sample_logits,
                    )
                    
                    aligned_stu_logits.append(aligned_stu_sample_logits)
                    aligned_tea_logits.append(aligned_tea_sample_logits)
                    total_unmask_rate += unmask_rate
                    
                except Exception as e:
                    print(f"Error in sample {i}: {e}")
                    # Fallback to direct KL divergence
                    aligned_stu_logits.append(stu_per_sample_logits)
                    aligned_tea_logits.append(tea_per_sample_logits)
                    total_unmask_rate += 1.0
            
            if not aligned_stu_logits:
                return torch.tensor(0.0).to(student_logits.device), log
                
            aligned_tea_logits = torch.cat(aligned_tea_logits, dim=0)
            aligned_stu_logits = torch.cat(aligned_stu_logits, dim=0)
            avg_unmask_rate = total_unmask_rate / batch_size
            
            # Compute KL divergence loss
            kd_loss = self.dist_func(
                aligned_stu_logits, 
                aligned_tea_logits
            )
            
            # Update log
            log["cdm_kd_loss"] = kd_loss.item()
            log["unmask_rate"] = avg_unmask_rate
            
            return kd_loss, log
            
        except Exception as e:
            print(f"CDM loss computation failed: {e}")
            # Fallback to simple KL divergence
            kd_loss = self.dist_func(student_logits, teacher_logits)
            log["cdm_kd_loss"] = kd_loss.item()
            log["unmask_rate"] = 1.0
            return kd_loss, log

    def transform_classification_logits(
        self,
        base_model_tokenizer,
        blending_model_tokenizer, 
        base_model_input_ids,
        base_model_logits,
        blending_model_input_ids,
        blending_model_logits,
    ):
        """Transform logits for classification task using CDM approach"""
        
        # For classification, we work with the vocabulary alignment
        # Convert input_ids to tokens for alignment analysis
        base_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        base_tokens = [base_model_tokenizer.convert_tokens_to_string([tok]) for tok in base_tokens if tok is not None]
        
        blending_tokens = blending_model_tokenizer.convert_ids_to_tokens(blending_model_input_ids)
        blending_tokens = [blending_model_tokenizer.convert_tokens_to_string([tok]) for tok in blending_tokens if tok is not None]
        
        # Get special tokens for each tokenizer
        base_special_token = self.get_special_token(base_model_tokenizer)
        blending_special_token = self.get_special_token(blending_model_tokenizer)
        
        # Calculate importance weights based on logits
        base_weights = calculate_weight(base_model_logits)
        blending_weights = calculate_weight(blending_model_logits)
        
        # Perform DTW alignment on tokens if we have sufficient tokens
        if len(base_tokens) > 1 and len(blending_tokens) > 1:
            try:
                _, _, blending_to_base, base_to_blending, _ = self.dtw(
                    blending_tokens, base_tokens, 
                    blending_weights, base_weights, 
                    norm_func=self.token_distance_func
                )
            except:
                # Fallback: direct alignment
                blending_to_base = [[i] for i in range(min(len(blending_tokens), len(base_tokens)))]
                base_to_blending = [[i] for i in range(min(len(base_tokens), len(blending_tokens)))]
        else:
            # Direct mapping for short sequences
            blending_to_base = [[0]]
            base_to_blending = [[0]]
        
        # Get top-k predictions for vocabulary alignment
        topK = min(self.topk, base_model_logits.size(-1))
        
        base_topk_ids = torch.topk(base_model_logits, topK).indices
        blending_topk_ids = torch.topk(blending_model_logits, topK).indices
        
        # Convert to tokens for similarity matching
        base_topk_tokens = []
        for ids in base_topk_ids:
            base_topk_tokens.append([base_model_tokenizer.decode(id.item()) for id in ids])
            
        blending_topk_tokens = []
        for ids in blending_topk_ids:
            blending_topk_tokens.append([blending_model_tokenizer.decode(id.item()) for id in ids])
        
        # Apply vocabulary alignment
        aligned_base_logits, aligned_blending_logits, unmask_rate = self.align_vocabulary_logits(
            base_model_logits,
            blending_model_logits,
            base_topk_ids,
            blending_topk_ids,
            base_topk_tokens,
            blending_topk_tokens
        )
        
        return aligned_base_logits, aligned_blending_logits, unmask_rate

    def align_vocabulary_logits(self, base_logits, blending_logits, base_topk_ids, blending_topk_ids, base_topk_tokens, blending_topk_tokens):
        """Align vocabulary between different tokenizers"""
        
        # Apply token mapping if available
        tea2stu_mapper = self.tea2stu_id_mapping
        stu2tea_mapper = self.stu2tea_id_mapping
        
        aligned_base_logits_list = []
        aligned_blending_logits_list = []
        total_mask_rate = 0.0
        
        # Forward mapping: teacher -> student
        stu_converted_topk_ids, tea_converted_topk_ids = self.get_dynamic_mapper(
            blending_topk_ids, base_topk_ids, blending_topk_tokens, base_topk_tokens,
            blending2base_mapper=tea2stu_mapper.clone(),
            em_mapper=self.em_tea2stu_id_mapping
        )
        
        stu_model_logits = base_logits.gather(-1, stu_converted_topk_ids)
        tea_model_logits = blending_logits.gather(-1, tea_converted_topk_ids)
        
        # Apply masking for unmapped tokens
        stu_logit_mask = stu_converted_topk_ids.eq(0)
        stu_model_logits.masked_fill_(stu_logit_mask, -10000.0)
        tea_logit_mask = tea_converted_topk_ids.eq(0)
        tea_model_logits.masked_fill_(tea_logit_mask, -10000.0)
        
        mask_rate = stu_logit_mask.sum().item() / (stu_logit_mask.size(0) * stu_logit_mask.size(1))
        total_mask_rate += mask_rate
        
        aligned_base_logits_list.append(tea_model_logits)
        aligned_blending_logits_list.append(stu_model_logits)
        
        # Reverse mapping: student -> teacher  
        tea_converted_topk_ids, stu_converted_topk_ids = self.get_dynamic_mapper(
            base_topk_ids, blending_topk_ids, base_topk_tokens, blending_topk_tokens,
            blending2base_mapper=stu2tea_mapper.clone(),
            em_mapper=self.em_stu2tea_id_mapping
        )
        
        stu_model_logits = base_logits.gather(-1, stu_converted_topk_ids)
        tea_model_logits = blending_logits.gather(-1, tea_converted_topk_ids)
        
        stu_logit_mask = stu_converted_topk_ids.eq(0)
        stu_model_logits.masked_fill_(stu_logit_mask, -10000.0)
        tea_logit_mask = tea_converted_topk_ids.eq(0)
        tea_model_logits.masked_fill_(tea_logit_mask, -10000.0)
        
        mask_rate = stu_logit_mask.sum().item() / (stu_logit_mask.size(0) * stu_logit_mask.size(1))
        total_mask_rate += mask_rate
        
        aligned_base_logits_list.append(stu_model_logits)
        aligned_blending_logits_list.append(tea_model_logits)
        
        # Combine both directions
        aligned_base_logits = torch.cat(aligned_base_logits_list, dim=-1)
        aligned_blending_logits = torch.cat(aligned_blending_logits_list, dim=-1)
        avg_mask_rate = total_mask_rate / 2
        unmask_rate = 1 - avg_mask_rate
        
        return aligned_base_logits, aligned_blending_logits, unmask_rate

    def get_dynamic_mapper(self, blending_topk_ids, base_topk_ids, blending_topk_tokens, base_topk_tokens, blending2base_mapper, em_mapper):
        """Create dynamic token mapping between vocabularies"""
        
        # Apply exact matching first
        em_converted_base_topk_ids = blending2base_mapper[blending_topk_ids]
        
        # Find unmapped tokens
        miss_hit_mask = torch.eq(em_converted_base_topk_ids, 0)
        
        if self.simi_threshold > 0.0001 and miss_hit_mask.any():
            # Apply fuzzy matching for unmapped tokens
            unmapped_blending_list = []
            unmapped_blending_tokens = []
            candidate_list = []
            candidate_tokens = []
            
            for pos in torch.nonzero(miss_hit_mask):
                unmapped_blending_list.append(blending_topk_ids[pos[0]][pos[1]])
                unmapped_blending_tokens.append(blending_topk_tokens[pos[0]][pos[1]])
                candidate_list.append(base_topk_ids[pos[0]])
                candidate_tokens.append(base_topk_tokens[pos[0]])
            
            # Find best matches using edit distance
            matched_ids = torch.nonzero(torch.eq(blending2base_mapper, 0)).reshape(-1).tolist()
            matched_set = set(matched_ids)
            
            for blending_id, blending_token, cand_ids, cand_tokens in zip(
                unmapped_blending_list, unmapped_blending_tokens, candidate_list, candidate_tokens
            ):
                if em_mapper[blending_id] != 0:
                    continue
                    
                cand_ids = cand_ids.tolist()
                cand_mapper = {tid: tok for tok, tid in zip(cand_tokens, cand_ids)}
                available_cands = list(set(cand_ids).difference(matched_set))
                
                if not available_cands:
                    continue
                    
                # Find best match using edit distance
                min_dist = float('inf')
                best_match_id = 0
                
                for cand_id in available_cands:
                    cand_token = cand_mapper[cand_id]
                    dist = self.token_distance_func(blending_token, cand_token)
                    
                    if dist < self.simi_threshold and dist < min_dist:
                        best_match_id = cand_id
                        min_dist = dist
                
                if best_match_id != 0:
                    blending2base_mapper[blending_id] = best_match_id
        
        converted_base_topk_ids = blending2base_mapper[blending_topk_ids].to(blending_topk_ids.device)
        unmatch_mask = torch.eq(converted_base_topk_ids, 0)
        masked_blending_topk_ids = blending_topk_ids.masked_fill_(unmatch_mask, 0)
        
        return converted_base_topk_ids, masked_blending_topk_ids

    def get_special_token(self, tokenizer):
        """Get special token for tokenizer"""
        tokenizer_class = tokenizer.__class__
        if tokenizer_class in self.TOKENIZER_TO_SPECIAL_TOKEN:
            return self.TOKENIZER_TO_SPECIAL_TOKEN[tokenizer_class]
        else:
            # Default fallback
            return getattr(tokenizer, 'pad_token', '_')

    def token_distance_func(self, token_a, token_b):
        """Calculate edit distance between two tokens - optimized for BERT + LLM2Vec"""
        
        # Handle special token mappings for BERT + LLM2Vec
        spec_tok_mapper = {
            '</s>': '<|im_end|>',
            '<|endoftext|>': '<|endoftext|>',
            '<s>': '[CLS]',  # Map LLM2Vec start token to BERT CLS
            '[SEP]': '</s>',  # Map BERT SEP to end token
        }
        
        if token_a in spec_tok_mapper and token_b in spec_tok_mapper.values():
            return 0.0
        if token_b in spec_tok_mapper and token_a in spec_tok_mapper.values():
            return 0.0
        if token_a in spec_tok_mapper and spec_tok_mapper[token_a] == token_b:
            return 0.0
        if token_b in spec_tok_mapper and spec_tok_mapper[token_b] == token_a:
            return 0.0
            
        # Clean tokens - handle BERT subword tokens (##) and LLM2Vec tokens
        clean_a = token_a.replace(" ", "").replace("##", "").replace("Ġ", "")
        clean_b = token_b.replace(" ", "").replace("##", "").replace("Ġ", "")
        
        if len(clean_a) == len(clean_b) == 0:
            return 0.0
            
        # Calculate normalized edit distance
        dist = editdistance.eval(clean_a, clean_b)
        normalized_dist = dist / (len(clean_a) + len(clean_b))
        
        return normalized_dist

    def dist_func(self, logits, teacher_logits, target=None, reduction=None):
        """Compute KL divergence between student and teacher logits"""
        lprobs = torch.log_softmax(logits / self.kd_temp, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) * (self.kd_temp ** 2)
        
        kld = teacher_probs * (teacher_lprobs - lprobs)
        
        # Handle inf values
        inf_mask = kld.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0)
        
        # Compute mean over valid elements
        kld_sum = kld.sum()
        valid_elements = torch.sum(~inf_mask).item()
        
        if valid_elements > 0:
            return kld_sum / valid_elements
        else:
            return torch.tensor(0.0, device=logits.device)

    def merge_tensor(self, values, mapping_list):
        """Merge tensor values according to mapping list"""
        merged_values = []
        for ids in mapping_list:
            if isinstance(ids, list) and len(ids) > 0:
                if len(ids) == 1:
                    merged_values.append(values[ids[0]])
                else:
                    merged_values.append(values[ids].mean(dim=0))
            else:
                # Handle single index case
                merged_values.append(values[ids])
                
        return torch.stack(merged_values, dim=0)

    def dtw(self, series_1, series_2, series1_factor, series2_factor, norm_func=None):
        """Dynamic Time Warping for sequence alignment"""
        
        if norm_func is None:
            norm_func = lambda a, b: editdistance.eval(str(a), str(b))
        
        matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        
        for i, (vec1, fc1) in enumerate(zip(series_1, series1_factor)):
            for j, (vec2, fc2) in enumerate(zip(series_2, series2_factor)):
                cost = norm_func(vec1, vec2) * fc1 * fc2
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                )
        
        matrix = matrix[1:, 1:]
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        matches = []
        mappings_series_1 = [[] for _ in range(matrix.shape[0])]
        mappings_series_2 = [[] for _ in range(matrix.shape[1])]
        
        while i > 0 or j > 0:
            matches.append((i, j))
            mappings_series_1[i].append(j)
            mappings_series_2[j].append(i)
            
            option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
            option_up = matrix[i - 1, j] if i > 0 else np.inf
            option_left = matrix[i, j - 1] if j > 0 else np.inf
            move = np.argmin([option_diag, option_up, option_left])
            
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        
        matches.append((0, 0))
        mappings_series_1[0].append(0)
        mappings_series_2[0].append(0)
        matches.reverse()
        
        for mp in mappings_series_1:
            mp.reverse()
        for mp in mappings_series_2:
            mp.reverse()
            
        return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
