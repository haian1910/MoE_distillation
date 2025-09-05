import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,  
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from utils import log_rank
from huggingface_hub import login
import torch.distributed as dist
import os


class ExpertNetwork(nn.Module):
    """Individual expert network for MoE with flexible output dimension"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ExpertNetwork, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.expert(x)

class GatingNetwork(nn.Module):
    """Gating network to compute expert weights"""
    def __init__(self, input_dim, num_experts, hidden_dim=1024):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x):
        # x is the [CLS] token representation
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        return weights

class MoELayer(nn.Module):
    """Mixture of Experts layer with heterogeneous expert architectures"""
    def __init__(self, input_dim, teacher1_dim=4096, teacher2_dim=1024, num_experts=6, expert_hidden_dim=1024):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.teacher1_dim = teacher1_dim
        self.teacher2_dim = teacher2_dim
        
        # Validate expert configuration
        if num_experts % 2 != 0:
            raise ValueError(f"num_experts must be even to split between two teachers, got {num_experts}")
        
        self.num_experts_per_teacher = num_experts // 2
        
        # Create expert networks - first half for teacher1, second half for teacher2
        self.experts = nn.ModuleList()
        
        # First half: experts for teacher1 (e.g., LLM2Vec with 4096 dim)
        for _ in range(self.num_experts_per_teacher):
            expert = ExpertNetwork(input_dim, expert_hidden_dim, teacher1_dim)
            self.experts.append(expert)
        
        # Second half: experts for teacher2 (e.g., Qwen with 1024 dim)
        for _ in range(self.num_experts_per_teacher):
            expert = ExpertNetwork(input_dim, expert_hidden_dim, teacher2_dim)
            self.experts.append(expert)
        
        # Gating network
        self.gating_network = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        """
        Args:
            x: [CLS] token representation [batch_size, input_dim]
        Returns:
            expert_outputs: List of outputs from each expert with varying dimensions
            gating_weights: Gating weights [batch_size, num_experts]
            teacher1_output: Weighted combination for teacher1 experts [batch_size, teacher1_dim]
            teacher2_output: Weighted combination for teacher2 experts [batch_size, teacher2_dim]
        """
        batch_size = x.size(0)
        
        # Compute gating weights
        gating_weights = self.gating_network(x)  # [batch_size, num_experts]
        
        # Get outputs from all experts
        expert_outputs = []
        teacher1_outputs = []
        teacher2_outputs = []
        
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # [batch_size, output_dim]
            expert_outputs.append(expert_output)
            
            if i < self.num_experts_per_teacher:
                # First half: teacher1 experts
                teacher1_outputs.append(expert_output)
            else:
                # Second half: teacher2 experts
                teacher2_outputs.append(expert_output)
        
        # Stack outputs by teacher
        teacher1_outputs_stacked = torch.stack(teacher1_outputs, dim=1)  # [batch_size, num_experts_per_teacher, teacher1_dim]
        teacher2_outputs_stacked = torch.stack(teacher2_outputs, dim=1)  # [batch_size, num_experts_per_teacher, teacher2_dim]
        
        # Split gating weights by teacher
        teacher1_weights = gating_weights[:, :self.num_experts_per_teacher]  # [batch_size, num_experts_per_teacher]
        teacher2_weights = gating_weights[:, self.num_experts_per_teacher:]  # [batch_size, num_experts_per_teacher]
        
        # Normalize weights within each teacher group
        teacher1_weights_norm = F.softmax(teacher1_weights, dim=-1)
        teacher2_weights_norm = F.softmax(teacher2_weights, dim=-1)
        
        # Compute weighted combinations for each teacher
        teacher1_weights_expanded = teacher1_weights_norm.unsqueeze(-1)  # [batch_size, num_experts_per_teacher, 1]
        teacher2_weights_expanded = teacher2_weights_norm.unsqueeze(-1)  # [batch_size, num_experts_per_teacher, 1]
        
        teacher1_final_output = torch.sum(teacher1_outputs_stacked * teacher1_weights_expanded, dim=1)  # [batch_size, teacher1_dim]
        teacher2_final_output = torch.sum(teacher2_outputs_stacked * teacher2_weights_expanded, dim=1)  # [batch_size, teacher2_dim]

     
        # Stack expert outputs for easier computation
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)

        # Compute weighted combination
        gating_weights_expanded = gating_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        final_output = torch.sum(expert_outputs_stacked * gating_weights_expanded, dim=1)  
        return expert_outputs, gating_weights, final_output, teacher1_final_output, teacher2_final_output

class MoEDistilledBERT(nn.Module):
    """BERT with MoE layer for knowledge distillation from two teachers"""
    def __init__(self, bert_model, teacher1_hidden_size=4096, teacher2_hidden_size=1024, 
                 num_experts=6, expert_hidden_dim=1024):
        super(MoEDistilledBERT, self).__init__()
        self.bert = bert_model
        self.bert_hidden_size = bert_model.config.hidden_size
        self.teacher1_hidden_size = teacher1_hidden_size
        self.teacher2_hidden_size = teacher2_hidden_size
        self.config = bert_model.config  # Make sure config is accessible
        
        # MoE layer with heterogeneous experts
        self.moe_layer = MoELayer(
            input_dim=self.bert_hidden_size,
            teacher1_dim=teacher1_hidden_size,
            teacher2_dim=teacher2_hidden_size,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )
        
        # Keep original classifier for final predictions
        self.classifier = bert_model.classifier
        
        # Make sure dropout is accessible
        self.dropout = bert_model.dropout
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
            return_moe_outputs=True, output_hidden_states=True, 
            return_dict=False, labels=None):
        """
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            return_moe_outputs: Whether to return MoE intermediate outputs
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return a dict or tuple
            labels: Labels for computing loss (if needed)
        """
        # Get BERT outputs with optional hidden states
        bert_outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Get [CLS] token representation
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout (same as original BERT)
        cls_output = self.dropout(cls_output)
        
        # Pass through MoE layer
        expert_outputs, gating_weights, final_moe_output, teacher1_final_output, teacher2_final_output = self.moe_layer(cls_output)
        
        # Get final classification logits using original classifier
        classification_logits = self.classifier(final_moe_output)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(classification_logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # Prepare output based on return_dict and return_moe_outputs
        if return_dict or True:  # Always return dict for consistency
            output = {
                'loss': loss,
                'logits': classification_logits,
                'pooler_output': final_moe_output,
                'last_hidden_state': bert_outputs.last_hidden_state,
            }
            
            # Add hidden states if requested
            if output_hidden_states:
                output['hidden_states'] = bert_outputs.hidden_states
                print(f"DEBUG: Added hidden_states with {len(bert_outputs.hidden_states)} layers")
            
            # Add MoE outputs if requested
            if return_moe_outputs:
                output.update({
                    'expert_outputs': expert_outputs,
                    'gating_weights': gating_weights,
                    'teacher1_output': teacher1_final_output,
                    'teacher2_output': teacher2_final_output,
                    'final_moe_output': final_moe_output,
                    'cls_representation': final_moe_output,
                })
            
            return output
        else:
            # Legacy tuple return format
            if return_moe_outputs:
                return {
                    'loss': loss,
                    'logits': classification_logits,
                    'expert_outputs': 'expert_outputs',
                    'gating_weights': 'gating_weights',
                    'teacher1_output': 'teacher1_output',
                    'teacher2_output': 'teacher2_output',
                    'final_moe_output': 'final_moe_output',
                    'cls_representation': final_moe_output,
                    'hidden_states': bert_outputs.hidden_states if output_hidden_states else None,
                    'pooler_output': final_moe_output
                }
            else:
                if loss is not None:
                    return {'loss': loss, 'logits': classification_logits}
                else:
                    return classification_logits
    
    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        """
        Save the model to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save MoE specific config
        moe_config = {
            'num_experts': self.moe_layer.num_experts,
            'num_experts_per_teacher': self.moe_layer.num_experts_per_teacher,
            'expert_hidden_dim': self.moe_layer.experts[0].expert[0].out_features,
            'teacher1_hidden_size': self.teacher1_hidden_size,
            'teacher2_hidden_size': self.teacher2_hidden_size,
            'bert_hidden_size': self.bert_hidden_size
        }
        moe_config_path = os.path.join(save_directory, "moe_config.json")
        with open(moe_config_path, 'w') as f:
            json.dump(moe_config, f, indent=2)
        
        log_rank(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a model from a directory.
        """
        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = AutoConfig.from_dict(config_dict)
        
        # Load MoE config
        moe_config_path = os.path.join(pretrained_model_name_or_path, "moe_config.json")
        with open(moe_config_path, 'r') as f:
            moe_config = json.load(f)
        
        # Create base BERT model
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use base BERT as template
            config=config,
            **kwargs
        )
        
        # Create MoE model
        model = cls(
            bert_model=bert_model,
            teacher1_hidden_size=moe_config['teacher1_hidden_size'],
            teacher2_hidden_size=moe_config['teacher2_hidden_size'],
            num_experts=moe_config['num_experts'],
            expert_hidden_dim=moe_config['expert_hidden_dim']
        )
        
        # Load state dict
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model
                
    def get_input_embeddings(self):
        """For compatibility with transformers"""
        return self.bert.bert.embeddings.word_embeddings
        
    def set_input_embeddings(self, new_embeddings):
        """For compatibility with transformers"""
        self.bert.bert.embeddings.word_embeddings = new_embeddings
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
        else:
            self.bert.bert.gradient_checkpointing = True
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings"""
        return self.bert.resize_token_embeddings(new_num_tokens)
    
    def get_output_embeddings(self):
        """Get output embeddings"""
        return self.classifier
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings"""
        self.classifier = new_embeddings
    
    def tie_weights(self):
        """Tie weights if needed"""
        pass  # No weight tying needed for this model
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation (if needed)"""
        return {"input_ids": input_ids}

class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student_model, self.student_tokenizer = self.load_student_model()
        
        if self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizers = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizers = None, {}
        if self.teacher_model and args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")

        if self.args.teacher_model_2_path is not None:
            self.teacher_model_2, self.teacher_tokenizers_2 = self.load_teacher_model_2()
        else:
            self.teacher_model_2, self.teacher_tokenizers_2 = None, {}
        if self.teacher_model_2 and args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")

    @staticmethod
    def add_distiller_args(parser):
        group = parser.add_argument_group("distiller", "distiller configurations")
        group.add_argument("--projector-config-path", type=str, default=None,
                           help='path to projector_config.json')
        group.add_argument("--projector-path", type=str, default=None,
                           help='path to pretrained projector')
        group.add_argument("--projector-lr", type=float, default=0.001,
                           help='learning rate only for projection')
        group.add_argument("--pretrained-projector", type=str, default=None,
                           help='pretrained projector name')
        group.add_argument("--pretrained-projector-lr", type=float, default=0.001,
                           help='learning rate only for pretrained projector')
        group.add_argument("--vocab-alignment-path", type=str, default=None,
                           help='path for the vocab alignment file')
        group.add_argument("--teacher-to-student-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, teacher-to-student)')
        group.add_argument("--teacher-to-student-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, teacher-to-student)')
        group.add_argument("--student-to-teacher-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, student-to-teacher)')
        group.add_argument("--student-to-teacher-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, student-to-teacher)')
        # MoE specific arguments
        group.add_argument("--num-experts", type=int, default=6,
                           help='number of experts in MoE layer (must be even)')
        group.add_argument("--expert-hidden-dim", type=int, default=1024,
                           help='hidden dimension for expert networks')
        group.add_argument("--teacher1-hidden-dim", type=int, default=4096,
                           help='hidden dimension for teacher1 (LLM2Vec)')
        group.add_argument("--teacher2-hidden-dim", type=int, default=1024,
                           help='hidden dimension for teacher2 (Qwen)')
        group.add_argument("--moe-lr", type=float, default=0.001,
                           help='learning rate for MoE components')
        return parser
    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tokenizer
        
    def set_and_load_existing_projectors(self):
        self.projectors = nn.ModuleDict()
        projector_config = json.load(open(self.args.projector_config_path))
        name_dict = {
            "s": self.hidden_size, 
            "t": self.teacher_hidden_size,
            "t1": getattr(self.args, 'teacher1_hidden_dim', 4096),  # Teacher1 hidden size
            "t2": getattr(self.args, 'teacher2_hidden_dim', 1024),  # Teacher2 hidden size
            "relu": nn.ReLU()
        }
        # auto-parse projector config strings to construct nn.Module
        for projector_name in projector_config:
            if projector_config[projector_name]["enabled"]:
                self.projectors[projector_name] = nn.Sequential()

                structure = projector_config[projector_name]["structure"].split("-")
                for i in range(len(structure)):
                    if structure[i] not in ["relu"]:
                        coef = 1 if not len(structure[i][:-1]) else int(structure[i][:-1])
                        base_size = name_dict[structure[i][-1:]] if structure[i][-1:] in name_dict else name_dict[structure[i][-2:]]
                        structure[i] = coef * base_size

                for i in range(len(structure) - 1):
                    if isinstance(structure[i], int) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(structure[i], structure[i+1])
                        )
                    elif isinstance(structure[i], int) and isinstance(structure[i+1], str):
                        self.projectors[projector_name].append(
                            name_dict[structure[i+1]]
                        )
                        last_size = structure[i]
                    elif isinstance(structure[i], str) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(last_size, structure[i+1])
                        )
                    else:
                        raise NotImplementedError(f"Invalid structure for '{structure}'")
                        
        # load existing projectors if already have
        self.load_existing_projectors()

    def load_existing_projectors(self):
        if self.args.projector_path is not None:
            projector_path = os.path.join(self.args.projector_path, "projector.pt")
        else:
            projector_path = os.path.join(self.args.model_path, "projector.pt")

        if os.path.exists(projector_path):
            projector_params = torch.load(projector_path, map_location=f"cuda:{self.device}")
            log_rank("Existing projector params: {}".format(list(projector_params.keys())))
            for key in self.projectors:
                try:
                    state_dict = {
                        n.split('.', 1)[1]: projector_params[n] for n in projector_params if n.startswith(key)
                    }
                    self.projectors[key].load_state_dict(state_dict)
                    log_rank("Load projector '{}' from current path.".format(key))
                except:
                    log_rank("Not compatible for projector '{}'".format(key))
                    continue
    
    def load_student_model(self):
        log_rank("Loading student model...")
    
        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError("Invalid model_dtype for f`{self.args.model_dtype}`")

        if self.args.peft is not None: #for LLM2Vec
            if self.args.peft == "lora":
                config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True)
                config.is_model_parallel = False
        
                # lấy tokenizer
                tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
                
                if hasattr(config, "n_embed"):
                    self.hidden_size = config.n_embed
                else:
                    self.hidden_size = config.hidden_size
        
                config.num_labels = self.args.num_labels
                model = AutoModelForSequenceClassification.from_pretrained(
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )

                model.config.pad_token_id = 2
                    
                model = PeftModel.from_pretrained(
                    model,
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                )
                model = model.merge_and_unload()  # This can take several minutes on cpu

                model = PeftModel.from_pretrained(
                    model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
                )
                model = model.merge_and_unload() 
                # Apply new LoRA adapter for fine-tuning
                if self.args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,  # SEQ_CLS là hợp lý nếu đang làm classification
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"
                        ]
                    )
                    model = get_peft_model(model, peft_config)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in model.parameters())
                    print(f"Trainable parameters: {trainable_params}/{all_params} ({trainable_params/all_params:.2%})")
            else:
                raise NotImplementedError
        else: #for BERT with MoE
            config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
            config.is_model_parallel = False
    
            # lấy tokenizer
            tokenizer = self.load_tokenizer("bert-base-uncased")
            
            if hasattr(config, "n_embed"):
                self.hidden_size = config.n_embed
            else:
                self.hidden_size = config.hidden_size
            config.num_labels = self.args.num_labels
            
            # Load base BERT model
            bert_model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", 
                config=config, 
                device_map=None, 
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            
            # Wrap with MoE layer with heterogeneous experts
            teacher1_hidden_size = getattr(self.args, 'teacher1_hidden_dim', 4096)  # LLM2Vec
            teacher2_hidden_size = getattr(self.args, 'teacher2_hidden_dim', 1024)  # Qwen
            
            model = MoEDistilledBERT(
                bert_model=bert_model,
                teacher1_hidden_size=teacher1_hidden_size,
                teacher2_hidden_size=teacher2_hidden_size,
                num_experts=getattr(self.args, 'num_experts', 6),
                expert_hidden_dim=getattr(self.args, 'expert_hidden_dim', 1024)
            )
            
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            trust_remote_code=True
        )
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        config.num_labels = self.args.num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        model.config.pad_token_id = 2
        teacher_model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        )    
        
        teacher_model = teacher_model.merge_and_unload()  # This can take several minutes on cpu

        # Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
        teacher_model = PeftModel.from_pretrained(
            teacher_model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
        )
        teacher_model = teacher_model.merge_and_unload()

        if hasattr(self.args, 'teacher_model_path') and self.args.teacher_model_path:
            
            # Path to the adapter model weights
            adapter_path = os.path.join(self.args.teacher_model_path, "adapter_model.bin")
            fixed_adapter_path = adapter_path + ".fixed"
            if not os.path.exists(fixed_adapter_path):
                if dist.get_rank() == 0:
                    # Load the checkpoint and fix the keys
                    checkpoint = torch.load(adapter_path)            
                    fixed_checkpoint = {}
                    
                    for key, value in checkpoint.items():
                        if "lora_A.weight" in key and "default" not in key:
                            key = key.replace("lora_A.weight", "lora_A.default.weight")
                        if "lora_B.weight" in key and "default" not in key:
                            key = key.replace("lora_B.weight", "lora_B.default.weight")
                        if "base_model.model.base_model.model" in key:
                            key = key.replace("base_model.model.base_model.model", "base_model.model")
                            
                        fixed_checkpoint[key] = value
                    
                    # Save the fixed checkpoint back to the original file
                    if fixed_checkpoint: 
                        torch.save(fixed_checkpoint, fixed_adapter_path)
            
            dist.barrier()  
            
            teacher_model = PeftModel.from_pretrained(
                teacher_model,
                self.args.teacher_model_path,
                adapter_name="default",
                adapter_weights_path=fixed_adapter_path
            )

        classifier_path = os.path.join(self.args.teacher_model_path, "classifier_head.bin")
        if os.path.exists(classifier_path):
            log_rank("Loading classifier head from trained model...")
            classifier_state_dict = torch.load(classifier_path, map_location="cpu")
            teacher_model.score.load_state_dict(classifier_state_dict)
        else:
            log_rank("No classifier head found in teacher model path. Using default classifier.")
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        return teacher_model, tokenizer

    def load_teacher_model_2(self):
        # log_rank("Loading teacher model")
        # config = AutoConfig.from_pretrained(
        #     "Qwen/Qwen3-Embedding-0.6B",
        #     trust_remote_code=True
        # )
        # config.is_model_parallel = False

        # tokenizer = self.load_tokenizer("Qwen/Qwen3-Embedding-0.6B")

        # if hasattr(config, "n_embed"):
        #     self.teacher_hidden_size = config.n_embed
        # else:
        #     self.teacher_hidden_size = config.hidden_size

        # config.num_labels = self.args.num_labels
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     "Qwen/Qwen3-Embedding-0.6B",
        #     config=config,
        #     device_map=None,
        #     torch_dtype=self.dtype,
        #     trust_remote_code=True,
        # )
        
        # # Set pad token ID if needed
        # if model.config.pad_token_id is None:
        #     model.config.pad_token_id = tokenizer.pad_token_id
        
        # # Check if we should load pre-trained weights before fine-tuning
        # if hasattr(self.args, 'pretrained_model_path') and self.args.pretrained_model_path:
        #     # Try to load the full model weights
        #     model_path = os.path.join(self.args.pretrained_model_path, "pytorch_model.bin")
        #     if os.path.exists(model_path):
        #         log_rank("Loading pretrained weights before fine-tuning...")
        #         model_state_dict = torch.load(model_path, map_location="cpu")
        #         model.load_state_dict(model_state_dict, strict=False)

        #     # Try to load classifier head if available
        #     classifier_path = os.path.join(self.args.pretrained_model_path, "classifier_head.bin")
        #     if os.path.exists(classifier_path):
        #         log_rank("Loading classifier head...")
        #         classifier_state_dict = torch.load(classifier_path, map_location="cpu")
        #         # Check if the model has a score attribute or classifier attribute
        #         if hasattr(model, "score"):
        #             model.score.load_state_dict(classifier_state_dict)
        #         elif hasattr(model, "classifier"):
        #             model.classifier.load_state_dict(classifier_state_dict)
        #         else:
        #             log_rank("Warning: Model does not have a recognized classifier attribute. Classifier head not loaded.")
        
        # # Make all parameters trainable for full fine-tuning
        # for param in model.parameters():
        #     param.requires_grad = False
        
        # return model, tokenizer
        log_rank("Loading teacher model")
        config = AutoConfig.from_pretrained(
            "BAAI/bge-m3",
            trust_remote_code=True
        )
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer("BAAI/bge-m3")

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        config.num_labels = self.args.num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-m3",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        
        # Set pad token ID if needed
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # Check if we should load pre-trained weights before fine-tuning
        if hasattr(self.args, 'pretrained_model_path') and self.args.pretrained_model_path:
            # Try to load the full model weights
            model_path = os.path.join(self.args.pretrained_model_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                log_rank("Loading pretrained weights before fine-tuning...")
                model_state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(model_state_dict, strict=False)

            # Try to load classifier head if available
            classifier_path = os.path.join(self.args.pretrained_model_path, "classifier_head.bin")
            if os.path.exists(classifier_path):
                log_rank("Loading classifier head...")
                classifier_state_dict = torch.load(classifier_path, map_location="cpu")
                # Check if the model has a score attribute or classifier attribute
                if hasattr(model, "score"):
                    model.score.load_state_dict(classifier_state_dict)
                elif hasattr(model, "classifier"):
                    model.classifier.load_state_dict(classifier_state_dict)
                else:
                    log_rank("Warning: Model does not have a recognized classifier attribute. Classifier head not loaded.")
        
        # Make all parameters trainable for full fine-tuning
        for param in model.parameters():
            param.requires_grad = False
        
        return model, tokenizer
    
    def add_optimizer_param_group(self, optimizer):
        """
        Add parameter groups to optimizer, ensuring no parameter appears in multiple groups
        """
        # Get all parameters already in the optimizer
        existing_params = set()
        for group in optimizer.param_groups:
            for param in group['params']:
                existing_params.add(param)
        
        # Add MoE parameters if they exist and haven't been added yet
        if hasattr(self.student_model, 'moe_layer'):
            moe_params = []
            for param in self.student_model.moe_layer.parameters():
                if param not in existing_params:
                    moe_params.append(param)
                    existing_params.add(param)
            
            if moe_params:
                optimizer.add_param_group({
                    "params": moe_params,
                    "lr": getattr(self.args, 'moe_lr', 0.001)
                })
                print(f"Added {len(moe_params)} MoE parameters to optimizer")
        
        # Add projector parameters if they exist
        if hasattr(self, "projectors"):
            if self.args.projector_lr:
                pretrained_proj = self.args.pretrained_projector.split(",") if self.args.pretrained_projector is not None else []
                
                # Regular projector parameters
                regular_proj_params = []
                for proj_name in self.projectors:
                    if proj_name not in pretrained_proj:
                        for param in self.projectors[proj_name].parameters():
                            if param not in existing_params:
                                regular_proj_params.append(param)
                                existing_params.add(param)
                
                if regular_proj_params:
                    optimizer.add_param_group({
                        "params": regular_proj_params,
                        "lr": self.args.projector_lr
                    })
                    print(f"Added {len(regular_proj_params)} regular projector parameters to optimizer")
                
                # Pretrained projector parameters
                pretrained_proj_params = []
                for proj_name in self.projectors:
                    if proj_name in pretrained_proj:
                        for param in self.projectors[proj_name].parameters():
                            if param not in existing_params:
                                pretrained_proj_params.append(param)
                                existing_params.add(param)
                
                if pretrained_proj_params:
                    optimizer.add_param_group({
                        "params": pretrained_proj_params,
                        "lr": self.args.pretrained_projector_lr
                    })
                    print(f"Added {len(pretrained_proj_params)} pretrained projector parameters to optimizer")
            else:
                # All projector parameters with same learning rate
                all_proj_params = []
                for proj_name in self.projectors:
                    for param in self.projectors[proj_name].parameters():
                        if param not in existing_params:
                            all_proj_params.append(param)
                            existing_params.add(param)
                
                if all_proj_params:
                    optimizer.add_param_group({
                        "params": all_proj_params,
                    })
                    print(f"Added {len(all_proj_params)} projector parameters to optimizer")
        
        return optimizer
    def get_model_parameters(self):
        """
        Get categorized model parameters for better debugging
        """
        params_info = {
            'bert_base': [],
            'moe_layer': [],
            'projectors': [],
            'classifier': [],
            'total_params': 0
        }
        
        # Get BERT base parameters
        if hasattr(self.student_model, 'bert'):
            for name, param in self.student_model.bert.named_parameters():
                if 'classifier' not in name:
                    params_info['bert_base'].append((name, param.numel()))
        
        # Get MoE parameters
        if hasattr(self.student_model, 'moe_layer'):
            for name, param in self.student_model.moe_layer.named_parameters():
                params_info['moe_layer'].append((name, param.numel()))
        
        # Get projector parameters
        if hasattr(self, 'projectors'):
            for proj_name, projector in self.projectors.items():
                for name, param in projector.named_parameters():
                    params_info['projectors'].append((f"{proj_name}.{name}", param.numel()))
        
        # Get classifier parameters
        if hasattr(self.student_model, 'classifier'):
            for name, param in self.student_model.classifier.named_parameters():
                params_info['classifier'].append((name, param.numel()))
        
        # Calculate total parameters
        params_info['total_params'] = sum(p.numel() for p in self.student_model.parameters())
        
        return params_info

    def print_model_info(self):
        """Print detailed model information"""
        params_info = self.get_model_parameters()
        
        print("="*50)
        print("MODEL PARAMETER SUMMARY")
        print("="*50)
        
        bert_params = sum(count for _, count in params_info['bert_base'])
        moe_params = sum(count for _, count in params_info['moe_layer'])
        proj_params = sum(count for _, count in params_info['projectors'])
        classifier_params = sum(count for _, count in params_info['classifier'])
        
        print(f"BERT Base Parameters: {bert_params:,}")
        print(f"MoE Layer Parameters: {moe_params:,}")
        print(f"Projector Parameters: {proj_params:,}")
        print(f"Classifier Parameters: {classifier_params:,}")
        print(f"Total Parameters: {params_info['total_params']:,}")
        print("="*50)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/params_info['total_params']:.2%})")
        print("="*50)

    def save_model_components(self, save_path):
        """
        Save model components separately for easier loading
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save main model
        if hasattr(self.student_model, 'bert'):
            torch.save(self.student_model.bert.state_dict(), 
                    os.path.join(save_path, "bert_model.pt"))
        
        # Save MoE layer
        if hasattr(self.student_model, 'moe_layer'):
            torch.save(self.student_model.moe_layer.state_dict(), 
                    os.path.join(save_path, "moe_layer.pt"))
        
        # Save classifier
        if hasattr(self.student_model, 'classifier'):
            torch.save(self.student_model.classifier.state_dict(), 
                    os.path.join(save_path, "classifier.pt"))
        
        # Save projectors
        if hasattr(self, 'projectors'):
            torch.save(self.projectors.state_dict(), 
                    os.path.join(save_path, "projectors.pt"))
        
        # Save tokenizer
        if hasattr(self, 'student_tokenizer'):
            self.student_tokenizer.save_pretrained(save_path)
        
        print(f"Model components saved to {save_path}")

    def load_model_components(self, load_path):
        """
        Load model components from separate files
        """
        # Load BERT model
        bert_path = os.path.join(load_path, "bert_model.pt")
        if os.path.exists(bert_path) and hasattr(self.student_model, 'bert'):
            self.student_model.bert.load_state_dict(torch.load(bert_path))
            print("Loaded BERT model")
        
        # Load MoE layer
        moe_path = os.path.join(load_path, "moe_layer.pt")
        if os.path.exists(moe_path) and hasattr(self.student_model, 'moe_layer'):
            self.student_model.moe_layer.load_state_dict(torch.load(moe_path))
            print("Loaded MoE layer")
        
        # Load classifier
        classifier_path = os.path.join(load_path, "classifier.pt")
        if os.path.exists(classifier_path) and hasattr(self.student_model, 'classifier'):
            self.student_model.classifier.load_state_dict(torch.load(classifier_path))
            print("Loaded classifier")
        
        # Load projectors
        projectors_path = os.path.join(load_path, "projectors.pt")
        if os.path.exists(projectors_path) and hasattr(self, 'projectors'):
            self.projectors.load_state_dict(torch.load(projectors_path))
            print("Loaded projectors")

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch["output_batch"]
        loss, logging_output = criterion(
            self,
            input_data, 
            output_data,
            logging_output,
            loss_denom,
        )
        return loss, logging_output
    
