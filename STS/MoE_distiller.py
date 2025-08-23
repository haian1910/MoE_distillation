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

class STSModel(nn.Module):
    """Wrapper for STS (Semantic Textual Similarity) tasks using a base model"""
    def __init__(self, base_model):
        super(STSModel, self).__init__()
        self.base_model = base_model
        self.config = base_model.config  # Expose config for save_pretrained
        
        # Get the hidden size from the base model
        self.hidden_size = base_model.config.hidden_size

        # Create a regression head for STS score prediction (0-5 scale typically)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1)
        )

        # Check model type to determine which arguments it accepts
        self.uses_token_type_ids = hasattr(base_model.config, "type_vocab_size") and base_model.config.type_vocab_size > 0
        
    def device(self):
        return next(self.parameters()).device
        
    def get_input_embeddings(self):
        """Return the input embeddings from the base model"""
        if hasattr(self.base_model, "get_input_embeddings"):
            return self.base_model.get_input_embeddings()
        elif hasattr(self.base_model, "bert") and hasattr(self.base_model.bert, "embeddings"):
            return self.base_model.bert.embeddings.word_embeddings  # BERT-specific
        elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            return self.base_model.model.embed_tokens  # LLaMA-like
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            return self.base_model.transformer.wte  # GPT-like
        else:
            raise NotImplementedError("Unsupported model architecture for embedding extraction")
            
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, token_type_ids=None, labels=None, **kwargs):
        # Filter kwargs to only include parameters accepted by the base model
        filtered_kwargs = {}
        for key, value in kwargs.items():
            # Skip labels as they'll be handled separately
            if key == 'labels':
                continue  # Don't pass labels to base model
            if key == 'token_type_ids' and not self.uses_token_type_ids:
                continue  # Don't pass token_type_ids if model doesn't use them

            filtered_kwargs[key] = value

        # Only pass token_type_ids if the model supports it
        if self.uses_token_type_ids and token_type_ids is not None:
            filtered_kwargs["token_type_ids"] = token_type_ids

        # Make sure we get hidden states and attentions
        filtered_kwargs["output_hidden_states"] = True
        filtered_kwargs["output_attentions"] = True

        # Get outputs from the base model with filtered kwargs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **filtered_kwargs
        )

        # Get the CLS token representation (for sentence embedding)
        pooled_output = outputs.last_hidden_state[:, 0]

        # Apply the regressor to get similarity score
        score = self.regressor(pooled_output)
        
        # Ensure the predicted score is within a reasonable range (0-5)
        score = torch.sigmoid(score) * 5.0

        loss = None
        if labels is not None:
            # Use MSE loss for regression task
            loss_fct = nn.MSELoss()
            loss = loss_fct(score.view(-1), labels.view(-1))

        # Create a comprehensive output structure
        class STSModelOutput:
            def __init__(self, loss, scores, hidden_states, attentions, last_hidden_state=None):
                self.loss = loss
                self.scores = scores
                self.hidden_states = hidden_states
                self.attentions = attentions
                self.last_hidden_state = last_hidden_state

        # Return complete output with original hidden states and attentions
        return STSModelOutput(
            loss=loss,
            scores=score,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            last_hidden_state=outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None
        )

    # Add HuggingFace compatibility methods
    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        """Save the model to the specified directory."""
        # Save the regressor separately
        os.makedirs(save_directory, exist_ok=True)
        regressor_path = os.path.join(save_directory, "regressor.pt")
        torch.save(self.regressor.state_dict(), regressor_path)

        # Save wrapper config
        config_dict = {
            "uses_token_type_ids": self.uses_token_type_ids
        }
        with open(os.path.join(save_directory, "sts_model_config.json"), "w") as f:
            json.dump(config_dict, f)

        # Save the base model
        return self.base_model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from pretrained."""
        # First load the base model
        base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Create the wrapper
        model = cls(base_model)

        # Load regressor weights if they exist
        regressor_path = os.path.join(pretrained_model_name_or_path, "regressor.pt")
        if os.path.exists(regressor_path):
            regressor_state_dict = torch.load(regressor_path, map_location="cpu")
            model.regressor.load_state_dict(regressor_state_dict)

        return model

class ExpertNetwork(nn.Module):
    """Individual expert network for MoE"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ExpertNetwork, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim) # Output dim same as input dim for residual addition
        )
    
    def forward(self, x):
        return self.expert(x)

class GatingNetwork(nn.Module):
    """Gating network to compute expert weights"""
    def __init__(self, input_dim, num_experts, hidden_dim=128):
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
    """Mixture of Experts layer"""
    def __init__(self, input_dim, output_dim, num_experts=3, expert_hidden_dim=128):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = input_dim

        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating_network = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        """
        Args:
            x: [CLS] token representation [batch_size, input_dim]
        Returns:
            expert_outputs: List of outputs from each expert [batch_size, output_dim]
            gating_weights: Gating weights [batch_size, num_experts]
            final_output: Weighted combination of expert outputs [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Compute gating weights
        gating_weights = self.gating_network(x)  # [batch_size, num_experts]
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [batch_size, output_dim]
            expert_outputs.append(expert_output)
        
        # Stack expert outputs for easier computation
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # Compute weighted combination
        gating_weights_expanded = gating_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        final_output = torch.sum(expert_outputs_stacked * gating_weights_expanded, dim=1)  # [batch_size, output_dim]
        
        return expert_outputs, gating_weights, final_output

class MoEDistilledBERT(nn.Module):
    """BERT with MoE layer for knowledge distillation - supports both classification and STS"""
    def __init__(self, bert_model, teacher_hidden_size, num_experts=3, expert_hidden_dim=128):
        super(MoEDistilledBERT, self).__init__()
        self.bert = bert_model.bert if hasattr(bert_model, 'bert') else bert_model  # Get the base BERT model
        self.bert_hidden_size = self.bert.config.hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.config = self.bert.config  # Make sure config is accessible
        
        # MoE layer
        self.moe_layer = MoELayer(
            input_dim=self.bert_hidden_size,
            output_dim=self.bert_hidden_size,  # Remain unchanged the dim
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )
        
        # Task-specific heads
            # STS regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.bert_hidden_size, self.bert_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.bert_hidden_size // 2, 1)
        )

        # Make sure dropout is accessible
        self.dropout = bert_model.dropout if hasattr(bert_model, 'dropout') else nn.Dropout(0.1)
        
        # Check if model uses token_type_ids
        self.uses_token_type_ids = hasattr(self.config, "type_vocab_size") and self.config.type_vocab_size > 0
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
            return_moe_outputs=True, labels=None, output_hidden_states=True, 
            output_attentions=False, return_dict=True, **kwargs):
        """
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            position_ids: Position ids
            return_moe_outputs: Whether to return MoE intermediate outputs
            labels: Labels for computing loss (if needed)
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attention weights
            return_dict: Whether to return as dictionary
        """
        # Filter kwargs for BERT forward pass
        bert_kwargs = {}
        if self.uses_token_type_ids and token_type_ids is not None:
            bert_kwargs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            bert_kwargs["position_ids"] = position_ids
        
        # Get BERT outputs - always request hidden states for MMD loss
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Always True for distillation
            output_attentions=output_attentions or True,  # Always True for distillation
            **bert_kwargs,
            return_dict=True  # Ensure outputs are in dict format
        )
        
        # Get [CLS] token representation
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout (same as original BERT)
        cls_output = self.dropout(cls_output)
        
        # Pass through MoE layer
        expert_outputs, gating_weights, moe_final_output = self.moe_layer(cls_output)
        
        # For STS, apply regressor to the moe_output to get similarity score
        scores = self.regressor(moe_final_output)
        # Ensure the predicted score is within a reasonable range (0-5)
        scores = torch.sigmoid(scores) * 5.0
        logits = scores  # For compatibility
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(scores.view(-1), labels.view(-1))

        # Create output structure
        class MoEModelOutput:
            def __init__(self, loss, logits, expert_outputs=None, gating_weights=None, 
                        moe_final_output=None, cls_representation=None, hidden_states=None, 
                        attentions=None, scores=None):
                self.loss = loss
                self.logits = logits
                self.scores = scores   # For STS compatibility
                self.expert_outputs = expert_outputs
                self.gating_weights = gating_weights
                self.moe_final_output = moe_final_output
                self.cls_representation = cls_representation
                self.hidden_states = hidden_states
                self.attentions = attentions
                self.last_hidden_state = hidden_states[-1] if hidden_states else None
                
            # Make it accessible as a dictionary as well
            def get(self, key, default=None):
                return getattr(self, key, default)
            
            def __getitem__(self, key):
                return getattr(self, key)
            
            def keys(self):
                return ['loss', 'logits', 'scores', 'expert_outputs', 'gating_weights', 
                    'moe_final_output', 'cls_representation', 'hidden_states', 'attentions']
        
        output = MoEModelOutput(
            loss=loss,
            logits=logits,
            scores=scores,
            expert_outputs=expert_outputs if return_moe_outputs else None,
            gating_weights=gating_weights if return_moe_outputs else None,
            moe_final_output=moe_final_output if return_moe_outputs else None,
            cls_representation=cls_output if return_moe_outputs else None,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions if output_attentions else None
        )
        
        if return_dict:
            return output
        elif loss is not None:
            return output  # Return the full output object for compatibility
        else:
            return logits

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
            'expert_hidden_dim': self.moe_layer.experts[0].expert[0].out_features,
            'teacher_hidden_size': self.teacher_hidden_size,
            'bert_hidden_size': self.bert_hidden_size,
            'uses_token_type_ids': self.uses_token_type_ids
        }
        moe_config_path = os.path.join(save_directory, "moe_config.json")
        with open(moe_config_path, 'w') as f:
            json.dump(moe_config, f, indent=2)
        
        # Save task-specific head
        regressor_path = os.path.join(save_directory, "regressor.pt")
        torch.save(self.regressor.state_dict(), regressor_path)
    
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
        
      
        bert_model = AutoModel.from_pretrained(
            "bert-base-uncased",
            config=config,
            **kwargs
        )
    
        # Create MoE model
        model = cls(
            bert_model=bert_model,
            teacher_hidden_size=moe_config['teacher_hidden_size'],
            num_experts=moe_config['num_experts'],
            expert_hidden_dim=moe_config['expert_hidden_dim'],
        )
        
        # Load state dict
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Load task-specific head if exists
        regressor_path = os.path.join(pretrained_model_name_or_path, "regressor.pt")
        if os.path.exists(regressor_path):
            regressor_state_dict = torch.load(regressor_path, map_location='cpu')
            model.regressor.load_state_dict(regressor_state_dict)
        
        return model
                
    def get_input_embeddings(self):
        """For compatibility with transformers"""
        return self.bert.embeddings.word_embeddings
        
    def set_input_embeddings(self, new_embeddings):
        """For compatibility with transformers"""
        self.bert.embeddings.word_embeddings = new_embeddings
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
        else:
            self.bert.gradient_checkpointing = True
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings"""
        return self.bert.resize_token_embeddings(new_num_tokens)
    
    def get_output_embeddings(self):
        """Get output embeddings"""
       
        return self.regressor
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings"""
       
        self.regressor = new_embeddings
    
    def tie_weights(self):
        """Tie weights if needed"""
        pass  # No weight tying needed for this model
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation (if needed)"""
        return {"input_ids": input_ids}

class STSMoEWrapper(nn.Module):
    """Wrapper for STS tasks using MoE BERT model"""
    def __init__(self, moe_bert_model):
        super(STSMoEWrapper, self).__init__()
        self.moe_bert = moe_bert_model
        self.config = moe_bert_model.config
        self.hidden_size = moe_bert_model.bert_hidden_size
        self.uses_token_type_ids = moe_bert_model.uses_token_type_ids
        
    def device(self):
        return next(self.parameters()).device
        
    def get_input_embeddings(self):
        """Return the input embeddings from the MoE model"""
        return self.moe_bert.get_input_embeddings()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing by delegating to the underlying MoE model"""
        if hasattr(self.moe_bert, 'gradient_checkpointing_enable'):
            self.moe_bert.gradient_checkpointing_enable()
        else:
            # Fallback to enabling on the BERT model directly
            if hasattr(self.moe_bert.bert, 'gradient_checkpointing_enable'):
                self.moe_bert.bert.gradient_checkpointing_enable()
            else:
                self.moe_bert.bert.gradient_checkpointing = True
            
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, token_type_ids=None, labels=None, **kwargs):
      # Extract return_moe_outputs from kwargs to avoid duplicate keyword argument
      return_moe_outputs = kwargs.pop('return_moe_outputs', False)
      
      # Forward pass through MoE BERT
      outputs = self.moe_bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          position_ids=position_ids,
          token_type_ids=token_type_ids,
          labels=labels,
          return_moe_outputs=return_moe_outputs,
          **kwargs
      )
      
      return outputs

    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        """Save the model to the specified directory."""
        return self.moe_bert.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from pretrained."""
        moe_bert = MoEDistilledBERT.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(moe_bert)

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
        group.add_argument("--num-experts", type=int, default=3,
                           help='number of experts in MoE layer')
        group.add_argument("--expert-hidden-dim", type=int, default=128,
                           help='hidden dimension for expert networks')
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
            "relu": nn.ReLU()
        }
        # auto-parse projector config strings to construct nn.Module
        for projector_name in projector_config:
            # for d in projector_config[loc]:
            if projector_config[projector_name]["enabled"]:
                self.projectors[projector_name] = nn.Sequential()

                structure = projector_config[projector_name]["structure"].split("-")
                for i in range(len(structure)):
                    if structure[i] not in ["relu"]:
                        coef = 1 if not len(structure[i][:-1]) else int(structure[i][:-1])
                        base_size = name_dict[structure[i][-1]]
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
        
                config.num_labels = getattr(self.args, 'num_labels', 2)  # Default to 2 for binary classification
                
                
                model = AutoModel.from_pretrained(
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
                        task_type = TaskType.FEATURE_EXTRACTION,
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
                    
                # For STS, wrap with STSModel
                model = STSModel(model)
                    
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
            
            
            # Load base BERT model for feature extraction
            bert_model = AutoModel.from_pretrained(
                "bert-base-uncased", 
                config=config, 
                device_map=None, 
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            
            # Get teacher hidden size
            teacher_hidden_size = 4096 # hidden dim của LLM2Vec teacher model
            
            # Create MoE model
            model = MoEDistilledBERT(
                bert_model=bert_model,
                teacher_hidden_size=teacher_hidden_size,
                num_experts=getattr(self.args, 'num_experts', 3),
                expert_hidden_dim=getattr(self.args, 'expert_hidden_dim', 128),
            )
            
            # For STS, wrap with STSMoEWrapper
            model = STSMoEWrapper(model)
            
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model from checkpoint...")

        if not os.path.exists(self.args.teacher_model_path):
            raise ValueError(f"Teacher model path does not exist: {self.args.teacher_model_path}")
        regressor_path = os.path.join(self.args.teacher_model_path, "regressor.pt")
        model_files = os.listdir(self.args.teacher_model_path)
        log_rank(f"Found files in teacher model directory: {model_files}")

        # normal loading
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

        base_model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        if hasattr(base_model.config, "pad_token_id"):
            base_model.config.pad_token_id = 2

        teacher_base_model = PeftModel.from_pretrained(
            base_model,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        )    

        teacher_base_model = teacher_base_model.merge_and_unload()

        teacher_base_model = PeftModel.from_pretrained(
            teacher_base_model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
        )
        teacher_base_model = teacher_base_model.merge_and_unload()

        def load_peft_model_with_remapped_keys(base_model, teacher_model_path):
            config_path = os.path.join(teacher_model_path, "adapter_config.json")
            if os.path.exists(config_path):
                from peft import PeftConfig
                peft_config = PeftConfig.from_pretrained(teacher_model_path)
                peft_model = PeftModel(base_model, peft_config)
            else:
                # If no config file, you'll need to manually create one as in the previous solution
                raise ValueError("No adapter_config.json found and direct loading failed")
            adap_path = os.path.join(teacher_model_path, "adapter_model.bin")
            # Remap keys to fix nesting and naming issues
            remapped_state_dict = {}
            checkpoint = torch.load(adap_path)

            for key, value in checkpoint.items():
                new_key = key.replace("base_model.model.base_model.model", "base_model.model")
                new_key = new_key.replace("lora_A.weight", "lora_A.default.weight")
                new_key = new_key.replace("lora_B.weight", "lora_B.default.weight")
                remapped_state_dict[new_key] = value
            
            # Load remapped state dictionary
            peft_model.load_state_dict(remapped_state_dict, strict=False)
            print("LoRA loaded")
            return peft_model

        teacher_base_model = load_peft_model_with_remapped_keys(
            teacher_base_model,
            self.args.teacher_model_path
        )

        teacher_model = STSModel(teacher_base_model)
        # Load regressor if available
        if os.path.exists(regressor_path):
            log_rank("Loading regressor weights")
            regressor_state_dict = torch.load(regressor_path, map_location="cpu")
            teacher_model.regressor.load_state_dict(regressor_state_dict)
        else:
            log_rank("No regressor.pt found, using initialized regressor")

        # Freeze the teacher model parameters
        for param in teacher_model.parameters():
            param.requires_grad = False

        log_rank("Teacher model loaded successfully")
        return teacher_model, tokenizer
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
    
