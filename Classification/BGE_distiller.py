import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,  
    AutoModelForSequenceClassification,
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
#token = os.getenv("HF_TOKEN")
#login(token=token)

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
                config = AutoConfig.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
                config.is_model_parallel = False

                # Load tokenizer
                tokenizer = self.load_tokenizer("BAAI/bge-m3")

                if hasattr(config, "n_embed"):
                    self.hidden_size = config.n_embed
                else:
                    self.hidden_size = config.hidden_size

                config.num_labels = self.args.num_labels

                # Load the model
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

                # For full fine-tuning, we don't need PEFT/LoRA
                if self.args.do_train:
                    # Make sure all parameters are trainable for full fine-tuning
                    for param in model.parameters():
                        param.requires_grad = True
                    
                    # Calculate and print trainable parameters info
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in model.parameters())
                    print(f"Trainable parameters: {trainable_params}/{all_params} ({trainable_params/all_params:.2%})")

            else:
                raise NotImplementedError
        else: #for BERT
            config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
            config.is_model_parallel = False
    
            # lấy tokenizer
            tokenizer = self.load_tokenizer("bert-base-uncased")
            
            if hasattr(config, "n_embed"):
                self.hidden_size = config.n_embed
            else:
                self.hidden_size = config.hidden_size
            config.num_labels = self.args.num_labels
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", 
                config=config, 
                device_map=None, 
                torch_dtype=self.dtype,
                trust_remote_code=True,)
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
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
        if hasattr(self, "projectors"):
            if self.args.projector_lr:
                pretrained_proj = self.args.pretrained_projector.split(",") if self.args.pretrained_projector is not None else []
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b not in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.projector_lr
                })
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.pretrained_projector_lr
                })
            else:
                optimizer.add_param_group({
                    "params": [p for b in self.projectors for p in self.projectors[b].parameters()],
                })
        return optimizer

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
