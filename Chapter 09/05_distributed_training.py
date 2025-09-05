import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertForSequenceClassification

# Initialize DDP
model = BertForSequenceClassification.from_pretrained("bert-base-uncased") #It can be any transformer model here YourTransformerModel() 
model = DDP(model, device_ids=[0, 1])  # Distribute across GPUs 0 and 1

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap model for memory-efficient training
model = FSDP(BertForSequenceClassification.from_pretrained("bert-base-uncased"))

# Enable mixed precision
model.half()
model.to("cuda")
