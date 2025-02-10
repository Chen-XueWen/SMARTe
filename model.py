import torch.nn as nn
from models.seq_encoder import SeqEncoder
from models.set_decoder import SetDecoder

class SMARTEModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.decoder = SetDecoder(config,
                                  num_iterations=args.num_iterations,
                                  num_generated_triples=args.num_generated_triples, 
                                  num_classes=args.num_classes,
                                  mesh_lr=args.mesh_lr,
                                  n_mesh_iters=args.n_mesh_iters,
                                  slot_dropout=args.slot_dropout)
        
    def forward(self, input_ids, attention_mask):
        
        last_hidden_state = self.encoder(input_ids, attention_mask)
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)# [bsz, num_generated_triples, seq_len]
        outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits, 'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits, 'tail_end_logits': tail_end_logits}
        
        return outputs
    
