import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import cosine_distance

class SetDecoder(nn.Module):
    def __init__(self, config, num_iterations, num_generated_triples, num_classes, mesh_lr, n_mesh_iters, slot_dropout):
        super().__init__()
        self.slot_attention = SlotAttention(
            in_features=config.hidden_size,
            num_iterations=num_iterations,
            num_slots=num_generated_triples,
            slot_size=config.hidden_size,
            mlp_hidden_size=config.hidden_size*2,
            mesh_lr=mesh_lr,
            n_mesh_iters=n_mesh_iters,
            slot_dropout=slot_dropout,
        )

        self.decoder2class = nn.Linear(config.hidden_size, num_classes)
        
        self.head_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        
        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        
        hidden_states = self.slot_attention(encoder_hidden_states)

        class_logits = self.decoder2class(hidden_states)
        
        head_start_logits = self.head_start_metric_3(torch.tanh(
            self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()
        head_end_logits = self.head_end_metric_3(torch.tanh(
            self.head_end_metric_1(hidden_states).unsqueeze(2) + self.head_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        tail_start_logits = self.tail_start_metric_3(torch.tanh(
            self.tail_start_metric_1(hidden_states).unsqueeze(2) + self.tail_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()
        tail_end_logits = self.tail_end_metric_3(torch.tanh(
            self.tail_end_metric_1(hidden_states).unsqueeze(2) + self.tail_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits


class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, mesh_lr, n_mesh_iters, slot_dropout, epsilon=1e-8):
        super().__init__()
        # Slot Attention Module adapted from: https://github.com/davzha/MESH
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(p=slot_dropout),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        
        # For Optimal Transport Conversion
        self.mlp_slot_marginals = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(p=slot_dropout),
            nn.Linear(self.mlp_hidden_size, 1, bias=False),)
        
        self.mlp_input_marginals = nn.Sequential(
            nn.Linear(self.in_features, self.mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(p=slot_dropout),
            nn.Linear(self.mlp_hidden_size, 1, bias=False),)
        
        self.mesh_lr = mesh_lr
        self.n_mesh_iters= n_mesh_iters


    def forward(self, inputs):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        
        a = self.num_slots * self.mlp_input_marginals(inputs).squeeze(2).softmax(dim=1)
        noise = torch.randn(batch_size, num_inputs, self.num_slots, device=slots.device)
        sh_u = sh_v = None
        
        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots = slots.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            b = self.num_slots * self.mlp_slot_marginals(slots).squeeze(2).softmax(dim=1)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].

            # l2_distance, cosine_distance, dot_prod
            cost = self.num_slots * cosine_distance(k, q)
            
            # Minimize Entropy of Sinkhorn
            cost, sh_u, sh_v = minimize_entropy_of_sinkhorn(
                cost, a, b, noise=noise, mesh_lr=self.mesh_lr, n_mesh_iters=self.n_mesh_iters
                )
            
            attn, *_ = sinkhorn(C=cost, a=a, b=b, u=sh_u, v=sh_v)
            
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


@torch.enable_grad()
def minimize_entropy_of_sinkhorn(
    C_0, a, b, noise=None, mesh_lr=1, n_mesh_iters=4
):
    C_t = C_0 + 0.001 * noise
    C_t.requires_grad_(True)

    u = None
    v = None
    for i in range(n_mesh_iters):
        attn, u, v = sinkhorn(C_t, a, b, u=u, v=v)
        entropy = torch.mean(torch.special.entr(attn.clamp(min=1e-20, max=1)), dim=[1, 2]).sum()
        (grad,) = torch.autograd.grad(entropy, C_t, retain_graph=True)
        grad = F.normalize(grad + 1e-20, dim=[1, 2])
        C_t = C_t - mesh_lr * grad

    return C_t, u, v

def sinkhorn(C, a, b, u, v, n_sh_iters=5):
    
    p = -C
    # clamp to avoid - inf
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    
    if u is None:
        u = torch.zeros_like(a)
    if v is None:
        v = torch.zeros_like(b)
    
    for _ in range(n_sh_iters):
        u = log_a - torch.logsumexp(p + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(p + u.unsqueeze(2), dim=1)
        
    logT = p + u.unsqueeze(2) + v.unsqueeze(1)
    
    return logT.exp(), u, v