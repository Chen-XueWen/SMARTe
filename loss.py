import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def SetLoss(outputs, targets, num_classes):
    indices = HungarianMatcher(outputs, targets)
    entity_loss = get_entity_loss(outputs, targets, indices)
    rel_loss = get_relation_loss(outputs, targets, indices, num_classes)
    set_loss = entity_loss + rel_loss
    return set_loss

def HungarianMatcher(outputs, targets):
    
    bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
    pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
    gold_rel = torch.cat([v["relation"] for v in targets])
    
    pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)
    pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
    pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
    pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)
    
    gold_head_start = torch.cat([v["head_start_index"] for v in targets])
    gold_head_end = torch.cat([v["head_end_index"] for v in targets])
    gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
    gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])
    
    cost = -pred_rel[:, gold_rel] - (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
    cost = cost.view(bsz, num_generated_triples, -1).cpu().detach()
    num_gold_triples = [len(v["relation"]) for v in targets]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]

    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def get_entity_loss(outputs, targets, indices):
    
    idx = get_src_permutation_idx(indices)
    selected_pred_head_start = outputs["head_start_logits"][idx]
    selected_pred_head_end = outputs["head_end_logits"][idx]
    selected_pred_tail_start = outputs["tail_start_logits"][idx]
    selected_pred_tail_end = outputs["tail_end_logits"][idx]
    
    target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
    target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])
    target_tail_start = torch.cat([t["tail_start_index"][i] for t, (_, i) in zip(targets, indices)])
    target_tail_end = torch.cat([t["tail_end_index"][i] for t, (_, i) in zip(targets, indices)])
    
    head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start.to(selected_pred_head_start.device))
    head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end.to(selected_pred_head_end.device))
    tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start.to(selected_pred_tail_start.device))
    tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end.to(selected_pred_tail_end.device))
    
    entity_loss = (head_start_loss + head_end_loss) + (tail_start_loss + tail_end_loss)
    
    return entity_loss

def get_relation_loss(outputs, targets, indices, num_classes):
    
    src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel+1]
    idx = get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)]).to(src_logits.device)
    # Last index is num_classes-1
    target_classes = torch.full(src_logits.shape[:2], num_classes-1,
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o
    
    rel_weight = torch.ones(num_classes).to(src_logits.device)
    rel_weight[-1] = 0.25 # For NA Coefficient
    
    rel_loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=rel_weight)
    
    return rel_loss

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
