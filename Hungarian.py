# -*- coding: utf-8 -*-

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

class HungarianLoss(nn.Module):
    def __init__(self, cost_class=1, num_classes=8):
        super().__init__()
        
        self.cost_class = cost_class
        self.num_classes = num_classes
        
    '''
    @torch.no_grad()
    def forward2(self, predictions, targets):

        batch, num_queries = predictions.shape[0], predictions.shape[1]
        
        out_prob = predictions.softmax(dim=-1)
        
        tgt_ids  = targets.unsqueeze(1)
        tgt_ids = tgt_ids.repeat(1, num_queries, 1)

        instance_ids = torch.arange(batch).to(out_prob.device)
        instance_ids = instance_ids.unsqueeze(1).unsqueeze(1)
        instance_ids = instance_ids.repeat(1, num_queries, num_queries)

        query_ids = torch.arange(num_queries).to(out_prob.device)
        query_ids = query_ids.unsqueeze(0).unsqueeze(2)
        query_ids = query_ids.repeat(batch, 1, num_queries)

        cost_class = -out_prob[instance_ids, query_ids, tgt_ids]

        C = self.cost_class * cost_class
        indices = [linear_sum_assignment(c) for i, c in enumerate(C)]
        print(indices)
    '''
    
    @torch.no_grad()
    def BipartiteMatching(self, predictions, targets):
        #predictions.shape [batch, 11, 8]
        #targets.shape [batch, 11]
        
        batch, num_queries = predictions.shape[0], predictions.shape[1]
        
        out_prob = predictions.view(-1, predictions.shape[2])
        out_prob = out_prob.softmax(dim=-1)
        
        tgt_ids = targets.view(-1)
        
        cost_class = -out_prob[:, tgt_ids]
        
        C = self.cost_class * cost_class
        C = C.view(batch, num_queries, -1).cpu()
        
        sizes = [len(_) for _ in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, predictions, targets):
        
        indices = self.BipartiteMatching(predictions, targets)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(predictions.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=predictions.device)
        target_classes[idx] = target_classes_o
        
        predictions = predictions.view(-1, self.num_classes)
        target_classes = target_classes.view(-1)
        
        HungarianLoss = F.cross_entropy(predictions, target_classes)
        
        return HungarianLoss
        


hl = HungarianLoss()
predictions = torch.randn(10, 11, 8)
targets = torch.randint(high=8, size=(10, 11))

print(targets)
loss = hl(predictions, targets)
print(loss)

