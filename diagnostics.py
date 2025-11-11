import torch
import torch.nn.functional as F

class DormantNeuronMonitor:
    def __init__(self, linear1_modules, threshold=1e-3):
        self.threshold = threshold
        self.max_act = {}     # layer_idx -> [H]
        self.handles = []
        for li, mod in enumerate(linear1_modules):
            self.handles.append(mod.register_forward_hook(self._hook_factory(li)))

    def _hook_factory(self, li):
        def hook(module, inp, out):
            # out shape: [B, R, C, H]
            with torch.no_grad():
                act = F.gelu(out.detach())
                act = act.reshape(-1, act.shape[-1])         # [T, H]
                m = act.abs().amax(dim=0)                    # [H]
                if li not in self.max_act:
                    self.max_act[li] = m
                else:
                    self.max_act[li] = torch.maximum(self.max_act[li], m)
        return hook

    def dormant_ratios(self):
        ratios = {}
        for li, m in self.max_act.items():
            ratios[li] = float((m <= self.threshold).float().mean().item())
        return ratios

    def reset(self):
        self.max_act.clear()

    def close(self):
        for h in self.handles: h.remove()
        self.handles = []

class FeatureEmbeddingTap:
    def __init__(self, linear_layer):
        self.last = None
        self.handle = linear_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out: (B, R, C-1, E)
        self.last = out.detach()

    def close(self):
        if self.handle: self.handle.remove(); self.handle = None


def effective_rank_from_embeddings(emb_tensor, max_tokens=50000, eps=1e-12):
    """
    Entropy-based effective rank: exp( -sum_i p_i log p_i ) with p_i = s_i^2 / sum s_i^2
    where s_i are singular values of centered embeddings.
    emb_tensor: (B, R, C-1, E)
    """
    if emb_tensor is None:
        return None
    X = emb_tensor.reshape(-1, emb_tensor.shape[-1])  # [T, E]
    if X.shape[0] > max_tokens:
        idx = torch.randperm(X.shape[0], device=X.device)[:max_tokens]
        X = X.index_select(0, idx)
    X = X - X.mean(dim=0, keepdim=True)
    # compute SVD (on CPU if memory tight)
    use_cpu = X.device.type == "cuda"
    if use_cpu:
        Xc = X.float().cpu()
        S = torch.linalg.svdvals(Xc)
    else:
        S = torch.linalg.svdvals(X.float())
    s2 = S**2
    p = s2 / (s2.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    eff_rank = torch.exp(H).item()
    return eff_rank

def flatten_grads(model, eps=1e-12):
    g = []
    for p in model.parameters():
        if p.grad is not None:
            g.append(p.grad.detach().reshape(-1))
    if not g:
        return None
    g = torch.cat(g)
    n = torch.linalg.norm(g)
    if n < eps:
        return None
    return (g / n).cpu()  # normalize for cosine, move to CPU for safety
