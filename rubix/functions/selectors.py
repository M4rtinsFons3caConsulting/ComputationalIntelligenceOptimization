import torch
from typing import List, Any

def _select_one(
    fitnesses: torch.Tensor,
    valid: torch.Tensor,
    selection_type: str,
    size: int
) -> int:
    
    """
    Selects one individual based on the tournament type from a valid subset.
    """

    sampled_list = valid[torch.randperm(len(valid))[:size]].tolist()
    valid_list = valid.tolist()

    if selection_type == 'rank':
        valid_fitnesses = fitnesses[valid]
        _, sorted_idx = torch.sort(valid_fitnesses)
        ranks = torch.zeros_like(valid_fitnesses, dtype=torch.long)
        ranks[sorted_idx] = torch.arange(len(valid_fitnesses))
        competitors = sorted(
            sampled_list,
            key=lambda i: ranks[valid_list.index(i)].item()
        )
        return competitors[0]

    elif selection_type == 'fitness':
        competitors = sorted(sampled_list, key=lambda i: fitnesses[i].item())
        return competitors[0]

    else:  # probabilistic
        comp_fitnesses = fitnesses[torch.tensor(sampled_list)]
        probs = torch.softmax(-comp_fitnesses, dim=0)
        selected_idx = torch.multinomial(probs, 1).item()

        return sampled_list[selected_idx]

def tournament(
    fitnesses: torch.Tensor, 
    **kwargs: Any
) -> List[int]:
    
    """
    Selects multiple individuals using tournament selection ignoring invalid fitness (torch.inf).
    """

    n = kwargs['n']
    tournament_type = kwargs['selection_method']
    size = kwargs['vs_size']

    valid = (fitnesses != float('inf')).nonzero(as_tuple=True)[0]

    return [_select_one(fitnesses, valid, tournament_type, size) for _ in range(n)]

def poison(
    fitnesses: torch.Tensor,
    **kwargs: Any
) -> List[int]:
    
    """
    Simulates a lethal environment by probabilistically eliminating individuals based on fitness.
    """

    threshold = kwargs['poison_percentile']
    sharpness = kwargs.get('poison_sharpness', 1.0)

    diff = fitnesses - threshold
    death_probs = 1 / (1 + torch.exp(-sharpness * diff))

    survive_mask = torch.rand_like(death_probs) > death_probs
    survivors = survive_mask.nonzero(as_tuple=True)[0]

    return [_select_one(fitnesses, survivors, selection_type="probabilistic", survivors.size()) for _ in range(n)]

def identity(
    fitnesses: torch.Tensor
) -> List[int]:
    
    """
    Identity operator used for testing or bypassing selection logic.
    """

    return torch.arange(fitnesses.size(0)).tolist()

def apply_selector(
    *args: Any, 
    **kwargs: Any
) -> List[int]:
    
    """
    Applies the provided selection function to given arguments.
    """
    return SELECTORS[kwargs['select_type']](*args, **kwargs)

SELECTORS = {
    "poison": poison,
    "tournament": tournament,
    "identity": identity
}
