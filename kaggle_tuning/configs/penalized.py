"""Fixed research candidates; C/alpha selected using pre-target season CV."""
from kaggle_tuning.penalized import KINDS, mix_config

CONFIGS = {
    f'{kind}_{replacement}_{weight}': mix_config(kind, weight, replacement)
    for kind in KINDS for replacement in ['all','logit','tree'] for weight in [.25,.5,1.]
}

CANDIDATE = mix_config('lasso_margin', .5, 'logit')
CANDIDATE_EXT = mix_config('lasso_margin', .5, 'logit', external=True)
CONFIGS.update(candidate=CANDIDATE, candidate_ext=CANDIDATE_EXT)
