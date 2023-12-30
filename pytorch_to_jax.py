from typing import Dict

import flax

def convert_from_pytorch(pt_state: Dict, params_flatten):
    """_summary_

    Args:
        pt_state (Dict): _description_
        params_flatten (_type_): _description_

    Returns:
        _type_: _description_
    """
    jax_state = dict(pt_state)

    for key, tensor in pt_state.items():
        tensor = tensor.cpu().numpy()

        if "embedding.weight" in key:
            del jax_state[key]
            key = key.replace("embedding.weight", "embedding.embedding")
            jax_state[key] = tensor
        
        if "layers." in key:
            del jax_state[key]
            key = key.replace("layers.", "layers_")
            jax_state[key] = tensor

        if "proj.weight" in key:
            del jax_state[key]
            key = key.replace("proj.weight", "proj.kernel")
            jax_state[key] = tensor

        if "conv1d.weight" in key:
            del jax_state[key]
            key = key.replace("conv1d.weight", "conv1d.kernel")
            jax_state[key] = tensor
        
        if "lm_head" in key:
            del jax_state[key]

    jax_state_transposed = {}

    for key in params_flatten.keys():
        if params_flatten[key].shape != jax_state[key].shape:
            jax_state_transposed[key] = jax_state[key].T
        else:
            jax_state_transposed[key] = jax_state[key]

        if params_flatten[key].dtype != jax_state[key].dtype:
            jax_state_transposed[key] = jax_state_transposed[key].numpy()
        else:
            jax_state_transposed[key] = jax_state_transposed[key]

        assert params_flatten[key].shape == jax_state_transposed[key].shape, f'The shape of {key} is not the same with param shape {params_flatten[key].shape} and jax_state shape {jax_state_transposed[key].shape}'
        assert params_flatten[key].dtype == jax_state_transposed[key].dtype, f'The dtype of {key} is not the same with param dtype {params_flatten[key].dtype} and jax_state dtype {jax_state_transposed[key].dtype}'

    params = flax.traverse_util.unflatten_dict(jax_state_transposed, sep=".")

    return params
