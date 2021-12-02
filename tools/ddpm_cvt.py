from copy import deepcopy


def convert_out_blocks(state_dict):
    cvt_state = dict()
    for k, v in state_dict.items():
        k_list = k.split('.')
        if k_list[2] == 'in_layers':
            k_list[2] = 'conv_1'
        if k_list[2] == 'emb_layers':
            # '1.0.emb_layers.1.weight' -> '1.0.norm_with_embedding.embedding_layer.1.weight'  # noqa
            k_list[2] = 'norm_with_embedding.embedding_layer'
        if k_list[2] == 'out_layers':
            # '1.0.out_layers.0.weight' -> '1.0.norm_with_embedding.norm.weight'  # noqa
            if k_list[3] == '0':
                k_list[2] = 'norm_with_embedding'
                k_list[3] = 'norm'
            if k_list[3] == '3':
                # '1.0.out_layers.3.weight' -> '1.0.conv_2.2.weight'
                k_list[2] = 'conv_2'
                k_list[3] = '2'

        if k_list[2] == 'skip_connection':
            k_list[2] = 'shortcut'
        if k_list[2] == 'proj_out':
            k_list[2] = 'proj'

        cvt_state['.'.join(k_list)] = v
    return cvt_state


def convert_mid_blocks(state_dict):
    cvt_state_dict = dict()
    for k, v in state_dict.items():
        k_list = k.split('.')
        if k_list[1] == 'in_layers':
            k_list[1] = 'conv_1'
        if k_list[1] == 'emb_layers':
            # '0.emb_layers.1.weight' -> '0.norm_with_embedding.embedding_layer.1.weight'  # noqa
            k_list[1] = 'norm_with_embedding.embedding_layer'
        if k_list[1] == 'out_layers':
            # '0.out_layers.0.weight' -> '0.norm_with_embedding.norm.weight'
            if k_list[2] == '0':
                k_list[1] = 'norm_with_embedding'
                k_list[2] = 'norm'
            # '0.out_layers.3.weight' -> '0.conv_2.2.weight'
            if k_list[2] == '3':
                k_list[1] = 'conv_2'
                k_list[2] = '2'

        if k_list[0] == '1':
            if k_list[1] == 'proj_out':
                k_list[1] = 'proj'

        cvt_state_dict['.'.join(k_list)] = v
    return cvt_state_dict


def convert_in_blocks_list(key):
    k_list = key.split('.')

    # handle attention
    if k_list[1] == '1':
        # '7.1.proj_out.weight' -> '7.1.proj.weight'
        if k_list[2] == 'proj_out':
            k_list[2] = 'proj'

    if k_list[2] == 'in_layers':
        # '1.0.in_layers.0.weight' -> '1.0.conv_1.0.weight'
        # '1.0.in_layers.2.weight' -> '1.0.conv_1.2.weight'
        k_list[2] = 'conv_1'
    if k_list[2] == 'emb_layers':
        # '1.0.emb_layers.1.weight' -> '1.0.norm_with_embedding.embedding_layer.1.weight'  # noqa
        k_list[2] = 'norm_with_embedding.embedding_layer'
    if k_list[2] == 'out_layers':
        # '1.0.out_layers.0.weight' -> '1.0.norm_with_embedding.norm.weight'
        if k_list[3] == '0':
            k_list[2] = 'norm_with_embedding'
            k_list[3] = 'norm'
        if k_list[3] == '3':
            # '1.0.out_layers.3.weight' -> '1.0.conv_2.2.weight'
            k_list[2] = 'conv_2'
            k_list[3] = '2'

    if k_list[2] == 'op':
        k_list[2] = 'downsample'
    if k_list[2] == 'skip_connection':
        k_list[2] = 'shortcut'

    return '.'.join(k_list)


def convert_in_blocks(state_dict):
    cvt_state = dict()
    for k, v in state_dict.items():
        if k[0] != '0':
            k = convert_in_blocks_list(k)
        cvt_state[k] = v
    return cvt_state


def convert_embedding_layer(state_dict):
    cvt_dict = dict()
    for k, v in state_dict.items():
        k = 'blocks.' + k
        cvt_dict[k] = v
    return cvt_dict


def convert_final_layer(state_dict):
    cvt_dict = dict()
    for k, v in state_dict.items():
        k_list = k.split('.')
        if k[0] == '0':
            k_list[0] = 'gn'
        if k[0] == '2':
            k_list[0] = 'conv'
        cvt_dict['.'.join(k_list)] = v
    return cvt_dict


def convert_whole_model(denoising_official):
    state_mmgen = dict()
    module_list_off = [
        denoising_official.time_embed, denoising_official.input_blocks,
        denoising_official.middle_block, denoising_official.output_blocks,
        denoising_official.out
    ]
    mmgen_module_prefix_list = [
        'time_embedding', 'in_blocks', 'mid_blocks', 'out_blocks', 'out'
    ]
    convert_fn_list = [
        convert_embedding_layer, convert_in_blocks, convert_mid_blocks,
        convert_out_blocks, convert_final_layer
    ]
    for module_off, cvt_fn, prefix in zip(module_list_off, convert_fn_list,
                                          mmgen_module_prefix_list):
        state_off = module_off.state_dict()
        state_cvted = cvt_fn(deepcopy(state_off))
        for k, v in state_cvted.items():
            state_mmgen[f'{prefix}.{k}'] = v

    return state_mmgen


def convert_whole_state_dict(state_dict_official):
    prefix_fn_mapping = dict(
        time_embed=[convert_embedding_layer, 'time_embedding'],
        input_blocks=[convert_in_blocks, 'in_blocks'],
        middle_block=[convert_mid_blocks, 'mid_blocks'],
        output_blocks=[convert_out_blocks, 'out_blocks'],
        out=[convert_final_layer, 'out'])

    official_state_dict = {
        prefix: dict()
        for prefix in prefix_fn_mapping.keys()
    }
    for k, v in state_dict_official.items():
        for prefix in prefix_fn_mapping.keys():
            # if prefix in k:
            if k.startswith(prefix):
                # avoid cvt module start with out_blocks to out
                if prefix == 'out' and k.startswith('output_blocks'):
                    continue
                new_k = '.'.join(k.split('.')[1:])
                official_state_dict[prefix][new_k] = v

    mmgen_state_dict = dict()
    for prefix, (cvt_fn, mmgen_prefix) in prefix_fn_mapping.items():
        state_dict_cvt = cvt_fn(official_state_dict[prefix])
        for k, v in state_dict_cvt.items():
            mmgen_state_dict[f'{mmgen_prefix}.{k}'] = v

    return mmgen_state_dict


def convert_state_dict_with_offRes(state_dict):
    cvt_dict = dict()
    for k, v in state_dict.items():
        k_split = k.split('.')
        if k_split[0] == 'time_embed':
            k_split[0] = 'time_embedding.blocks'
        elif k_split[0] == 'input_blocks':
            k_split[0] = 'in_blocks'
        elif k_split[0] == 'middle_block':
            k_split[0] = 'mid_blocks'
        elif k_split[0] == 'output_blocks':
            k_split[0] = 'out_blocks'
        elif k_split[0] == 'out':
            if k_split[1] == '0':
                k_split[1] = 'gn'
            elif k_split[1] == '2':
                k_split[1] = 'conv'
        if len(k_split) >= 4:
            if k_split[3] == 'op':
                k_split[3] = 'downsample'
            if k_split[3] == 'proj_out':
                k_split[3] = 'proj'
            if k_split[2] == 'proj_out':
                k_split[2] = 'proj'

        cvt_dict['.'.join(k_split)] = v.clone()
    return cvt_dict
