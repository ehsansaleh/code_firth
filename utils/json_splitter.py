import json
import copy
import numpy as np
from os.path import abspath, dirname

proj_dir = dirname(dirname(abspath(__file__)))


def main(args_configid, split_var, n_split):
    if '/' in args_configid:
        args_configid_split = args_configid.split('/')
        config_id = args_configid_split[-1]
        config_tree = '/'.join(args_configid_split[:-1])
    else:
        config_id = args_configid
        config_tree = ''

    cfg_path = f'{proj_dir}/configs/{config_tree}/{config_id}.json'
    print(f'  * Splitting {cfg_path} into {n_split} parts along the {split_var} variable.')
    with open(cfg_path) as f:
        config_dict = json.load(f)

    if split_var == 'seed':
        start_seed = config_dict['start_seed']
        num_seeds = config_dict['num_seeds']
        rng_seed_list = list(range(start_seed, start_seed+num_seeds))

        splitted_seed_list = np.array_split(rng_seed_list, n_split)
        for i, seed_split in enumerate(splitted_seed_list):
            if len(seed_split) < 1:
                print(f'Worker {i} has nothing to do! Are you okay with that?!')
                continue
            splitted_start_seed = int(seed_split[0])
            splitted_num_seeds = len(seed_split)

            splitted_cfgdict = copy.deepcopy(config_dict)
            splitted_cfgdict["start_seed"] = splitted_start_seed
            splitted_cfgdict["num_seeds"] = splitted_num_seeds

            splitted_json_path = f'{proj_dir}/configs/{config_tree}/{config_id}_part{i}.json'
            print(f'  *   --> Generating {splitted_json_path}.')
            with open(splitted_json_path, 'w') as f:
                json.dump(splitted_cfgdict, f, indent=4)
        print('  ' + '-' * 80)
    else:
        assert split_var.endswith('_list')
        split_var_list = config_dict[split_var]
        assert isinstance(split_var_list, list)

        splitted_var_list = np.array_split(split_var_list, n_split)

        for i, var_split in enumerate(splitted_var_list):
            if len(var_split) < 1:
                print(f'Worker {i} has nothing to do! Are you okay with that?!')
                continue

            splitted_cfgdict = copy.deepcopy(config_dict)
            splitted_cfgdict[split_var] = var_split.tolist()
            splitted_json_path = f'{proj_dir}/configs/{config_tree}/{config_id}_part{i}.json'
            print(f'  *   --> Generating {splitted_json_path}.')
            with open(splitted_json_path, 'w') as f:
                json.dump(splitted_cfgdict, f, indent=4)
        print('  ' + '-'*80)


if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        description = 'Splitting a json config into multiple parts'
        my_parser = argparse.ArgumentParser(description=description)
        my_parser.add_argument('-c', '--configid',
                               default='01_firth_1layer/firth_1layer',
                               type=str, required=True,
                               help='The json config id (e.g., if your json is located at' +
                                    ' "configs/01_firth_1layer/firth_1layer.json"' +
                                    ', then its config id is "01_firth_1layer/firth_1layer"')
        my_parser.add_argument('-v', '--variable',
                               default='firth_coeff_list',
                               type=str, required=True,
                               help='The variable along which the splitting happens.' +
                                    'It can be either "seed", or one of the "*_list" ' +
                                    'variable names in the json config files.')
        my_parser.add_argument('-n', '--n_split', default=8,
                               type=int, required=True,
                               help='The number of splitted output json files.')
        args_parser = my_parser.parse_args()
        args_configid = args_parser.configid
        split_var = args_parser.variable
        n_split = args_parser.n_split
    else:
        args_configid = '01_firth_1layer/firth_1layer'
        split_var = 'firth_coeff_list'
        n_split = 8

    main(args_configid, split_var, n_split)
