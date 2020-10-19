# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import numpy as np


def get_json_content(file_path):
    """
    Load json file content

    Parameters
    ----------
    file_path:
        path to the file

    Raises
    ------
    TypeError
        Error with the file path
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except TypeError as err:
        print('Error: ', err)
        return None


def generate_pcs(nni_search_space_content):
    """
    Generate the Parameter Configuration Space (PCS) which defines the
    legal ranges of the parameters to be optimized and their default values.
    Generally, the format is:
    # parameter_name categorical {value_1, ..., value_N} [default value]
    # parameter_name ordinal {value_1, ..., value_N} [default value]
    # parameter_name integer [min_value, max_value] [default value]
    # parameter_name integer [min_value, max_value] [default value] log
    # parameter_name real [min_value, max_value] [default value]
    # parameter_name real [min_value, max_value] [default value] log
    Reference: https://automl.github.io/SMAC3/stable/options.html

    Parameters
    ----------
    nni_search_space_content: search_space
        The search space in this experiment in nni

    Returns
    -------
    Parameter Configuration Space (PCS)
        the legal ranges of the parameters to be optimized and their default values

    Raises
    ------
    RuntimeError
        unsupported type or value error or incorrect search space
    """
    categorical_dict = {}
    search_space = nni_search_space_content

    def dump_categorical(fd, key, categories):
        choice_len = len(categories)
        if key in categorical_dict:
            raise RuntimeError(
                '%s has already existed, please make sure search space has no duplicate key.' % key)
        categorical_dict[key] = search_space[key]['_value']
        fd.write('%s categorical {%s} [0]\n' % (key, ','.join(map(str, range(choice_len)))))

    with open('param_config_space.pcs', 'w') as pcs_fd:
        if isinstance(search_space, dict):
            for key in search_space.keys():
                if isinstance(search_space[key], dict):
                    try:
                        if search_space[key]['_type'] == 'choice':
                            dump_categorical(pcs_fd, key, search_space[key]['_value'])
                        elif search_space[key]['_type'] == 'randint':
                            lower, upper = search_space[key]['_value']
                            if lower + 1 == upper:
                                dump_categorical(pcs_fd, key, [lower])
                            else:
                                pcs_fd.write('%s integer [%d, %d] [%d]\n' % (key, lower, upper - 1, lower))
                        elif search_space[key]['_type'] == 'uniform':
                            low, high = search_space[key]['_value']
                            if low == high:
                                dump_categorical(pcs_fd, key, [low])
                            else:
                                pcs_fd.write('%s real [%s, %s] [%s]\n' % (key, low, high, low))
                        elif search_space[key]['_type'] == 'loguniform':
                            # use np.round here to ensure that the rounded default value is in the range,
                            # which will be rounded in configure_space package
                            low, high = list(np.round(np.log(search_space[key]['_value']), 10))
                            if low == high:
                                dump_categorical(pcs_fd, key, [search_space[key]['_value'][0]])
                            else:
                                pcs_fd.write('%s real [%s, %s] [%s]\n' % (key, low, high, low))
                        elif search_space[key]['_type'] == 'quniform':
                            low, high, q = search_space[key]['_value'][0:3]
                            vals = np.clip(np.arange(np.round(low / q), np.round(high / q) + 1) * q, low, high).tolist()
                            pcs_fd.write('%s ordinal {%s} [%s]\n' % (
                                key,
                                json.dumps(vals)[1:-1],
                                json.dumps(vals[0])))
                        else:
                            raise RuntimeError('unsupported _type %s' % search_space[key]['_type'])
                    except:
                        raise RuntimeError('_type or _value error.')
        else:
            raise RuntimeError('incorrect search space.')
        return categorical_dict
    return None


def generate_scenario(ss_content):
    """
    Generate the scenario. The scenario-object (smac.scenario.scenario.Scenario) is used to configure SMAC and
    can be constructed either by providing an actual scenario-object, or by specifing the options in a scenario file.
    Reference: https://automl.github.io/SMAC3/stable/options.html
    The format of the scenario file is one option per line:
    OPTION1 = VALUE1
    OPTION2 = VALUE2
    ...
    Parameters
    ----------
    abort_on_first_run_crash: bool
        If true, SMAC will abort if the first run of the target algorithm crashes. Default: True,
        because trials reported to nni tuner would always in success state
    algo: function
        Specifies the target algorithm call that SMAC will optimize. Interpreted as a bash-command.
        Not required by tuner, but required by nni's training service for running trials
    always_race_default:
        Race new incumbents always against default configuration
    cost_for_crash:
        Defines the cost-value for crashed runs on scenarios with quality as run-obj. Default: 2147483647.0.
        Trials reported to nni tuner would always in success state
    cutoff_time:
        Maximum runtime, after which the target algorithm is cancelled. `Required if *run_obj* is runtime`
    deterministic: bool
        If true, the optimization process will be repeatable.
    execdir:
        Specifies the path to the execution-directory. Default: .
        Trials are executed by nni's training service
    feature_file:
        Specifies the file with the instance-features.
        No features specified or feature file is not supported
    initial_incumbent:
        DEFAULT is the default from the PCS. Default: DEFAULT. Must be from: [‘DEFAULT’, ‘RANDOM’].
    input_psmac_dirs:
        For parallel SMAC, multiple output-directories are used.
        Parallelism is supported by nni
    instance_file:
        Specifies the file with the training-instances. Not supported
    intensification_percentage:
        The fraction of time to be used on intensification (versus choice of next Configurations). Default: 0.5.
        Not supported, trials are controlled by nni's training service and kill be assessor
    maxR: int
        Maximum number of calls per configuration. Default: 2000.
    memory_limit:
        Maximum available memory the target algorithm can occupy before being cancelled.
    minR: int
        Minimum number of calls per configuration. Default: 1.
    output_dir:
        Specifies the output-directory for all emerging files, such as logging and results.
        Default: smac3-output_2018-01-22_15:05:56_807070.
    overall_obj:
    	PARX, where X is an integer defining the penalty imposed on timeouts (i.e. runtimes that exceed the cutoff-time).
        Timeout is not supported
    paramfile:
        Specifies the path to the PCS-file.
    run_obj:
        Defines what metric to optimize. When optimizing runtime, cutoff_time is required as well.
        Must be from: [‘runtime’, ‘quality’].
    runcount_limit: int
        Maximum number of algorithm-calls during optimization. Default: inf.
        Use default because this is controlled by nni
    shared_model:
        Whether to run SMAC in parallel mode. Parallelism is supported by nni
    test_instance_file:
        Specifies the file with the test-instances. Instance is not supported
    tuner-timeout:
        Maximum amount of CPU-time used for optimization. Not supported
    wallclock_limit: int
        Maximum amount of wallclock-time used for optimization. Default: inf.
        Use default because this is controlled by nni

    Returns
    -------
    Scenario:
        The scenario-object (smac.scenario.scenario.Scenario) is used to configure SMAC and can be constructed
        either by providing an actual scenario-object, or by specifing the options in a scenario file
    """
    with open('scenario.txt', 'w') as sce_fd:
        sce_fd.write('deterministic = 0\n')
        # sce_fd.write('output_dir = \n')
        sce_fd.write('paramfile = param_config_space.pcs\n')
        sce_fd.write('run_obj = quality\n')

    return generate_pcs(ss_content)


if __name__ == '__main__':
    generate_scenario('search_space.json')
