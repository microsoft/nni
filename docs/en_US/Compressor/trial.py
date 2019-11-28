import nni
from nni.compression.torch import *
params = nni.get_parameters()
conv0_sparsity = params['prune_method']['conv0_sparsity']
conv1_sparsity = params['prune_method']['conv1_sparsity']
# these raw sparsity should be scaled if you need total sparsity constrained
config_list_level = [{ 'sparsity': conv0_sparsity, 'op_name': 'conv0' },
                     { 'sparsity': conv1_sparsity, 'op_name': 'conv1' }]
config_list_agp = [{'initial_sparsity': 0, 'final_sparsity': conv0_sparsity,
                    'start_epoch': 0, 'end_epoch': 3,
                    'frequency': 1,'op_name': 'conv0' },
                   {'initial_sparsity': 0, 'final_sparsity': conv1_sparsity,
                    'start_epoch': 0, 'end_epoch': 3,
                    'frequency': 1,'op_name': 'conv1' },]
PRUNERS = {'level':LevelPruner(model, config_list_level), 'agp':AGP_Pruner(model, config_list_agp)}
pruner = PRUNERS(params['prune_method']['_name'])
pruner.compress()
# fine tuning
acc = evaluate(model) # evaluation
nni.report_final_results(acc)