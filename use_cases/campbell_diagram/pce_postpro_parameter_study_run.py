"""
This script was used to postprocess the parameter study on different post-processing settings for the PCE Analysis
"""


import sys
import os
import numpy as np
from wtuq_framework.uq_framework import UQFramework
from run_analysis import CampbellDiagramModel
from extra_uq_results_analysis import result_analysis_EE
import copy

"""
Program start
"""
try:
    use_case_configspec = os.path.join(os.path.dirname(__file__), 'config', 'use_case_config.spec')
except IOError:
    print('use_case_config.spec has to be defined for the use case')

sys.argv = sys.argv + \
           ['-c'] + \
           ['./config/hawcstab2/pce_postpro_parameter_study/config_hawcstab2_IEA_15MW_pce_template.txt'] + \
           ['-r'] + \
           ['./results/IEA_15MW_pce_first_run/*'] + \
           ['-f']

template_cfg_path = sys.argv[2]
# read template
with open(template_cfg_path, 'r') as f:
    lines_template_cfg = f.readlines()

for mac_op0 in [0.7, 0.8, 0.85, 0.9]:
    for mac_hs2 in [['withdamp', 0.5], ['withdamp', 0.6], ['withdamp', 0.7], ['nodamp', 0.7], ['nodamp', 0.8], ['nodamp', 0.9]]:
        for polynomial_order in [2, 3, 4]:
            for n_collocation_nodes in [4762]:

                modified_cfg_lines = copy.deepcopy(lines_template_cfg)

                run_name = '{}_{}_{}_{}_{}'.format(mac_op0, mac_hs2[1], mac_hs2[0], polynomial_order, n_collocation_nodes)

                # modify template
                anchor_mac_op0 = '$minimum_MAC_mode_picking$'
                anchor_mac_hs2_0 = '$with_damp$'
                anchor_mac_hs2_1 = '$minimum_MAC_mode_tracking$'
                anchor_polynomial_order = '$polynomial_order$'
                anchor_n_collocation_nodes = '$nr_collocation_nodes$'
                anchor_run_name = '$run_name$'

                for idx, line in enumerate(lines_template_cfg):
                    if anchor_mac_op0 in line:
                        modified_cfg_lines[idx] = line.replace(anchor_mac_op0, str(mac_op0))
                    if anchor_mac_hs2_0 in line:
                        if mac_hs2[0] == 'withdamp':
                            modified_cfg_lines[idx] = line.replace(anchor_mac_hs2_0, 'True')
                        else:
                            modified_cfg_lines[idx] = line.replace(anchor_mac_hs2_0, 'False')
                    if anchor_mac_hs2_1 in line:
                        modified_cfg_lines[idx] = line.replace(anchor_mac_hs2_1, str(mac_hs2[1]))
                    if anchor_polynomial_order in line:
                        modified_cfg_lines[idx] = line.replace(anchor_polynomial_order, str(polynomial_order))
                    if anchor_n_collocation_nodes in line:
                        modified_cfg_lines[idx] = line.replace(anchor_n_collocation_nodes, str(n_collocation_nodes))
                    if anchor_run_name in line:
                        modified_cfg_lines[idx] = line.replace(anchor_run_name, str(run_name))


                # save template
                modified_cfg_path = template_cfg_path.replace('template', str(run_name))
                with open(modified_cfg_path, 'w') as f:
                    f.writelines(modified_cfg_lines)

                sys.argv[2] = modified_cfg_path

                framework = UQFramework(use_case_configspec)

                model = CampbellDiagramModel(tool_config=framework.config['use_case']['tool'],
                                             preprocessor_config=framework.config['use_case']['preprocessor'],
                                             model_inputs=framework.give_standard_model_inputs())

                features = [model.edge_mode_one, model.edge_mode_two, model.edge_mode_three, model.edge_mode_four,
                            model.edge_mode_five, model.edge_mode_six, model.edge_mode_seven]
                UQResultsAnalysis = framework.main(model, features=features, return_postprocessor=True)

                # result_analysis_pce(UQResultsAnalysis)
