"""
Run EE with increasing number of repetitions
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

template_cfg_path = sys.argv[2]
# read template
with open(template_cfg_path, 'r') as f:
    lines_template_cfg = f.readlines()

for nr_morris_repetitions in np.arange(62, 64):

    modified_cfg_lines = copy.deepcopy(lines_template_cfg)

    # modify template
    anchor = '$nr_morris_repetitions$'
    for idx, line in enumerate(lines_template_cfg):
        if anchor in line:
            modified_cfg_lines[idx] = line.replace(anchor, str(nr_morris_repetitions))

    # save template
    modified_cfg_path = template_cfg_path.replace('template', str(nr_morris_repetitions))
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

    # result_analysis_EE(UQResultsAnalysis)
