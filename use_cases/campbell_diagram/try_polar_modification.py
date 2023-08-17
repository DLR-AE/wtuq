from tool_interfaces.bladed_lin_interface import BladedLinModel
import numpy as np

# config = {'template_project': r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\demo_a.prj'}
config = {'template_project': r'C:\Users\verd_he\projects\Torque2024\reference_model\15MW_NREL_DNV_v1.1_onshore_tower.prj'}
test = BladedLinModel(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\test_polar_modification', config)
preprocessed_data = {'polar_cl': 0,
                     'polar_alpha_tes': 0,
                     'polar_alpha_max': 0,
                     'polar_alpha_sr': 0.2}
preprocessed_data = {'polar_alpha_max': np.array(0.0),
                     'polar_clalpha': 0.2,
                     'polar_cd0': 0.0}

test.create_simulation(preprocessed_data)