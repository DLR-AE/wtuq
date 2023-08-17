from tool_interfaces.bladed_lin_interface import BladedLinModel
import numpy as np

# config = {'template_project': r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\demo_a.prj'}
config = {'template_project': r'C:\Users\verd_he\projects\Torque2024\reference_model\15MW_NREL_DNV_v1.1_onshore_tower.prj'}
test = BladedLinModel(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\test_blade_structure_modification', config)

preprocessed_data = {'blade_chord_root': 0,
                     'blade_chord_tip': 0.1,
                     'blade_twist': 0.1,
                     'blade_na_x': 0.1,
                     'blade_na_y': 0.1,
                     'blade_flap_stiffness': 0.1,
                     'blade_edge_stiffness': 0.1,
                     'blade_mass': 0.1,
                     'blade_radial_cog': 0,
                     'blade_pa_orientation': 0.1,
                     'blade_cog_x': 0.01,
                     'blade_cog_y': 0.1,
                     'blade_sc_x': 0.01,
                     'blade_sc_y': 0.1}

test.create_simulation(preprocessed_data)
