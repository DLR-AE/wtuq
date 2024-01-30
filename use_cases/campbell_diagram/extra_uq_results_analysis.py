"""
This module contains a collection of mostly independent methods for the analysis and plotting of the uncertainty
quantification results. The methods are very "Torque2024" specific, but can be used as a blueprint for other case studies.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['hatch.color']     = 'grey'
matplotlib.rcParams['hatch.linewidth'] = 0.2
import numpy as np
import os
import glob
import pandas as pd
from wtuq_framework.uq_results_analysis import UQResultsAnalysis
from wtuq_framework.helperfunctions import load_dict_h5py
import chaospy as cp
import numpoly

uncertain_param_label_dict = {'tower_bending_stiffness': 'Tower bending stiffness',
                              'tower_torsional_stiffness': 'Tower torsional stiffness',
                              'tower_shear_stiffness': 'Tower shear stiffness',
                              'tower_mass': 'Tower mass',
                              'tower_damping_1': 'Damping $\mathregular{1^{st}}$ tower mode',
                              'tower_damping_2': 'Damping $\mathregular{2^{nd}}$ tower mode',
                              'blade_mass': 'Blade mass',
                              'blade_edge_stiffness': 'Blade edgewise stiffness',
                              'blade_flap_stiffness': 'Blade flapwise stiffness',
                              'blade_tors_stiffness': 'Blade torsional stiffness',
                              'blade_pa_orientation': 'Blade principal axis orientation',
                              'blade_cog_x': r'Blade c.o.g. position $\parallel$ to chord',
                              'blade_cog_y': r'Blade c.o.g. position $\bot$ to chord',
                              'blade_sc_x': r'Blade shear center position $\parallel$ to chord',
                              'blade_sc_y': r'Blade shear center position $\bot$ to chord',
                              'blade_na_x': r'Blade prebend',
                              'blade_na_y': r'Blade sweep',
                              'blade_chord_tip': 'Blade chord length (tip)',
                              'blade_chord_root': 'Blade chord length (root)',
                              'blade_twist': 'Blade twist angle',
                              'blade_damping_1': 'Damping $\mathregular{1^{st}}$ blade mode',
                              'blade_damping_2': 'Damping $\mathregular{2^{nd}}$ blade mode',
                              'blade_damping_3': 'Damping $\mathregular{3^{rd}}$ blade mode',
                              'blade_damping_4': 'Damping $\mathregular{4^{th}}$ blade mode',
                              'cone_angle': 'Cone angle',
                              'polar_clalpha': r'Polar gradient $\mathrm{C_{l_{\alpha}}}$',
                              'polar_alpha_max': r'Polar max. $\alpha$',
                              'polar_cd0': r'Polar $\mathrm{C_{d_{0}}}$',
                              'separation_time_constant': 'Dyn. stall - separation time constant',
                              'pressure_lag_time_constant': 'Dyn. stall - pressure lag time constant',
                              'vortex_lift_time_constant': 'Dyn. stall - vortex lift time constant',
                              'nacelle_mass': 'Nacelle mass',
                              'nacelle_yaw_inertia': 'Nacelle yaw inertia',
                              'nacelle_rolling_inertia': 'Nacelle rolling inertia',
                              'nacelle_nodding_inertia': 'Nacelle nodding inertia',
                              'nacelle_cog_vertical': 'Nacelle cog position vertical',
                              'nacelle_cog_horizontal': 'Nacelle cog position horizontal',
                              'LSS_stiffness': 'Drivetrain stiffness',
                              'LSS_damping': 'Drivetrain damping',
                              'ds_a1': 'Dyn. stall param a1',
                              'ds_a2': 'Dyn. stall param a2',
                              'ds_b1': 'Dyn. stall param b1',
                              'ds_b2': 'Dyn. stall param b2',
                              'ds_taupre': r'Dyn. stall param $\tau_{\mathrm{pre}}$',
                              'ds_taubly': r'Dyn. stall param $\tau_{\mathrm{bly}}$',
                              'fw_mixing_ratio': 'BEM far wake mixing ratio',
                              'fw_poly_coef': 'BEM far wake polynomial coefficients',
                              'nw_mixing_ratio': 'BEM near wake mixing ratio',
                              'nw_poly_coef': 'BEM near wake polynomial coefficients'}

features = ['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four',
            'edge_mode_five', 'edge_mode_six', 'edge_mode_seven']
# features = ['edge_mode_five', 'edge_mode_six']
feat_colors = {'edge_mode_one': 'C0',
               'edge_mode_two': 'C1',
               'edge_mode_three': 'C2',
               'edge_mode_four': 'C6',
               'edge_mode_five': 'C9',
               'edge_mode_six': 'C8',
               'edge_mode_seven': 'C4',
               'total': 'C7'}
feat_names = {'edge_mode_one': '$\mathregular{1^{st}}$ edgewise backward whirl',
              'edge_mode_two': '$\mathregular{1^{st}}$ edgewise collective',
              'edge_mode_three': '$\mathregular{1^{st}}$ edgewise forward whirl',
              'edge_mode_four': '$\mathregular{2^{nd}}$ edgewise collective',
              'edge_mode_five': '$\mathregular{2^{nd}}$ tower f-a',
              'edge_mode_six': '$\mathregular{2^{nd}}$ edgewise backward whirl',
              'edge_mode_seven': '$\mathregular{2^{nd}}$ edgewise forward whirl',
              'total': 'Total'}
short_feat_names = {'edge_mode_one': '$\mathregular{1^{st}}$ edgewise BW',
                    'edge_mode_two': '$\mathregular{1^{st}}$ edgewise coll.',
                    'edge_mode_three': '$\mathregular{1^{st}}$ edgewise FW',
                    'edge_mode_four': '$\mathregular{2^{nd}}$ edgewise coll.',
                    'edge_mode_five': '$\mathregular{2^{nd}}$ tower f-a',
                    'edge_mode_six': '$\mathregular{2^{nd}}$ edgewise BW',
                    'edge_mode_seven': '$\mathregular{2^{nd}}$ edgewise FW',
                    'total': 'Total'}
extra_short_feat_names = {'edge_mode_one': '$\mathregular{1^{st}}$ BW',
                    'edge_mode_two': '$\mathregular{1^{st}}$ col.',
                    'edge_mode_three': '$\mathregular{1^{st}}$ FW',
                    'edge_mode_four': '$\mathregular{2^{nd}}$ col.',
                    'edge_mode_five': '$\mathregular{2^{nd}}$ tow. f-a',
                    'edge_mode_six': '$\mathregular{2^{nd}}$ BW',
                    'edge_mode_seven': '$\mathregular{2^{nd}}$ FW',
                    'total': 'Total'}

base_reference_directory = r'./reference_model/hawcstab2/IEA-15-240-RWT-Onshore-master-25Points-reference-run'
cmb_filename = r'IEA_15MW_RWT_Onshore.cmb'
amp_filename = r'IEA_15MW_RWT_Onshore.amp'
opt_filename = r'IEA_15MW_RWT_Onshore_25Points.opt'

from tool_interfaces.hawcstab2_interface import HAWCStab2Model
ref_hs2_results = HAWCStab2Model(base_reference_directory, {'cmb_filename': cmb_filename, 'amp_filename': amp_filename, 'opt_filename': opt_filename})
result_dict = ref_hs2_results.extract_results()
reference_frequency_progression = {'Rotor 1st edgewise backward whirl': result_dict['frequency'][5, :],
                                   'Rotor 1st edgewise collective': result_dict['frequency'][6, :],
                                   'Rotor 1st edgewise forward whirl': result_dict['frequency'][7, :],
                                   'Rotor 2nd edgewise collective': result_dict['frequency'][11, :],
                                   'Rotor 2nd edgewise backward whirl A': result_dict['frequency'][12, :],
                                   'Rotor 2nd edgewise backward whirl B': result_dict['frequency'][13, :],
                                   'Rotor 2nd edgewise forward whirl': result_dict['frequency'][14, :]}

reference_damping_progression = {'Rotor 1st edgewise backward whirl': result_dict['damping'][5, :],
                                 'Rotor 1st edgewise collective': result_dict['damping'][6, :],
                                 'Rotor 1st edgewise forward whirl': result_dict['damping'][7, :],
                                 'Rotor 2nd edgewise collective': result_dict['damping'][11, :],
                                 'Rotor 2nd edgewise backward whirl A': result_dict['damping'][12, :],
                                 'Rotor 2nd edgewise backward whirl B': result_dict['damping'][13, :],
                                 'Rotor 2nd edgewise forward whirl': result_dict['damping'][14, :]}

# ws_range = np.linspace(3, 25, 12)
ws_range = np.array([3, 5, 7, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 17,
                     19, 21, 23, 25])

n_ws = len(ws_range)

def campbell_diagram_with_uncertainty_bands(UQResultsAnalysis):
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    nr_wind_speeds = 6  # uq_results.data['CampbellDiagramModel'].evaluations.shape[1]
    start_wind_speed = 10
    nr_modes = 4  # uq_results.data['CampbellDiagramModel'].evaluations.shape[2]
    unpacked_mean_campbell_data = uq_results.data['CampbellDiagramModel'].mean.reshape(
        (nr_wind_speeds * 2, nr_modes))
    unpacked_percentile_5_campbell_data = uq_results.data['CampbellDiagramModel'].percentile_5.reshape(
        (nr_wind_speeds * 2, nr_modes))
    unpacked_percentile_95_campbell_data = uq_results.data['CampbellDiagramModel'].percentile_95.reshape(
        (nr_wind_speeds * 2, nr_modes))

    fig = plt.figure('Campbell Diagram with Uncertainty Bands')
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    color_seq = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for ii, mode_name in enumerate(['mode1', 'mode2', 'mode3', 'mode4']):
        ws_range = np.arange(start_wind_speed, nr_wind_speeds + start_wind_speed)
        ax1.plot(ws_range, unpacked_mean_campbell_data[:nr_wind_speeds, ii], color=color_seq[ii], label=mode_name)
        ax2.plot(ws_range, unpacked_mean_campbell_data[nr_wind_speeds:, ii], color=color_seq[ii], label=mode_name)

        ax1.fill_between(ws_range, unpacked_percentile_5_campbell_data[:nr_wind_speeds, ii],
                         unpacked_percentile_95_campbell_data[:nr_wind_speeds, ii],
                         color=color_seq[ii], alpha=0.2)
        ax2.fill_between(ws_range, unpacked_percentile_5_campbell_data[nr_wind_speeds:, ii],
                         unpacked_percentile_95_campbell_data[nr_wind_speeds:, ii],
                         color=color_seq[ii], alpha=0.2)
        '''
        ax1.plot(np.arange(start_wind_speed, nr_wind_speeds + start_wind_speed),
                 unpacked_percentile_5_campbell_data[:nr_wind_speeds, ii], '--', color=color_seq[ii])
        ax2.plot(np.arange(start_wind_speed, nr_wind_speeds + start_wind_speed),
                 unpacked_percentile_5_campbell_data[nr_wind_speeds:, ii], '--', color=color_seq[ii])

        ax1.plot(np.arange(start_wind_speed, nr_wind_speeds + start_wind_speed),
                 unpacked_percentile_95_campbell_data[:nr_wind_speeds, ii], '--', color=color_seq[ii])
        ax2.plot(np.arange(start_wind_speed, nr_wind_speeds + start_wind_speed),
                 unpacked_percentile_95_campbell_data[nr_wind_speeds:, ii], '--', color=color_seq[ii])
        '''

    ax1.legend()
    ax1.grid()
    ax2.grid()
    plt.show()


def analyse_unstable_pce_runs(dir, base_output_name='CampbellViewerDatabase_', get_run_name_from='from_input_params'):
    """
    Main problem: seems like memory overload if too many datasets are loaded in to the database, therefore divide it
    into multiple databases
    """
    from campbellviewer.settings.globals import database
    import json
    # uq_results = UQResultsAnalysis.get_UQ_results(os.path.join(dir[:-1], 'uq_results', 'CampbellDiagramModel.h5'))

    fig = plt.figure('Samples for unsteady simulations')
    ax = fig.gca()

    nr_datasets_added = 0
    max_nr_datasets = 20
    database_nr = 0

    min_total_required_model_modification = 999
    iteration_min_total_required_model_modification = ''
    for dir in glob.glob(dir):

        if dir.split('/')[-1] == 'uq_results':
            continue

        test = load_dict_h5py(os.path.join(dir, 'result_dict.hdf5'))

        if np.any(test['damping'] < 0):

            try:
                with open(os.path.join(dir, 'input_parameters.txt')) as f:
                    data = f.read()
                input_params = json.loads(data)
                print(input_params)

                input_param_names = ['tower_bending_stiffness_scalar_property',
                                     'tower_mass_scalar_property',
                                     'polar_clalpha_scalar_property',
                                     'blade_chord_tip_scalar_property',
                                     'blade_na_x_scalar_property',
                                     'blade_flap_stiffness_scalar_property',
                                     'blade_edge_stiffness_scalar_property',
                                     'blade_mass_scalar_property',
                                     'blade_cog_x_scalar_property',
                                     'blade_cog_y_scalar_property',
                                     'blade_sc_x_scalar_property',
                                     'blade_sc_y_scalar_property']

                total_required_model_modification = 0
                for i, name in enumerate(input_param_names):
                    ax.plot(i, input_params[name], color='C0', marker='o')
                    if name == 'blade_cog_y_scalar_property' or name == 'blade_sc_y_scalar_property':
                        total_required_model_modification += np.abs(10 * input_params[name])
                    else:
                        total_required_model_modification += np.abs(input_params[name])

                run_name = dir.split('/')[-1]
                if total_required_model_modification < min_total_required_model_modification:
                    min_total_required_model_modification = total_required_model_modification
                    print('Minimum total required model modification', min_total_required_model_modification)
                    iteration_min_total_required_model_modification = run_name

                database.add_data(run_name, 'hawcstab2',
                                  tool_specific_info={'filenamecmb': os.path.join(dir, 'htc', 'IEA_15MW_RWT_Onshore.cmb'),
                                                      'filenameamp': os.path.join(dir, 'htc', 'IEA_15MW_RWT_Onshore.amp'),
                                                      'filenameopt': os.path.join(dir, 'htc', 'IEA_15MW_RWT_Onshore_12Points.opt'),
                                                      'skip_header_CMB': 1,
                                                      'skip_header_AMP': 5,
                                                      'skip_header_OP': 1})
                nr_datasets_added += 1

                if nr_datasets_added == max_nr_datasets:
                    # database.save(fname='{}_{}.nc'.format(base_output_name, database_nr))
                    database['HAWCStab2'] = dict()
                    database_nr += 1

                    nr_datasets_added = 0

            except FileNotFoundError:
                print('File not found in directory', dir)

    database.save(fname='{}_{}.nc'.format(base_output_name, database_nr))
    print(iteration_min_total_required_model_modification)
    plt.show()


def transform_to_campbellviewer_database(dir, base_output_name='CampbellViewerDatabase_', get_run_name_from='from_input_params'):
    """
    Main problem: seems like memory overload if too many datasets are loaded in to the database, therefore divide it
    into multiple databases
    """
    from campbellviewer.settings.globals import database
    import json
    # uq_results = UQResultsAnalysis.get_UQ_results(os.path.join(dir[:-1], 'uq_results', 'CampbellDiagramModel.h5'))

    nr_datasets_added = 0
    max_nr_datasets = 20
    database_nr = 0
    for dir in glob.glob(dir):

        try:
            with open(os.path.join(dir, 'input_parameters.txt')) as f:
                data = f.read()
            input_params = json.loads(data)

            run_name = dir.split('/')[-1]  # default name if specific run name can not be found
            if get_run_name_from == 'from_input_params':
                for param, value in input_params.items():
                    if value < 0:
                        # CUTOUT 'SCALAR PROPERTY'
                        run_name = param[:-16] + '_low'
                    if value > 0:
                        # CUTOUT 'SCALAR PROPERTY'
                        run_name = param[:-16] + '_high'

            database.add_data(run_name, 'hawcstab2',
                              tool_specific_info={'filenamecmb': os.path.join(dir, 'htc', 'IEA_15MW_RWT_Onshore.cmb'),  # 'IEA_15MW_RWT_Onshore.cmb'),
                                                  'filenameamp': os.path.join(dir, 'htc', 'IEA_15MW_RWT_Onshore.amp'),  # 'IEA_15MW_RWT_Onshore.amp'),
                                                  'filenameopt': os.path.join(dir, 'htc', 'IEA_15MW_RWT_Onshore_12Points.opt'),  # 'IEA_15MW_RWT_Onshore.opt'
                                                  'skip_header_CMB': 1,
                                                  'skip_header_AMP': 5,
                                                  'skip_header_blade_only_AMP': 3,
                                                  'skip_header_OP': 1,
                                                  'blade-only': False})
            nr_datasets_added += 1

            if nr_datasets_added == max_nr_datasets:
                database.save(fname='{}_{}.nc'.format(base_output_name, database_nr))
                database['HAWCStab2'] = dict()
                database_nr += 1

                nr_datasets_added = 0

        except FileNotFoundError:
            print('File not found in directory', dir)

    database.save(fname='{}_{}.nc'.format(base_output_name, database_nr))


def result_analysis_OAT(UQResultsAnalysis):
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    UQResultsAnalysis.uncertain_param_names[UQResultsAnalysis.uncertainpy_results[0]] = uq_results.uncertain_parameters

    oat_max = dict()
    oat_mean = dict()
    oat_max[UQResultsAnalysis.uncertainpy_results[0]] = dict()
    oat_mean[UQResultsAnalysis.uncertainpy_results[0]] = dict()

    for feature in uq_results.data:
        if feature == 'CampbellDiagramModel':
            continue
        oat_max[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].s_oat_max
        oat_mean[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].s_oat_mean

    oat_max[UQResultsAnalysis.uncertainpy_results[0]]['total'] = np.max(np.vstack(([oat_max[UQResultsAnalysis.uncertainpy_results[0]][feat] for feat in features])), axis=0)
    oat_mean[UQResultsAnalysis.uncertainpy_results[0]]['total'] = np.mean(np.vstack(([oat_mean[UQResultsAnalysis.uncertainpy_results[0]][feat] for feat in features])), axis=0)

    oat_plot_paper(UQResultsAnalysis, oat_max, ylabel='max. OAT metric')
    oat_plot_paper(UQResultsAnalysis, oat_mean, ylabel='mean OAT metric')

    compare_oat_metrics(UQResultsAnalysis, oat_max, ylabel='max. OAT metric')
    compare_oat_metrics(UQResultsAnalysis, oat_mean, ylabel='mean OAT metric')

    compare_macs(UQResultsAnalysis, uq_results, features, ws_range)
    # compare_QoI_evaluations_per_iter_with_mac_values(UQResultsAnalysis, uq_results, features, ws_range)

    campbell_diagram_from_OAT(UQResultsAnalysis, uq_results)


def result_analysis_EE(UQResultsAnalysis):
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    UQResultsAnalysis.uncertain_param_names[UQResultsAnalysis.uncertainpy_results[0]] = uq_results.uncertain_parameters

    campbell_diagrams_from_EE(UQResultsAnalysis, uq_results)
    campbell_diagrams_from_EE_with_rejected_with_mac_values(UQResultsAnalysis, uq_results)
    # ee_plot_like_sobols(UQResultsAnalysis, uq_results)

    ee_mean_max = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_mean_mean = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_max = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_mean = {UQResultsAnalysis.uncertainpy_results[0]: {}}

    for feature in features:
        ee_mean_max[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_mean_max
        ee_mean_mean[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_mean_mean
        ee_std_max[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_std_max
        ee_std_mean[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_std_mean

    for dicti in [ee_mean_max[UQResultsAnalysis.uncertainpy_results[0]],
                  ee_std_max[UQResultsAnalysis.uncertainpy_results[0]]]:
        dicti['total'] = np.max(np.vstack(([dicti[feat] for feat in features])), axis=0)

    for dicti in [ee_mean_mean[UQResultsAnalysis.uncertainpy_results[0]],
                  ee_std_mean[UQResultsAnalysis.uncertainpy_results[0]]]:
        dicti['total'] = np.mean(np.vstack(([dicti[feat] for feat in features])), axis=0)


    compare_oat_metrics_with_std_bar(UQResultsAnalysis, ee_mean_max, ee_std_max, ylabel='max. EE mean')
    compare_oat_metrics_with_std_bar(UQResultsAnalysis, ee_mean_mean, ee_std_mean, ylabel='mean EE mean')

    compare_oat_metrics(UQResultsAnalysis, ee_mean_max, ylabel='max. EE mean')
    compare_oat_metrics(UQResultsAnalysis, ee_mean_mean, ylabel='mean EE mean')
    compare_oat_metrics(UQResultsAnalysis, ee_std_max, ylabel='max. EE std')
    compare_oat_metrics(UQResultsAnalysis, ee_std_mean, ylabel='mean EE std')

    # Mean of EE damping edge mode 5 & 6
    ee_mean_damp_mode_5_6 = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_damp_mode_5_6 = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    print('Last 8 indices in ee -> damping > 15 m/s')
    for feature in ['edge_mode_five', 'edge_mode_six']:
        ee_mean_damp_mode_5_6[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.mean(uq_results.data[feature].ee_mean[:, -8:], axis=1)
        ee_std_damp_mode_5_6[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.mean(uq_results.data[feature].ee_std[:, -8:], axis=1)

    compare_oat_metrics(UQResultsAnalysis, ee_mean_damp_mode_5_6, ylabel='mean EE damping edge mode 5 and 6',
                        manual_feature_selection=['edge_mode_five', 'edge_mode_six'])

    # EE of damping only
    ee_mean_max_damp = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_mean_mean_damp = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_max_damp = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_mean_damp = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    print('Last 25 indices in ee -> damping')
    for feature in features:
        ee_mean_max_damp[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.max(uq_results.data[feature].ee_mean[:, -25:], axis=1)
        ee_mean_mean_damp[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.mean(uq_results.data[feature].ee_mean[:, -25:], axis=1)
        ee_std_max_damp[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.max(uq_results.data[feature].ee_std[:, -25:], axis=1)
        ee_std_mean_damp[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.mean(uq_results.data[feature].ee_std[:, -25:], axis=1)

    compare_oat_metrics(UQResultsAnalysis, ee_mean_max_damp, ylabel='max. EE mean only damping qoi',
                        manual_feature_selection=['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four',
                                                  'edge_mode_five', 'edge_mode_six', 'edge_mode_seven'])
    compare_oat_metrics(UQResultsAnalysis, ee_mean_mean_damp, ylabel='mean EE mean only damping qoi',
                        manual_feature_selection=['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four',
                                                  'edge_mode_five', 'edge_mode_six', 'edge_mode_seven'])
    compare_oat_metrics(UQResultsAnalysis, ee_std_max_damp, ylabel='max. EE std only damping qoi',
                        manual_feature_selection=['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four',
                                                  'edge_mode_five', 'edge_mode_six', 'edge_mode_seven'])
    compare_oat_metrics(UQResultsAnalysis, ee_std_mean_damp, ylabel='mean EE std only damping qoi',
                        manual_feature_selection=['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four',
                                                  'edge_mode_five', 'edge_mode_six', 'edge_mode_seven'])

    # std_vs_ee_plot(UQResultsAnalysis, uq_results)


def result_analysis_PCE(UQResultsAnalysis, UQResultsAnalysis_control_run=None):
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])
    for feature in features:
        print('Feature {}: {} / {} samples used'.format(feature,
                                                        len(uq_results.data[feature].masked_evaluations),
                                                        len(uq_results.data[feature].evaluations)))

    input_params_rejected_samples(UQResultsAnalysis, features, max_iterations=200)

    # compare_QoI_evaluations_per_iter_with_loo(UQResultsAnalysis, uq_results, features, ws_range, max_iterations=20)
    compare_QoI_evaluations_per_iter_with_mac_values(UQResultsAnalysis, uq_results, features, ws_range, max_iterations=40)
    compare_QoI_evaluations_per_iter_with_rejected_with_mac_values(UQResultsAnalysis, stop_iteration=100, plot_mac=True)
    # compare_macs(UQResultsAnalysis, uq_results, features, ws_range)

    sobol_index_plots2(UQResultsAnalysis, uq_results, ws_range)

    surrogate_model_verification(UQResultsAnalysis, uq_results, features, ws_range)
    #surrogate_model_verification_with_control_data(UQResultsAnalysis, uq_results, features, ws_range, UQResultsAnalysis_control_run)

    campbell_diagram_from_PCE(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range)
    #first_order_campbell_diagrams(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range)


def compare_macs(UQResultsAnalysis, uq_results, features, ws_range):

    for feature in features:

        mac = uq_results.data[uq_results.model_name]['mac_{}'.format(feature)]

        fig = plt.figure('MAC values - {}'.format(feature), figsize=(10, 10))
        ax = fig.gca()

        param_names = [str(iter_run_dir).split('/')[-1] for iter_run_dir in uq_results.data[uq_results.model_name]['iteration_run_directory']]
        df = pd.DataFrame(mac, columns=ws_range[1:])  # , index=param_names

        plot1 = sns.heatmap(df, ax=ax, annot=True, fmt=".2f", cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
        # plot.set_xticklabels(plot.get_xticklabels(), rotation=65, ha='right', rotation_mode='anchor')
        ax.set_xlabel('Wind speed / m/s')
        plt.tight_layout()

        # fig = plot.get_figure()
        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'MAC - {}.png'.format(feature)), dpi=300)

        # plt.show()
        plt.close()


def compare_QoI_evaluations_per_iter_with_loo(UQResultsAnalysis, uq_results, features, ws_range, max_iterations=1E10):

    fig1 = plt.figure('Comparison QoI evaluations', figsize=(16, 12))
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    plt.subplots_adjust(hspace=0)
    ax1.set_ylabel('Frequency in Hz')
    ax2.set_xlabel('Wind speed in m/s')
    ax2.set_ylabel('Damping ratio in %')

    for feature in features:

        evaluations = uq_results.data[feature].masked_evaluations
        evaluations_loo = uq_results.data[feature].evaluations_loo
        nr_nodes_for_loo = evaluations_loo.shape[0]
        nr_iter_to_show = min(nr_nodes_for_loo, max_iterations)

        ax1.plot(ws_range, evaluations[:nr_iter_to_show, :n_ws].T, markersize=10, label=feature)
        ax2.plot(ws_range, evaluations[:nr_iter_to_show, n_ws:].T, markersize=10, label=feature)

        ax1.plot(ws_range, evaluations_loo[:nr_iter_to_show, :n_ws].T, linestyle='--', markersize=10, label=feature)
        ax2.plot(ws_range, evaluations_loo[:nr_iter_to_show, n_ws:].T, linestyle='--', markersize=10, label=feature)

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig1.savefig(os.path.join(UQResultsAnalysis.output_dir, 'QoI_evaluations.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig1)


def compare_QoI_evaluations_per_iter_with_mac_values(UQResultsAnalysis, uq_results, features, ws_range, max_iterations=None):

    fig1 = plt.figure('Comparison QoI evaluations', figsize=(16, 12))
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    plt.subplots_adjust(hspace=0)
    ax1.set_ylabel('Frequency in Hz')
    ax2.set_xlabel('Wind speed in m/s')
    ax2.set_ylabel('Damping ratio in %')

    from matplotlib import colors
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    # cmap = plt.cm.cool
    # cmap = sns.color_palette("coolwarm", as_cmap=True)
    color_norm = colors.Normalize(vmin=0.2, vmax=0.95)

    for feature in features:

        #if 'masked_evaluations' in uq_results.data[feature]:
        #    evaluations = uq_results.data[feature].masked_evaluations
        #else:
        evaluations = uq_results.data[feature].evaluations

        mac = uq_results.data[uq_results.model_name]['mac_{}'.format(feature)]
        mac_op0 = [np.real(mac) for mac in uq_results.data[uq_results.model_name]['mac_to_ref_{}'.format(feature)]]

        for ii_eval, sample_evaluations in enumerate(evaluations):
            if ii_eval > max_iterations:
                break
            if not np.any(np.isnan(sample_evaluations)):
                ax1.plot(ws_range, sample_evaluations[:n_ws], markersize=10, label=feature)#, color=feat_colors[feature])
                ax2.plot(ws_range, sample_evaluations[n_ws:], markersize=10, label=feature)#, color=feat_colors[feature])

                for ws_ii, ws in enumerate(ws_range):
                    freq_eval = sample_evaluations[ws_ii]
                    damp_eval = sample_evaluations[n_ws + ws_ii]
                    if ws_ii == 0:
                        ax1.text(ws, freq_eval, '{:.2f}'.format(mac_op0[ii_eval]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac_op0[ii_eval])))
                        ax2.text(ws, damp_eval, '{:.2f}'.format(mac_op0[ii_eval]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac_op0[ii_eval])))
                    else:
                        ax1.text(ws, freq_eval, '{:.2f}'.format(mac[ii_eval, ws_ii-1]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac[ii_eval, ws_ii-1])))
                        ax2.text(ws, damp_eval, '{:.2f}'.format(mac[ii_eval, ws_ii-1]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac[ii_eval, ws_ii-1])))

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig1.savefig(os.path.join(UQResultsAnalysis.output_dir, 'QoI_evaluations_with_mac.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig1)


def compare_QoI_evaluations_per_iter_with_rejected_with_mac_values(UQResultsAnalysis, plot_mac=True,
                                                                   start_iteration=0, stop_iteration=1E10):

    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    fig1 = plt.figure('Comparison QoI evaluations', figsize=(16, 12))
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    plt.subplots_adjust(hspace=0)
    ax1.set_ylabel('Frequency in Hz')
    ax2.set_xlabel('Wind speed in m/s')
    ax2.set_ylabel('Damping ratio in %')

    from matplotlib import colors
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    # cmap = plt.cm.cool
    # cmap = sns.color_palette("coolwarm", as_cmap=True)
    color_norm = colors.Normalize(vmin=0.2, vmax=0.95)

    feature_order = ['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four',
                     'edge_mode_five', 'edge_mode_six', 'edge_mode_seven']

    for feature in features:

        index_feature = feature_order.index(feature)

        mac = uq_results.data[uq_results.model_name]['mac_{}'.format(feature)]
        mac_op0 = [np.real(mac) for mac in uq_results.data[uq_results.model_name]['mac_to_ref_{}'.format(feature)]]
        mac_last_op = [np.real(mac) for mac in uq_results.data[uq_results.model_name]['mac_to_ref_last_op_{}'.format(feature)]]

        for ii_eval in range(start_iteration, min(stop_iteration, len(uq_results.data['CampbellDiagramModel'].postprocessing_successful))):

            if bool(uq_results.data['CampbellDiagramModel'].postprocessing_successful[ii_eval, index_feature]):
                color = 'green'
            else:
                color = 'red'

            ax1.plot(ws_range, uq_results.data['CampbellDiagramModel'].frequency[ii_eval, :, index_feature], color=color, markersize=10)
            ax2.plot(ws_range, uq_results.data['CampbellDiagramModel'].damping[ii_eval, :, index_feature], color=color, markersize=10)

            if plot_mac:
                for ws_ii, ws in enumerate(ws_range):
                    freq_eval = uq_results.data['CampbellDiagramModel'].frequency[ii_eval, ws_ii, index_feature]
                    damp_eval = uq_results.data['CampbellDiagramModel'].damping[ii_eval, ws_ii, index_feature]
                    if ws_ii == 0:
                        ax1.text(ws, freq_eval, '{:.2f}'.format(mac_op0[ii_eval]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac_op0[ii_eval])))
                        ax2.text(ws, damp_eval, '{:.2f}'.format(mac_op0[ii_eval]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac_op0[ii_eval])))
                    elif ws_ii == 24:
                        ax1.text(ws, freq_eval, '{:.2f}'.format(mac_last_op[ii_eval]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac_last_op[ii_eval])))
                        ax2.text(ws, damp_eval, '{:.2f}'.format(mac_last_op[ii_eval]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac_last_op[ii_eval])))
                    else:
                        ax1.text(ws, freq_eval, '{:.2f}'.format(mac[ii_eval, ws_ii-1]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac[ii_eval, ws_ii-1])))
                        ax2.text(ws, damp_eval, '{:.2f}'.format(mac[ii_eval, ws_ii-1]), fontsize='small', weight="bold",
                                 ha='center', va='center', color=cmap(color_norm(mac[ii_eval, ws_ii-1])))

    fig1.savefig(os.path.join(UQResultsAnalysis.output_dir, 'campbell_accepted_rejected_iters_{}-{}.png'.format(start_iteration, stop_iteration-1)), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig1)


def input_params_rejected_samples(UQResultsAnalysis, features, max_iterations=None):
    import json

    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    iteration_run_directories = uq_results.data[uq_results.model_name].iteration_run_directory

    print('Careful: features assumed to have the same order as the mode selection from the postprocessor')
    fig, ax = plt.subplots(nrows=1, ncols=len(features), num='1D input parameters visualization',
                           figsize=(14, 10), sharey=True)
    if isinstance(ax, plt.Axes):
        ax = [ax]

    for feat_ii, feature in enumerate(features):
        input_params_accepted = []
        input_params_rejected_mode_picking = []
        input_params_rejected_mode_tracking = []
        input_params_rejected_mode_tracking_wrt_ref = []

        for iter_ii, dir in enumerate(iteration_run_directories):

            if max_iterations is not None:
                if max_iterations < iter_ii:
                    break

            try:
                with open(os.path.join(dir.decode('utf8'), 'input_parameters.txt')) as f:
                    data = f.read()
                input_params = json.loads(data)
            except FileNotFoundError:
                raise FileNotFoundError('File not found in directory', dir.decode('utf8'))

            if uq_results.data[uq_results.model_name].postprocessing_successful[iter_ii, feat_ii]:
                input_params_accepted.append(input_params)

            if not uq_results.data[uq_results.model_name].success_mode_picking[iter_ii, feat_ii]:
                input_params_rejected_mode_picking.append(input_params)

            if not uq_results.data[uq_results.model_name].success_mode_tracking[iter_ii, feat_ii]:
                input_params_rejected_mode_tracking.append(input_params)

            if not uq_results.data[uq_results.model_name].success_mode_tracking_to_ref_last_op[iter_ii, feat_ii]:
                input_params_rejected_mode_tracking_wrt_ref.append(input_params)

        for iparam, param in enumerate(uq_results.uncertain_parameters):

            for accepted_input_params in input_params_accepted:
                ax[feat_ii].plot(accepted_input_params[param], iparam, marker="o", color='C0')

            for rejected_input_params in input_params_rejected_mode_picking:
                ax[feat_ii].plot(rejected_input_params[param], iparam, marker="s", color='C3')

            for rejected_input_params in input_params_rejected_mode_tracking:
                ax[feat_ii].plot(rejected_input_params[param], iparam, marker="x", color='C1')

            for rejected_input_params in input_params_rejected_mode_tracking_wrt_ref:
                ax[feat_ii].plot(rejected_input_params[param], iparam, marker="+", color='C2')

        ax[feat_ii].set_title(feature)

    ax[0].yaxis.set_ticks(np.arange(len(uq_results.uncertain_parameters)))
    ax[0].set_yticklabels(uq_results.uncertain_parameters)

    point_accepted = matplotlib.lines.Line2D([0], [0], label='Accepted samples', marker="o", color='C0', linestyle='')
    point_rejected1 = matplotlib.lines.Line2D([0], [0], label='Rejected samples (mode picking)', marker="s", color='C3', linestyle='')
    point_rejected2 = matplotlib.lines.Line2D([0], [0], label='Rejected samples (mode tracking)', marker="x", color='C1', linestyle='')
    point_rejected3 = matplotlib.lines.Line2D([0], [0], label='Rejected samples (mode tracking wrt ref)', marker="+", color='C2', linestyle='')

    ax[-1].legend(handles=[point_accepted, point_rejected1, point_rejected2, point_rejected3])

    plt.tight_layout()

    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'rejected_samples.png'), dpi=300)
    plt.show()


def surrogate_model_verification(UQResultsAnalysis, uq_results, features, ws_range):
    fig, ax = plt.subplots(nrows=2, ncols=len(features), figsize=(12, 8))
    if isinstance(ax, plt.Axes):
        ax = [ax]

    for i_feat, feature in enumerate(features):

        fig_mode_individual, ax_mode_individual = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))

        evaluations_loo = uq_results.data[feature].evaluations_loo
        nr_nodes_for_loo = evaluations_loo.shape[0]
        if 'masked_evaluations' in uq_results.data[feature]:
            evaluations = uq_results.data[feature].masked_evaluations
        else:
            evaluations = uq_results.data[feature].evaluations

        success_percentage = len(uq_results.data[feature].masked_evaluations) / len(uq_results.data[feature].evaluations)
        if success_percentage > 0.8:
            successful_postprocessing = True
        else:
            successful_postprocessing = False

        for axx in [ax[:, i_feat], ax_mode_individual]:

            axx[0].set_title('pp success = {:.1f}%'.format(success_percentage*100))

            if successful_postprocessing:
                axx[0].scatter(evaluations[:nr_nodes_for_loo, :n_ws], evaluations_loo[:nr_nodes_for_loo, :n_ws], color='C0', s=3, alpha=0.5, label='Frequency')
                axx[1].scatter(evaluations[:nr_nodes_for_loo, n_ws:], evaluations_loo[:nr_nodes_for_loo, n_ws:], color='C1', s=3, alpha=0.5, label='Damping')
            else:
                axx[0].scatter(evaluations[:nr_nodes_for_loo, :n_ws], evaluations_loo[:nr_nodes_for_loo, :n_ws], color='red', s=3, alpha=0.5, label='Frequency')
                axx[1].scatter(evaluations[:nr_nodes_for_loo, n_ws:], evaluations_loo[:nr_nodes_for_loo, n_ws:], color='red', s=3, alpha=0.5, label='Damping')

            for axxx in axx:
                axxx.set_ylabel('Leave-one-out surrogate model evaluations')
                axxx.grid(True)
                axxx.set_axisbelow(True)
                axxx.legend(loc='upper left', markerscale=4)

            axx[1].set_xlabel('Model evaluations (training data)')

        plt.tight_layout()
        fig_mode_individual.savefig(os.path.join(UQResultsAnalysis.output_dir, 'surrogate_model_verification_{}.png'.format(feature)), bbox_inches='tight', dpi=300)
        plt.close()


    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'surrogate_model_verification.png'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


def surrogate_model_verification_with_control_data(UQResultsAnalysis, uq_results, features, ws_range, UQResultsAnalysis_control_run):
    uq_results_control_run = UQResultsAnalysis_control_run.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])
    iteration_run_directories = uq_results.data[uq_results.model_name].iteration_run_directory

    fig, ax = plt.subplots(nrows=2, ncols=len(features), figsize=(12, 8))
    if isinstance(ax, plt.Axes):
        ax = [ax]

    for i_feat, feature in enumerate(features):

        fig_mode_individual, ax_mode_individual = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))

        control_point_nodes = uq_results_control_run.data[feature].nodes  # [iter, :]
        control_point_masked_evaluations = uq_results_control_run.data[feature].masked_evaluations

        polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)

        evaluated_polynom = numpoly.call(polynom, control_point_nodes).T

        success_percentage = len(uq_results.data[feature].masked_evaluations) / len(uq_results.data[feature].evaluations)
        if success_percentage > 0.8:
            successful_postprocessing = True
        else:
            successful_postprocessing = False

        for axx in [ax[:, i_feat], ax_mode_individual]:

            axx[0].set_title('pp success = {:.1f}%'.format(success_percentage*100))

            if successful_postprocessing:
                axx[0].scatter(control_point_masked_evaluations[:, :n_ws], evaluated_polynom[:, :n_ws], color='C0', s=3, alpha=0.5, label='Frequency')
                axx[1].scatter(control_point_masked_evaluations[:, n_ws:], evaluated_polynom[:, n_ws:], color='C1', s=3, alpha=0.5, label='Damping')
            else:
                axx[0].scatter(control_point_masked_evaluations[:, :n_ws], evaluated_polynom[:, :n_ws], color='red', s=3, alpha=0.5, label='Frequency')
                axx[1].scatter(control_point_masked_evaluations[:, n_ws:], evaluated_polynom[:, n_ws:], color='red', s=3, alpha=0.5, label='Damping')

            for axxx in axx:
                axxx.set_ylabel('Leave-one-out surrogate model evaluations')
                axxx.grid(True)
                axxx.set_axisbelow(True)
                axxx.legend(loc='upper left', markerscale=4)

            axx[1].set_xlabel('Model evaluations (training data)')

            plt.tight_layout()
            fig_mode_individual.savefig(os.path.join(UQResultsAnalysis.output_dir, 'surrogate_model_verification_with_control_data_{}.png'.format(feature)), bbox_inches='tight', dpi=300)
            plt.close()

    plt.tight_layout()
    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'surrogate_model_verification_with_control_data.png'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


def first_order_effects(UQResultsAnalysis, uq_results, features, ws_range):

    uncertain_param_names = uq_results.uncertain_parameters
    show_op_points_indices = [0, 2, 5, 9, 13, 17, 20, 22, 24]

    for feature in features:
        polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)

        fig, axes = plt.subplots(nrows=2, ncols=len(show_op_points_indices), num='First order effects', figsize=(22, 7))
        #fig.tight_layout()

        for iter, uncertain_param_name in enumerate(uncertain_param_names):

            samples = uq_results.data[feature].nodes[iter, :]

            # scale_to_minus1_plus1 = np.linspace(-1, 1, nr_samples_plot_1D)

            # mean values of all input distributions
            print('Assumed that all distributions have mean = 0 for first order effect plotting!')
            sample_means = np.zeros((len(uncertain_param_names), len(samples)))

            # replace mean value of this iter with the sample points
            sample_means[iter] = samples

            evaluated_polynom = numpoly.call(polynom, sample_means)

            for ii, i_op_point in enumerate(show_op_points_indices):
                axes[0, ii].plot(samples, evaluated_polynom[i_op_point, :], '.', label=uncertain_param_name)
                axes[1, ii].plot(samples, evaluated_polynom[i_op_point+n_ws, :], '.', label=uncertain_param_name)

            #sample_means[iter] = samples_plot
            #evaluated_polynom = numpoly.call(U_hat[model_name], sample_means)
            #ax.plot(samples_plot, evaluated_polynom, label=uncertain_param_name + ': sampling within bounds training data')

            #ax2.plot(scale_to_minus1_plus1, evaluated_polynom, label=uncertain_param_name)

        for ii, i_op_point in enumerate(show_op_points_indices):
            axes[0, ii].set_title('ws = {} m/s'.format(ws_range[i_op_point]))

        axes[0, -1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'first_order_effects_{}.png'.format(feature)), bbox_inches='tight', dpi=300)
        # fig2.savefig(os.path.join(self.output_dir, 'first_order_effects_scaled_to_-1_to_1.png'), bbox_inches='tight')
        plt.show()


def sobol_index_plots(UQResultsAnalysis, uq_results):

    qoi_names = ['Freq. @ {} m/s'.format(ws) for ws in range(2, 22, 2)] + ['Damp. @ {} m/s'.format(ws) for ws in range(2, 22, 2)]
    for feature in uq_results.data:
        if feature == uq_results.model_name and uq_results.model_ignore == True:
            continue

        for which_sobol, sobol_arr in zip(['First order Sobol index', 'Total Sobol index'],
                                          [uq_results.data[feature].sobol_first, uq_results.data[feature].sobol_total]):

            plt.figure('{} - {}'.format(which_sobol, feature), figsize=(15, 5))
            df = pd.DataFrame(sobol_arr, index=uq_results.uncertain_parameters, columns=qoi_names)

            plot = sns.heatmap(df, annot=True, fmt=".3f", cbar=False)
            plot.set_xticklabels(plot.get_xticklabels(), rotation=65, ha='right', rotation_mode='anchor')
            plt.tight_layout()

            # plt.show()

            fig = plot.get_figure()
            fig.savefig(os.path.join(UQResultsAnalysis.output_dir, '{} - {}.png'.format(which_sobol, feature)), dpi=300)


def sobol_index_plots2(UQResultsAnalysis, uq_results, ws_range):

    param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]
    for feature in features:
        for which_sobol, sobol_arr in zip(['First order Sobol index', 'Total Sobol index'],
                                          [uq_results.data[feature].sobol_first_manual, uq_results.data[feature].sobol_total_manual]):
            tabular_contour_plot(feature, which_sobol, sobol_arr, param_names, UQResultsAnalysis.output_dir)


def ee_plot_like_sobols(UQResultsAnalysis, uq_results):

    param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]
    for feature in features:
        normalized_ee_mean = uq_results[feature].ee_mean / np.max(uq_results[feature].ee_mean)
        tabular_contour_plot(feature, 'ee_mean', normalized_ee_mean, param_names, UQResultsAnalysis.output_dir)


def tabular_contour_plot(feature, name, arr, param_names, output_dir):

    fig_subplots, ax_subplots = plt.subplots(num='{} - {} - vertical'.format(name, feature),
                                                 nrows=2, ncols=1, figsize=(10, 10), sharex=True)

    arr_freq, arr_damp = np.split(arr, 2, axis=1)
    df_freq = pd.DataFrame(arr_freq, index=param_names, columns=ws_range)
    df_damp = pd.DataFrame(arr_damp, index=param_names, columns=ws_range)

    plot1 = sns.heatmap(df_freq, ax=ax_subplots[0], annot=True, fmt=".3f", cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
    plot2 = sns.heatmap(df_damp, ax=ax_subplots[1], annot=True, fmt=".3f", cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
    # plot.set_xticklabels(plot.get_xticklabels(), rotation=65, ha='right', rotation_mode='anchor')
    ax_subplots[1].set_xlabel('Wind speed / m/s')
    ax_subplots[0].set_title('Frequency sensitivity')
    ax_subplots[1].set_title('Damping sensitivity')
    plt.tight_layout()

    # fig = plot.get_figure()
    fig_subplots.savefig(os.path.join(output_dir, '{} - {}.png'.format(name, feature)), dpi=300)

    #plt.show()
    plt.close()


def compare_oat_metrics(UQResultsAnalysis, oat_metric, ylabel, manual_feature_selection=None):
    """
    Compare oat metrics of multiple framework runs with each other. Assumed that those runs are made with the
    same uncertain parameter sets. Bar plot made with pandas.
    """
    existing_sorting_index = None
    if manual_feature_selection == None:
        desired_features = ['total'] + features
    else:
        desired_features = manual_feature_selection

    for visualization_mode in ['sorted']:  # 'unsorted',
        fig_subplots, ax_subplots = plt.subplots(num='Comparison OAT metrics - All Features - {}'.format(ylabel),
                                                 nrows=1, ncols=len(desired_features), figsize=(14, 10.5), sharex=True)

        for ifeat, feature in enumerate(desired_features):
            fig = plt.figure('Comparison OAT metrics - {} - {}'.format(feature, ylabel), figsize=(5.5, 10))
            ax1 = fig.gca()

            metric_pd_dict = dict()  # key => legend label
            color_list = list()
            for uq_result in oat_metric:
                oat_max_nans_eliminated = oat_metric[uq_result][feature]
                oat_max_nans_eliminated[np.isnan(oat_metric[uq_result][feature])] = -1
                metric_pd_dict[UQResultsAnalysis.run_label[uq_result]] = oat_max_nans_eliminated
                color_list.append(UQResultsAnalysis.run_color[uq_result])

            # CUT OFF THE SCALAR_PROPERTY FOR NOW
            index = [name[:-16] for name in UQResultsAnalysis.uncertain_param_names[uq_result]]
            index = [uncertain_param_label_dict[idx] for idx in index]
            df1 = pd.DataFrame(metric_pd_dict, index=index)
            if visualization_mode == 'sorted':
                if existing_sorting_index is None:
                    df1 = df1.sort_values(by=[UQResultsAnalysis.run_label[UQResultsAnalysis.uncertainpy_results[0]]])
                    existing_sorting_index = df1.index
                else:
                    df1 = df1.reindex(existing_sorting_index)

            if color_list[0] == None:
                df1.plot.barh(ax=ax1, legend=False)
                df1.plot.barh(ax=ax_subplots[ifeat], legend=False)
            else:
                df1.plot.barh(ax=ax1, color=color_list, legend=False)
                df1.plot.barh(ax=ax_subplots[ifeat], color=color_list, legend=False)

            # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            for ax in [ax1, ax_subplots[ifeat]]:
                ax.set_title(feat_names[feature])
                ax.grid(axis='y')
                ax.set_xlabel('EE value')
                ax.set_axisbelow(True)

            if ifeat > 0:
                ax_subplots[ifeat].tick_params(labelleft=False)

            fig.tight_layout()

            fig.savefig(os.path.join(UQResultsAnalysis.output_dir, '{} - {} - {}.png'.format(feature, ylabel, visualization_mode)), bbox_inches='tight', dpi=300)

            # plt.show(block=False)
            plt.close(fig)
        fig_subplots.tight_layout()
        fig_subplots.savefig(os.path.join(UQResultsAnalysis.output_dir, 'All features - {} - {}.png'.format(ylabel, visualization_mode)), bbox_inches='tight', dpi=300)

        # plt.show()
        plt.close(fig_subplots)


def compare_oat_metrics_with_std_bar(UQResultsAnalysis, oat_metric, oat_metric_std, ylabel, manual_feature_selection=None):
    """
    Compare oat metrics of multiple framework runs with each other. Assumed that those runs are made with the
    same uncertain parameter sets. Bar plot made with pandas.
    """
    existing_sorting_index = None
    if manual_feature_selection == None:
        desired_features = ['total'] + features
    else:
        desired_features = manual_feature_selection

    for visualization_mode in ['sorted']:  # 'unsorted',
        fig_subplots, ax_subplots = plt.subplots(num='Comparison OAT metrics - All Features - {}'.format(ylabel),
                                                 nrows=1, ncols=len(desired_features), figsize=(14, 10.5), sharex=True)

        for ifeat, feature in enumerate(desired_features):
            fig = plt.figure('Comparison OAT metrics - {} - {}'.format(feature, ylabel), figsize=(5.5, 10))
            ax1 = fig.gca()

            metric_pd_dict = dict()  # key => legend label
            metric_std_pd_dict = dict()
            color_list = list()
            for uq_result in oat_metric:
                oat_metric_nans_eliminated = oat_metric[uq_result][feature]
                oat_metric_std_nans_eliminated = oat_metric_std[uq_result][feature]

                oat_metric_nans_eliminated[np.isnan(oat_metric[uq_result][feature])] = -1
                oat_metric_std_nans_eliminated[np.isnan(oat_metric_std[uq_result][feature])] = -1

                metric_pd_dict[UQResultsAnalysis.run_label[uq_result]] = oat_metric_nans_eliminated
                metric_std_pd_dict[UQResultsAnalysis.run_label[uq_result]] = oat_metric_std_nans_eliminated

                color_list.append(UQResultsAnalysis.run_color[uq_result])

            # CUT OFF THE SCALAR_PROPERTY FOR NOW
            index = [name[:-16] for name in UQResultsAnalysis.uncertain_param_names[uq_result]]
            index = [uncertain_param_label_dict[idx] for idx in index]
            df1 = pd.DataFrame(metric_pd_dict, index=index)
            df_std = pd.DataFrame(metric_std_pd_dict, index=index)
            if visualization_mode == 'sorted':
                if existing_sorting_index is None:
                    df1 = df1.sort_values(by=[UQResultsAnalysis.run_label[UQResultsAnalysis.uncertainpy_results[0]]])
                    existing_sorting_index = df1.index
                else:
                    df1 = df1.reindex(existing_sorting_index)
                df_std = df_std.reindex(existing_sorting_index)

            if color_list[0] == None:
                df1.plot.barh(ax=ax1, xerr=df_std, legend=False, capsize=4)
                df1.plot.barh(ax=ax_subplots[ifeat], xerr=df_std, legend=False, capsize=4)
            else:
                df1.plot.barh(ax=ax1, xerr=df_std, color=color_list, legend=False, capsize=4)
                df1.plot.barh(ax=ax_subplots[ifeat], xerr=df_std, color=color_list, legend=False, capsize=4)

            # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            for ax in [ax1, ax_subplots[ifeat]]:
                ax.set_title(feat_names[feature])
                ax.grid(axis='y')
                ax.set_xlabel('EE value')
                ax.set_axisbelow(True)

            if ifeat > 0:
                ax_subplots[ifeat].tick_params(labelleft=False)

            fig.tight_layout()

            fig.savefig(os.path.join(UQResultsAnalysis.output_dir, '{} - {} - {} - with std bar.png'.format(feature, ylabel, visualization_mode)), bbox_inches='tight', dpi=300)
            # plt.show()
            # plt.show(block=False)
            plt.close(fig)
        fig_subplots.tight_layout()
        fig_subplots.savefig(os.path.join(UQResultsAnalysis.output_dir, 'All features - {} - {} - with std bar.png'.format(ylabel, visualization_mode)), bbox_inches='tight', dpi=300)

        # plt.show()
        plt.close(fig_subplots)


def campbell_diagram_from_OAT(UQResultsAnalysis, uq_results):

    nr_params = len(uq_results.uncertain_parameters)

    for param_idx in range(nr_params):

        param_name = uq_results.uncertain_parameters[param_idx]

        fig = plt.figure('Campbell Diagram OAT - Param {}'.format(param_name), figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for feature in uq_results:

            if feature == 'CampbellDiagramModel':
                continue

            freq_and_damp = uq_results[feature].evaluations[param_idx+1]
            if np.any(np.isnan(freq_and_damp)):
                continue

            ax1.plot(ws_range, freq_and_damp[:n_ws], color=feat_colors[feature])
            ax2.plot(ws_range, freq_and_damp[n_ws:], color=feat_colors[feature])

        ax1.set_xlim([min(ws_range)-1, max(ws_range)+1])
        ax1.set_ylim([0, 2.5])
        ax2.set_ylim([-1, 10])
        ax2.fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')

        # ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax1.grid()
        ax2.grid()
        ax1.set_ylabel('Frequency / Hz')
        ax2.set_ylabel('Damping Ratio / %')
        ax2.set_xlabel('Wind speed / m/s')
        fig.tight_layout()

        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'Campbell diagram {}.png'.format(param_name)), dpi=300)
        plt.close(fig)
        # plt.show()


def campbell_diagrams_from_EE(UQResultsAnalysis, uq_results):

    nr_params = len(uq_results.uncertain_parameters)
    nr_iters_per_repetition = nr_params + 1
    nr_morris_repetitions = int(len(uq_results[uq_results.model_name].evaluations) / nr_iters_per_repetition)

    for repetition in range(nr_morris_repetitions):

        fig = plt.figure('Campbell Diagram EE - Repetition {}'.format(repetition+1), figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for feature in features:

            if feature == 'CampbellDiagramModel':
                continue

            for iteration in range(nr_iters_per_repetition * repetition, nr_iters_per_repetition * (1 + repetition)):
                freq_and_damp = uq_results[feature].evaluations[iteration]
                if np.any(np.isnan(freq_and_damp)):
                    continue

                ax1.plot(ws_range, freq_and_damp[:n_ws], color=feat_colors[feature])
                ax2.plot(ws_range, freq_and_damp[n_ws:], color=feat_colors[feature])

        ax1.set_xlim([min(ws_range)-1, max(ws_range)+1])
        ax1.set_ylim([0, 2.5])
        ax2.set_ylim([-1, 5])
        ax2.fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')

        # ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax1.grid()
        ax2.grid()
        ax1.set_ylabel('Frequency / Hz')
        ax2.set_ylabel('Damping Ratio / %')
        ax2.set_xlabel('Wind speed / m/s')
        fig.tight_layout()

        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'Campbell diagram {}.png'.format(repetition)), dpi=300)
        plt.close(fig)
        # plt.show()


def campbell_diagrams_from_EE_with_rejected_with_mac_values(UQResultsAnalysis, uq_results):
    nr_params = len(uq_results.uncertain_parameters)
    nr_iters_per_repetition = nr_params + 1
    nr_morris_repetitions = int(len(uq_results[uq_results.model_name].evaluations) / nr_iters_per_repetition)

    for repetition in range(nr_morris_repetitions):

        compare_QoI_evaluations_per_iter_with_rejected_with_mac_values(UQResultsAnalysis,
                                                                       start_iteration=repetition*nr_iters_per_repetition,
                                                                       stop_iteration=repetition*nr_iters_per_repetition+nr_iters_per_repetition,
                                                                       # stop_iteration=(repetition+1)*nr_iters_per_repetition,
                                                                       plot_mac=True)


def std_vs_ee_plot(UQResultsAnalysis, uq_results):

    for feature in features:

        fig, ax = plt.subplots(figsize=(18, 10))

        for idx in range(len(uq_results.uncertain_parameters)):
            if idx < 10:
                marker = 'o'
            else:
                marker = 's'
            ax.scatter(uq_results.data[feature].ee_mean[idx], uq_results.data[feature].ee_std[idx],
                       label=uq_results.uncertain_parameters[idx], marker=marker)

        ax.set_xlabel('Mean')
        ax.set_ylabel('Standard Dev.')
        ax.grid()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'EE - {}.png'.format(feature)), bbox_inches='tight', dpi=300)
        plt.close(fig)
        # plt.show()


def campbell_diagram_from_PCE(UQResultsAnalysis, uq_results, desired_features, feat_colors, feat_names, ws_range):

    nr_ws = len(ws_range)

    fig = plt.figure('Campbell Diagram PCE', figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    for feature in desired_features:

        if 'masked_evaluations' in uq_results.data[feature]:
            evaluations = uq_results.data[feature].masked_evaluations
        else:
            evaluations = uq_results.data[feature].evaluations

        feature_mean = np.mean(evaluations, axis=0)
        feature_std = np.std(evaluations, axis=0)

        ax1.plot(ws_range, feature_mean[:nr_ws], color=feat_colors[feature], label=feat_names[feature])
        ax2.plot(ws_range, feature_mean[nr_ws:], color=feat_colors[feature])

        ax1.fill_between(ws_range,
                         y1=feature_mean[:nr_ws] - feature_std[:nr_ws],
                         y2=feature_mean[:nr_ws] + feature_std[:nr_ws],
                         where=None, facecolor=feat_colors[feature], alpha=0.5)

        ax2.fill_between(ws_range,
                         y1=feature_mean[nr_ws:] - feature_std[nr_ws:],
                         y2=feature_mean[nr_ws:] + feature_std[nr_ws:],
                         where=None, facecolor=feat_colors[feature], alpha=0.5)

    ax1.set_xlim([min(ws_range)-1, max(ws_range)+1])
    # ax1.set_ylim([0, 1 ])
    ax2.set_ylim([-1, 5])
    ax2.fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')

    ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax1.grid()
    ax2.grid()
    ax1.set_ylabel('Frequency / Hz')
    ax2.set_ylabel('Damping Ratio / %')
    ax2.set_xlabel('Wind speed / m/s')
    fig.tight_layout()

    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'Campbell diagram - PCE.png'), dpi=300)
    #plt.close(fig)
    plt.show()


def first_order_campbell_diagrams(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range):

    nr_ws = len(ws_range)
    uncertain_param_names = uq_results.uncertain_parameters

    fig, axes = plt.subplots(nrows=2, ncols=len(uncertain_param_names), num='First Order Campbell Diagram PCE',
                             figsize=(12, 8), sharex=True)

    for iter, uncertain_param_name in enumerate(uncertain_param_names):

        for feature in features:

            polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)

            samples_uncertain_param = uq_results.data[feature].nodes[iter, :]

            # scale_to_minus1_plus1 = np.linspace(-1, 1, nr_samples_plot_1D)

            # mean values of all input distributions
            print('Assumed that all distributions have mean = 0 for first order effect plotting!')
            sample_means = np.zeros((len(uncertain_param_names), len(samples_uncertain_param)))

            # replace mean value of this iter with the sample points
            sample_means[iter] = samples_uncertain_param

            evaluated_polynom = numpoly.call(polynom, sample_means)
            mean_evaluated_polynom = np.mean(evaluated_polynom, axis=1)
            std_evaluated_polynom = np.std(evaluated_polynom, axis=1)

            axes[0, iter].plot(ws_range, mean_evaluated_polynom[:nr_ws], color=feat_colors[feature], label=feat_names[feature])
            axes[1, iter].plot(ws_range, mean_evaluated_polynom[nr_ws:], color=feat_colors[feature], label=feat_names[feature])

            axes[0, iter].fill_between(ws_range,
                             y1=mean_evaluated_polynom[:nr_ws] - std_evaluated_polynom[:nr_ws],
                             y2=mean_evaluated_polynom[:nr_ws] + std_evaluated_polynom[:nr_ws],
                             where=None, facecolor=feat_colors[feature], alpha=0.5)

            axes[1, iter].fill_between(ws_range,
                             y1=mean_evaluated_polynom[nr_ws:] - std_evaluated_polynom[nr_ws:],
                             y2=mean_evaluated_polynom[nr_ws:] + std_evaluated_polynom[nr_ws:],
                             where=None, facecolor=feat_colors[feature], alpha=0.5)

        axes[0, iter].set_xlim([min(ws_range)-1, max(ws_range)+1])
        # ax1.set_ylim([0, 1 ])
        axes[1, iter].set_ylim([-1, 5])
        axes[1, iter].fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')
        axes[0, iter].grid()
        axes[1, iter].grid()
        axes[1, iter].set_xlabel('Wind speed / m/s')
        axes[0, iter].set_title(uncertain_param_name)
        if iter > 0:
            axes[0, iter].tick_params(labelleft=False)

    # axes[0, -1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    axes[0, 0].set_ylabel('Frequency / Hz')
    axes[1, 0].set_ylabel('Damping Ratio / %')

    fig.tight_layout()

    fig.subplots_adjust(bottom=0.2)  #, wspace=0.33)
    axes[1, 3].legend(loc='upper center',
                      bbox_to_anchor=(0.5, -0.2), ncol=3)  # , fancybox=False, shadow=False)

    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'first_order_campbell_diagram_pce.png'), bbox_inches='tight', dpi=300)
    plt.show()


def compare_campbell_plot(self):
    """
    Compare the frequency/damping of different runs for the different tools
    """
    fig = plt.figure('Comparison Campbell frequencies/damping', dpi=300)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for run_directory, tool_name in zip(self.run_dirs, self.tool_names):
        if tool_name == 'hawcstab2' or tool_name == 'bladed-lin':
            mode_names_test = ['test'] * self.campbell_diagram['frequency'][run_directory].shape[1]
            ax1.plot(self.campbell_diagram['frequency'][run_directory], '+', label=mode_names_test)
            ax2.plot(self.campbell_diagram['damping'][run_directory], '+')
        else:
            x_array = np.arange(self.campbell_diagram['frequency'][run_directory].shape[0])
            x_array = np.repeat(x_array.reshape(x_array.size, 1),
                                self.campbell_diagram['frequency'][run_directory].shape[1], axis=1)
            ax1.scatter(x_array, self.campbell_diagram['frequency'][run_directory],
                        s=30 * self.campbell_diagram['DMD_part_factor'][run_directory] / np.max(
                            self.campbell_diagram['DMD_part_factor'][run_directory]),
                        label=self.run_label[run_directory])
            ax2.scatter(x_array, self.campbell_diagram['damping'][run_directory],
                        s=30 * self.campbell_diagram['DMD_part_factor'][run_directory] / np.max(
                            self.campbell_diagram['DMD_part_factor'][run_directory]))

            #ax1.plot(, '+')
            #ax2.plot(self.campbell_diagram['damping'][run_directory], '+')

    # ax1.legend()

    ax1.grid()
    ax2.grid()
    ax1.set_ylim([0, 5])
    ax2.set_ylim([-5, 5])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Damping [%]')
    plt.show()


def variance_decomposition_campbell_diagrams_isolated_nonlinear_effects(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range):

    import matplotlib.gridspec as gridspec

    def plot_mean_and_variance(axx, ws_range, means, stds, feature):
        axx[0].plot(ws_range, means[:n_ws], color=feat_colors[feature], label=feat_names[feature])
        axx[1].plot(ws_range, means[n_ws:], color=feat_colors[feature], label=feat_names[feature])

        axx[0].fill_between(ws_range,
                            y1=means[:n_ws] - stds[:n_ws],
                            y2=means[:n_ws] + stds[:n_ws],
                            where=None, facecolor=feat_colors[feature], alpha=0.5)

        axx[1].fill_between(ws_range,
                            y1=means[n_ws:] - stds[n_ws:],
                            y2=means[n_ws:] + stds[n_ws:],
                            where=None, facecolor=feat_colors[feature], alpha=0.5)


    uncertain_param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]


    ### Isolated Variance each parameter
    for iter, uncertain_param_name in enumerate(uncertain_param_names):

        fig, axes = plt.subplots(2, 5, num='Variance decomposed Campbell diagram PCE',
                                 figsize=(15, 8), tight_layout=True, sharex=True)

        for feature in features:

            # uncertainpy values
            mean_all_params = uq_results.data[feature].mean

            #polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)
            #linear, quadratic, cubic, quartic = get_variance_decomposed_isolated_nonlinear_effects(polynom, 7, 50)
            variance_decomposed_isolated_nonlinear_effects = uq_results.data[feature].variance_decomposed_isolated_nonlinear_effects
            std_decomposed_isolated_nonlinear_effects = np.sqrt(variance_decomposed_isolated_nonlinear_effects)

            variance_total_isolated = np.sum(variance_decomposed_isolated_nonlinear_effects, axis=0)
            std_total_isolated = np.sqrt(variance_total_isolated)

            # directly from polynomial
            #polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)
            #samples_uncertain_param = uq_results.data[feature].nodes[iter, :]

            # mean values of all input distributions
            #print('Assumed that all distributions have mean = 0 for first order effect plotting!')
            #sample_means = np.zeros((len(uncertain_param_names), len(samples_uncertain_param)))

            # replace mean value of this iter with the sample points
            #sample_means[iter] = samples_uncertain_param

            #evaluated_polynom = numpoly.call(polynom, sample_means)
            #mean_evaluated_polynom = np.mean(evaluated_polynom, axis=1)
            #std_evaluated_polynom = np.std(evaluated_polynom, axis=1)

            plot_mean_and_variance([axes[0, 0], axes[1, 0]], ws_range, mean_all_params, std_total_isolated[iter, :], feature)

            for polynomial_order in [1, 2, 3, 4]:
                plot_mean_and_variance([axes[0, polynomial_order], axes[1, polynomial_order]], ws_range, mean_all_params, std_decomposed_isolated_nonlinear_effects[polynomial_order-1, iter, :], feature)

        titles = ['Total', 'Linear', 'Quadratic', 'Qubic', 'Quartic']

        for iter in range(5):
            axes[1, iter].fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')
            axes[0, iter].grid()
            axes[1, iter].grid()
            axes[0, iter].set_title(titles[iter])
            axes[0, iter].set_xlim([min(ws_range)-1, max(ws_range)+1])
            axes[1, iter].set_xlim([min(ws_range)-1, max(ws_range)+1])
            axes[0, iter].set_ylim([0, 2.5])
            axes[1, iter].set_ylim([-1, 5])
            axes[1, iter].set_xlabel('Wind speed / m/s')

        #if iter in [0, 3, 6]:
        axes[0, 0].set_ylabel('Frequency / Hz')
        axes[1, 0].set_ylabel('Damping Ratio / %')
        #else:
        #    freq_axes[iter].tick_params(labelleft=False)
        #    damp_axes[iter].tick_params(labelleft=False)


        fig.tight_layout()

        # fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'variance_decomposed_campbell_diagram_pce_XXX.pdf'), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def analyse_pce(UQResultsAnalysis, manual_features):
    import scipy

    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])
    uncertain_parameter_labels = [uncertain_param_label_dict[name[:-16]] for name in uq_results.uncertain_parameters]

    optimization_successes = []
    optimization_x = []
    for i_feat, feature in enumerate(manual_features):

        polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)

        ##########
        # Minimize damping
        ##########
        res = scipy.optimize.minimize(poly, np.zeros(7), bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)], args=polynom)
        optimization_successes.append(res.success)
        optimization_x.append(res.x)
        print('\nMinimum value of {} mode: {}'.format(feat_names[feature], res.fun))
        print('Optimization success:', res.success)
        print('Required uncertain parameter settings:\n')
        for iparam, param in enumerate(uncertain_parameter_labels):
            print('{}: {:.3f}'.format(param, res.x[iparam]))

        ##########
        # Minimize required modifications for instability
        ##########
        res = scipy.optimize.minimize(necessary_modifications,
                                      np.zeros(7),
                                      bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
                                      constraints={'type': 'eq', 'fun': poly, 'args': [polynom]})
        optimization_successes.append(res.success)
        optimization_x.append(res.x)

        print('\nMinimum required modifications for an unstable {} mode'.format(feat_names[feature]))
        print('Optimization success:', res.success)
        print('Required uncertain parameter settings:\n')
        for iparam, param in enumerate(uncertain_parameter_labels):
            print('{}: {:.3f}'.format(param, res.x[iparam]))

        ##########
        # Minimize required modifications for instability without cog modification
        ##########
        res = scipy.optimize.minimize(necessary_modifications,
                                      np.zeros(7),
                                      bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0), (-0.1, 0.1)],
                                      constraints={'type': 'eq', 'fun': poly, 'args': [polynom]})
        optimization_successes.append(res.success)
        optimization_x.append(res.x)

        print('\nMinimum required modifications for an unstable {} mode, without cog modification'.format(feat_names[feature]))
        print('Optimization success:', res.success)
        print('Required uncertain parameter settings:\n')
        for iparam, param in enumerate(uncertain_parameter_labels):
            print('{}: {:.3f}'.format(param, res.x[iparam]))


        ##########
        # Minimize required modifications for instability without cog and sc modification
        ##########
        res = scipy.optimize.minimize(necessary_modifications,
                                      np.zeros(7),
                                      bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0), (0.0, 0.0)],
                                      constraints={'type': 'eq', 'fun': poly, 'args': [polynom]})
        optimization_successes.append(res.success)
        optimization_x.append(res.x)

        print('\nMinimum required modifications for an unstable {} mode, without cog and sc modification'.format(feat_names[feature]))
        print('Optimization success:', res.success)
        print('Required uncertain parameter settings:\n')
        for iparam, param in enumerate(uncertain_parameter_labels):
            print('{}: {:.3f}'.format(param, res.x[iparam]))

        ##########
        # Minimize required modifications for instability without cog, sc and tors modification
        ##########
        res = scipy.optimize.minimize(necessary_modifications,
                                      np.zeros(7),
                                      bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0), (-0.1, 0.1), (0.0, 0.0), (0.0, 0.0)],
                                      constraints={'type': 'eq', 'fun': poly, 'args': [polynom]})
        optimization_successes.append(res.success)
        optimization_x.append(res.x)

        print('\nMinimum required modifications for an unstable {} mode, without cog, sc and tors modification'.format(feat_names[feature]))
        print('Optimization success:', res.success)
        print('Required uncertain parameter settings:\n')
        for iparam, param in enumerate(uncertain_parameter_labels):
            print('{}: {:.3f}'.format(param, res.x[iparam]))

    optimization_result_plot(['{} instability'.format(feat_names[manual_features[0]]),
                              '{} instability'.format(feat_names[manual_features[1]]),
                              '{} instability, without c.o.g. modification'.format(feat_names[manual_features[1]]),
                              '{} instability, without c.o.g. and shear center modification'.format(feat_names[manual_features[1]])],
                             uncertain_parameter_labels,
                             None,
                             np.array(optimization_x),
                             use_optimizations = [1, 6, 7, 8])


def poly(param_values, polynom):
    return min(numpoly.call(polynom, param_values))


def necessary_modifications(param_values):
    return np.sum(np.abs(param_values))


def optimization_result_plot(optimization_run_labels, uncertain_parameter_labels, optimization_successes, optimization_x, use_optimizations):

    selected_data = optimization_x[use_optimizations]

    length = len(uncertain_parameter_labels)

    # Set plot parameters
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    width = 0.2 # width of bar
    x = np.arange(length)

    colors = ['#000080', '#0F52BA', '#6593F5', '#73C2FB']
    for optimization_idx in range(len(use_optimizations)):
        ax.bar(x + (optimization_idx * width), selected_data[optimization_idx], width, color=colors[optimization_idx], label=optimization_run_labels[optimization_idx])

        for xval, val in zip(x + (optimization_idx * width), selected_data[optimization_idx]):
            if val < 0:
                va = 'top'
            else:
                va = 'bottom'

            if abs(val) > 0.001 and abs(val) < 0.05:
                ax.text(xval, val+(0.005*np.sign(val)), '{:.1f}%'.format(val*100), ha='center', va=va, rotation=90*np.sign(val), fontweight='bold')  #, color=colors[optimization_idx])
            elif abs(val) > 0.05:
                ax.text(xval, 0.005*np.sign(val), '{:.1f}%'.format(val*100), ha='center', va=va, rotation=90*np.sign(val), color='white', fontweight='bold')

        ax.bar(x + (optimization_idx * width), 0.1 * np.ones(selected_data[optimization_idx].shape), width, color='none', edgecolor='k', ls='--', alpha=0.5)  # , linewidth=2  , ls='--'
        ax.bar(x + (optimization_idx * width), -0.1 * np.ones(selected_data[optimization_idx].shape), width, color='none', edgecolor='k', ls='--', alpha=0.5)  # , linewidth=2  , ls='--'

    ax.bar(x + (1.5 * width), 0.1 * np.ones(selected_data[optimization_idx].shape), 4*width, color='none', edgecolor='k', alpha=1)  #, bottom=-0.1)  # , linewidth=2  , ls='--'
    ax.bar(x + (1.5 * width), -0.1 * np.ones(selected_data[optimization_idx].shape), 4*width, color='none', edgecolor='k', alpha=1)  # , linewidth=2  , ls='--'

    ax.set_ylabel('Required modification')
    ax.set_ylim(-0.12, 0.12)
    ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax.set_yticklabels(['-10%', '-5%', 'Ref.', '+5%', '+10%'])

    ax.set_xticks(x + width + width/2)
    ax.set_xticklabels(uncertain_parameter_labels)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), title='Minimum required input modifications for:')
    leg._legend_box.align = "left"
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    fig.tight_layout()
    fig.savefig(r'../../../figures/optimization_results.pdf', bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def run_hawcstab2_for_specific_uncertain_params(UQResultsAnalysis):

    tool_config = {'tool_name': 'hawcstab2',
                   'master_model_directory_path': './reference_model/hawcstab2/IEA-15-240-RWT-Onshore-master-25Points',
                   'cmb_filename': 'IEA_15MW_RWT_Onshore.cmb',
                   'amp_filename': 'IEA_15MW_RWT_Onshore.amp',
                   'opt_filename': 'IEA_15MW_RWT_Onshore_25Points.opt',
                   'main_model_htc_file': 'IEA_15MW_RWT_Onshore.htc',
                   'data_shaft_st_file': 'IEA_15MW_RWT_Shaft_st.dat',
                   'data_tower_st_file': 'IEA_15MW_RWT_Tower_st.dat',
                   'data_blade_geom_file': 'IEA_15MW_RWT_ae.dat',
                   'data_blade_st_file': 'IEA_15MW_RWT_Blade_st_noFPM.st',
                   'data_pc_file': 'IEA_15MW_RWT_pc_OpenFASTpolars_3dcorr.dat',
                   'path_to_reference_bin': './reference_model/hawcstab2/IEA-15-240-RWT-Onshore-master-25Points-reference-run/htc/IEA_15MW_RWT_Onshore.bin',
                   'path_to_reference_cmb': './reference_model/hawcstab2/IEA-15-240-RWT-Onshore-master-25Points-reference-run/htc/IEA_15MW_RWT_Onshore.cmb',
                   'n_op_points': 25,
                   'nmodes': 60,
                   'ndofs': 732,
                   'mode_indices_ref': [5, 6, 7, 11, 12, 13, 14],
                   'postpro_method': 'MAC_based_mode_picking_and_tracking_mode_specific_thresholds',
                   'minimum_MAC_mode_picking_mode_specific': [0.0, 0.0, 0.0, 0.0, 0.85, 0.82, 0.9],
                   'minimum_MAC_mode_tracking_mode_specific': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                   'minimum_MAC_mode_tracking_wrt_ref_mode_specific': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   'with_damp': False}


    #Minimum value of Rotor 2nd edgewise backward whirl A mode: -2.8123143978090566
    #Optimization success: True
    #Required uncertain parameter settings:

    #Tower bending stiffness: -0.100
    #Blade chord length (tip): 0.100
    #Blade edgewise stiffness: 0.100
    #Blade torsional stiffness: -0.100
    #Blade mass: 0.100
    #Blade cog position $\parallel$ to chord: -0.100
    #Blade shear center position $\parallel$ to chord: 0.100

    iteration_run_directory_five_minimum = os.path.join('./results', 'IEA_15MW_pce_analysis_edge_mode_five_minimum')
    preprocessed_data_five_minimum = {'tower_bending_stiffness': -0.100,
                                      'blade_chord_tip': 0.100,
                                      'blade_edge_stiffness': 0.100,
                                      'blade_tors_stiffness': -0.100,
                                      'blade_mass': 0.100,
                                      'blade_cog_x': -0.100,
                                      'blade_sc_x': 0.100 }

    #Minimum required modifications for an unstable Rotor 2nd edgewise backward whirl A mode
    #Optimization success: True
    #Required uncertain parameter settings:

    #Tower bending stiffness: 0.000
    #Blade chord length (tip): 0.000
    #Blade edgewise stiffness: 0.000
    #Blade torsional stiffness: -0.000
    #Blade mass: 0.000
    #Blade cog position $\parallel$ to chord: -0.063
    #Blade shear center position $\parallel$ to chord: 0.015

    iteration_run_directory_five_minimum_modifications = os.path.join('./results', 'IEA_15MW_pce_analysis_edge_mode_five_minimum_modifications')
    preprocessed_data_five_minimum_modifications = {'tower_bending_stiffness': 0.000,
                                      'blade_chord_tip': 0.000,
                                      'blade_edge_stiffness': 0.000,
                                      'blade_tors_stiffness': -0.000,
                                      'blade_mass': 0.000,
                                      'blade_cog_x': -0.063,
                                      'blade_sc_x': 0.015 }

    #Minimum value of Rotor 2nd edgewise backward whirl B mode: -8.740075554599038
    #Optimization success: True
    #Required uncertain parameter settings:

    #Tower bending stiffness: 0.100
    #Blade chord length (tip): 0.100
    #Blade edgewise stiffness: -0.100
    #Blade torsional stiffness: -0.100
    #Blade mass: 0.100
    #Blade cog position $\parallel$ to chord: -0.100
    #Blade shear center position $\parallel$ to chord: 0.100

    iteration_run_directory_six_minimum = os.path.join('./results', 'IEA_15MW_pce_analysis_edge_mode_six_minimum')
    preprocessed_data_six_minimum = {'tower_bending_stiffness': 0.100,
                                      'blade_chord_tip': 0.100,
                                      'blade_edge_stiffness': -0.100,
                                      'blade_tors_stiffness': -0.100,
                                      'blade_mass': 0.100,
                                      'blade_cog_x': -0.100,
                                      'blade_sc_x': 0.100 }

    #Minimum required modifications for an unstable Rotor 2nd edgewise backward whirl B mode
    #Optimization success: True
    #Required uncertain parameter settings:

    #Tower bending stiffness: 0.000
    #Blade chord length (tip): 0.000
    #Blade edgewise stiffness: 0.000
    #Blade torsional stiffness: 0.000
    #Blade mass: 0.026
    #Blade cog position $\parallel$ to chord: -0.045
    #Blade shear center position $\parallel$ to chord: 0.002

    iteration_run_directory_six_minimum_modifications= os.path.join('./results', 'IEA_15MW_pce_analysis_edge_mode_six_minimum_modifications')
    preprocessed_data_six_minimum_modifications = {'tower_bending_stiffness': 0.000,
                                      'blade_chord_tip': 0.000,
                                      'blade_edge_stiffness': 0.000,
                                      'blade_tors_stiffness': 0.000,
                                      'blade_mass': 0.026,
                                      'blade_cog_x': -0.045,
                                      'blade_sc_x': 0.002 }

    #Minimum required modifications for an unstable Rotor 2nd edgewise backward whirl B mode, without cog modification
    #Optimization success: True
    #Required uncertain parameter settings:

    #Tower bending stiffness: 0.011
    #Blade chord length (tip): 0.006
    #Blade edgewise stiffness: -0.000
    #Blade torsional stiffness: -0.100
    #Blade mass: 0.052
    #Blade cog position $\parallel$ to chord: 0.000
    #Blade shear center position $\parallel$ to chord: 0.013

    iteration_run_directory_six_minimum_modifications_wo_cog = os.path.join('./results', 'IEA_15MW_pce_analysis_edge_mode_six_minimum_modifications_wo_cog')
    preprocessed_data_six_minimum_modifications_wo_cog = {'tower_bending_stiffness': 0.011,
                                      'blade_chord_tip': 0.006,
                                      'blade_edge_stiffness': -0.000,
                                      'blade_tors_stiffness': -0.100,
                                      'blade_mass': 0.052,
                                      'blade_cog_x': 0.000,
                                      'blade_sc_x': 0.013 }

    #Minimum required modifications for an unstable Rotor 2nd edgewise backward whirl B mode, without cog and sc modification
    #Optimization success: True
    #Required uncertain parameter settings:

    #Tower bending stiffness: 0.010
    #Blade chord length (tip): 0.047
    #Blade edgewise stiffness: 0.000
    #Blade torsional stiffness: -0.100
    #Blade mass: 0.037
    #Blade cog position $\parallel$ to chord: 0.000
    #Blade shear center position $\parallel$ to chord: 0.000

    iteration_run_directory_six_minimum_modifications_wo_cog_sc = os.path.join('./results', 'IEA_15MW_pce_analysis_edge_mode_six_minimum_modifications_wo_cog_sc')
    preprocessed_data_six_minimum_modifications_wo_cog_sc = {'tower_bending_stiffness': 0.010,
                                      'blade_chord_tip': 0.047,
                                      'blade_edge_stiffness': 0.000,
                                      'blade_tors_stiffness': -0.100,
                                      'blade_mass': 0.037,
                                      'blade_cog_x': 0.000,
                                      'blade_sc_x': 0.000 }

    #Minimum required modifications for an unstable Rotor 2nd edgewise backward whirl B mode, without cog, sc and tors modification
    #Optimization success: True
    #Required uncertain parameter settings:

    #Tower bending stiffness: 0.100
    #Blade chord length (tip): 0.052
    #Blade edgewise stiffness: -0.040
    #Blade torsional stiffness: 0.000
    #Blade mass: 0.100
    #Blade cog position $\parallel$ to chord: 0.000
    #Blade shear center position $\parallel$ to chord: 0.000

    iteration_run_directory_six_minimum_modifications_wo_cog_sc_tors = os.path.join('./results', 'IEA_15MW_pce_analysis_edge_mode_six_minimum_modifications_wo_cog_sc_tors')
    preprocessed_data_six_minimum_modifications_wo_cog_sc_tors = {'tower_bending_stiffness': 0.100,
                                      'blade_chord_tip': 0.052,
                                      'blade_edge_stiffness': -0.040,
                                      'blade_tors_stiffness': 0.000,
                                      'blade_mass': 0.100,
                                      'blade_cog_x': 0.000,
                                      'blade_sc_x': 0.000 }


    from wtuq_framework.helperfunctions import save_dict_h5py

    preprocessed_datas = [preprocessed_data_five_minimum,
                          preprocessed_data_five_minimum_modifications,
                          preprocessed_data_six_minimum,
                          preprocessed_data_six_minimum_modifications,
                          preprocessed_data_six_minimum_modifications_wo_cog,
                          preprocessed_data_six_minimum_modifications_wo_cog_sc,
                          preprocessed_data_six_minimum_modifications_wo_cog_sc_tors]
    run_dirs = [iteration_run_directory_five_minimum,
                iteration_run_directory_five_minimum_modifications,
                iteration_run_directory_six_minimum,
                iteration_run_directory_six_minimum_modifications,
                iteration_run_directory_six_minimum_modifications_wo_cog,
                iteration_run_directory_six_minimum_modifications_wo_cog_sc,
                iteration_run_directory_six_minimum_modifications_wo_cog_sc_tors]
    names = ['five_minimum',
             'five_minimum_modifications',
             'six_minimum',
             'six_minimum_modifications',
             'six_minimum_modifications_wo_cog',
             'six_minimum_modifications_wo_cog_sc',
             'six_minimum_modifications_wo_cog_sc_tors']

    for preprocessed_data, iteration_run_directory, name in zip(preprocessed_datas, run_dirs, names):
        simulation_tool = HAWCStab2Model(iteration_run_directory, tool_config)

        use_result_dict = True
        if use_result_dict is False:
            simulation_tool.create_simulation(preprocessed_data)
            simulation_tool.run_simulation(None)
            result_dict = simulation_tool.extract_results()
            save_dict_h5py(os.path.join(iteration_run_directory, 'result_dict.hdf5'), {key: value for key, value in result_dict.items() if key != 'cv_bladedlin_data'})
        else:
            result_dict = load_dict_h5py(os.path.join(iteration_run_directory, 'result_dict.hdf5'))

        model_inputs = {'run_directory': None,
                        'restart_directory': None,
                        'restart_h5': None,
                        'run_type': None}

        from run_analysis import CampbellDiagramModel
        test = CampbellDiagramModel(tool_config, dict(), model_inputs)
        test.simulation_tool = simulation_tool
        success, campbell_dict = test._postprocessor_hs2(result_dict)
        print(success)
        for mode_ii in range(7):
            print(campbell_dict['frequency'][:, mode_ii], campbell_dict['damping'][:, mode_ii])

        uncertain_params_correct_order = ['tower_bending_stiffness', 'blade_chord_tip', 'blade_edge_stiffness', 'blade_tors_stiffness',
                                          'blade_mass', 'blade_cog_x', 'blade_sc_x']
        nodes = [preprocessed_data[param] for param in uncertain_params_correct_order]

        fig = plt.figure('Reference Campbell Diagram', figsize=(9, 4.5))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        ax1.plot([], [], color='none', label=r'$\bf{HAWCStab2}$ $\bf{results}$')

        for feat_ii, feature in enumerate(features):

            ax1.plot(ws_range, campbell_dict['frequency'][:, feat_ii], color=feat_colors[feature], lw=2, label=feat_names[feature])
            ax2.plot(ws_range, campbell_dict['damping'][:, feat_ii], color=feat_colors[feature], lw=2, label=feat_names[feature])

        ax1.plot([], [], color='none', label=r'$\bf{PCE}$ $\bf{model}$ $\bf{predictions}$')

        for feat_ii, feature in enumerate(features):
            polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)
            evaluated_polynom = numpoly.call(polynom, nodes).T

            ax1.plot(ws_range, evaluated_polynom[:n_ws], '--', color=feat_colors[feature], lw=2, label=feat_names[feature])
            ax2.plot(ws_range, evaluated_polynom[n_ws:], '--', color=feat_colors[feature], lw=2, label=feat_names[feature])

        ax1.set_xlim([min(ws_range)-1, max(ws_range)+1])

        # ax2.set_ylim([-5, 10])
        ax1.set_ylim([0, 2.5])
        ax2.set_ylim([-1, 5])
        ax2.fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')

        ax1.grid()
        ax2.grid()
        ax1.set_ylabel('Frequency / Hz')
        ax2.set_ylabel('Damping Ratio / %')
        ax2.set_xlabel('Wind speed / m/s')
        fig.tight_layout()
        fig.subplots_adjust(right=0.6)  #, wspace=0.33)
        ax1.legend(bbox_to_anchor=(1.02, 1.04), loc="upper left")

        fig.savefig(r'../../../figures/campbell_diagram_pce_analysis_studies_{}.pdf'.format(name), bbox_inches='tight', dpi=600)
        # plt.show()
        plt.close()


def oat_plot_paper(UQResultsAnalysis, oat_metric, ylabel):
    """
    Compare oat metrics of multiple framework runs with each other. Assumed that those runs are made with the
    same uncertain parameter sets. Bar plot made with pandas.
    """
    desired_features = ['total'] + features
    existing_sorting_index = None

    fig_subplots, ax_subplots = plt.subplots(num='OAT Plot Paper - {}'.format(ylabel),
                                             nrows=1, ncols=len(desired_features)-1, figsize=(12, 9), sharex=True)

    for ifeat, feature in enumerate(desired_features):

        metric_pd_dict = dict()  # key => legend label
        for uq_result in oat_metric:
            oat_max_nans_eliminated = oat_metric[uq_result][feature]
            oat_max_nans_eliminated[np.isnan(oat_metric[uq_result][feature])] = -1
            metric_pd_dict[UQResultsAnalysis.run_label[uq_result]] = oat_max_nans_eliminated

        # CUT OFF THE SCALAR_PROPERTY FOR NOW
        index = [name[:-16] for name in UQResultsAnalysis.uncertain_param_names[uq_result]]
        index = [uncertain_param_label_dict[idx] for idx in index]
        df1 = pd.DataFrame(metric_pd_dict, index=index)

        if existing_sorting_index is None:
            df1 = df1.sort_values(by=[UQResultsAnalysis.run_label[UQResultsAnalysis.uncertainpy_results[0]]])
            existing_sorting_index = df1.index
        else:
            df1 = df1.reindex(existing_sorting_index)

        if ifeat > 0:  # 0 = total -> just used for sorting
            df1.plot.barh(ax=ax_subplots[ifeat-1], legend=False)

            ax_subplots[ifeat-1].set_title(extra_short_feat_names[feature])
            ax_subplots[ifeat-1].grid(axis='y')
            ax_subplots[ifeat-1].set_xlabel('$\mathregular{S_{OAT/EE}}$')
            ax_subplots[ifeat-1].set_axisbelow(True)

            if ifeat > 1:
                ax_subplots[ifeat-1].tick_params(labelleft=False)

    fig_subplots.tight_layout()
    fig_subplots.savefig(os.path.join(UQResultsAnalysis.output_dir, 'oat_plot_paper_{}.png'.format(ylabel)), bbox_inches='tight', dpi=600)

    # plt.show()
    plt.close(fig_subplots)


def combined_oat_ee_plot_paper(UQResultsAnalysis_OAT, UQResultsAnalysis_EE, extra_label=''):
    uq_results_oat = UQResultsAnalysis_OAT.get_UQ_results(UQResultsAnalysis_OAT.uncertainpy_results[0])
    uq_results_ee = UQResultsAnalysis_EE.get_UQ_results(UQResultsAnalysis_EE.uncertainpy_results[0])

    oat_max = dict()
    oat_mean = dict()

    for feature in uq_results_oat.data:
        if feature == 'CampbellDiagramModel':
            continue
        oat_max[feature] = uq_results_oat.data[feature].s_oat_max
        oat_mean[feature] = uq_results_oat.data[feature].s_oat_mean

    oat_max['total'] = np.max(np.vstack(([oat_max[feat] for feat in features])), axis=0)
    oat_mean['total'] = np.mean(np.vstack(([oat_mean[feat] for feat in features])), axis=0)

    ee_mean_max = dict()
    ee_mean_mean = dict()
    ee_std_max = dict()
    ee_std_mean = dict()

    for feature in uq_results_ee.data:
        if feature == 'CampbellDiagramModel':
            continue
        ee_mean_max[feature] = uq_results_ee.data[feature].ee_mean_max
        ee_mean_mean[feature] = uq_results_ee.data[feature].ee_mean_mean
        ee_std_max[feature] = uq_results_ee.data[feature].ee_std_max
        ee_std_mean[feature] = uq_results_ee.data[feature].ee_std_mean

    ee_mean_max['total'] = np.max(np.vstack(([ee_mean_max[feat] for feat in features])), axis=0)
    ee_std_max['total'] = np.max(np.vstack(([ee_std_max[feat] for feat in features])), axis=0)
    ee_mean_mean['total'] = np.mean(np.vstack(([ee_mean_mean[feat] for feat in features])), axis=0)
    ee_std_mean['total'] = np.mean(np.vstack(([ee_std_mean[feat] for feat in features])), axis=0)

    for label, oat_metric, ee_metric, ee_metric_std in zip(['mean', 'max'], [oat_mean, oat_max], [ee_mean_mean, ee_mean_max], [ee_std_mean, ee_std_max]):
        fig_subplots, ax_subplots = plt.subplots(num='Comparison OAT and EE metrics - All Features - {}'.format(label),
                                                 nrows=1, ncols=len(features), figsize=(10, 12), sharex=True)

        existing_sorting_index = None
        desired_features = ['total'] + features
        for ifeat, feature in enumerate(desired_features):

            metric_pd_dict = dict()  # key => legend label
            std_pd_dict = dict()

            oat_mean_nans_eliminated = oat_metric[feature]
            oat_mean_nans_eliminated[np.isnan(oat_metric[feature])] = -1
            metric_pd_dict['OAT'] = oat_mean_nans_eliminated
            std_pd_dict['OAT'] = np.zeros(oat_mean_nans_eliminated.shape)  # no standard deviation in OAT

            ee_mean_mean_nans_eliminated = ee_metric[feature]
            ee_mean_mean_nans_eliminated[np.isnan(ee_metric[feature])] = -1
            ee_values_all_oat_params = np.zeros(oat_mean_nans_eliminated.shape)
            ee_std_values_all_oat_params = np.zeros(oat_mean_nans_eliminated.shape)
            indices = [uq_results_oat.uncertain_parameters.index(ee_uncertain_param) for ee_uncertain_param in uq_results_ee.uncertain_parameters]
            ee_values_all_oat_params[indices] = ee_mean_mean_nans_eliminated
            ee_std_values_all_oat_params[indices] = ee_metric_std[feature]  # NANS are not checked for the STD here.
            metric_pd_dict['EE'] = ee_values_all_oat_params
            std_pd_dict['EE'] = ee_std_values_all_oat_params

            # CUT OFF THE SCALAR_PROPERTY FOR NOW
            index = [name[:-16] for name in uq_results_oat.uncertain_parameters]
            index = [uncertain_param_label_dict[idx] for idx in index]
            df1 = pd.DataFrame(metric_pd_dict, index=index)
            df_errors = pd.DataFrame(std_pd_dict, index=index)
            sorted_metrics = True
            if sorted_metrics is True:
                if existing_sorting_index is None:
                    df1 = df1.sort_values(by=['OAT'])
                    existing_sorting_index = df1.index
                else:
                    df1 = df1.reindex(existing_sorting_index)
                df_errors = df_errors.reindex(existing_sorting_index)

            if ifeat > 0:  # not plotting total

                df1.plot.barh(ax=ax_subplots[ifeat-1], xerr=df_errors, legend=False, error_kw=dict(lw=1), color=['#114360', '#faccfa'])  # , yerr=df_errors

                ax_subplots[ifeat-1].set_title(extra_short_feat_names[feature])
                ax_subplots[ifeat-1].grid(axis='y')
                ax_subplots[ifeat-1].set_xlabel('$\mathregular{S_{OAT/EE}}$')
                ax_subplots[ifeat-1].set_axisbelow(True)

                if ifeat > 1:
                    ax_subplots[ifeat-1].tick_params(labelleft=False)

        h, l = ax_subplots[-1].get_legend_handles_labels()
        h.append(matplotlib.lines.Line2D([0], [0], color='k', lw=1))
        l.append('EE std')

        ax_subplots[-1].legend(handles=h, labels=l, loc='lower right')


        ax_subplots[0].axhline(xmin=-3, xmax=9, color='k', y=34.5, zorder=0, clip_on=False)
        for ax in ax_subplots[1:]:
            ax.axhline(xmin=0, xmax=1, color='k', y=34.5, zorder=0, clip_on=False)  #'C3'

        remember_xlim = ax_subplots[-1].get_xlim()
        ax_subplots[-1].arrow(1.4, 35, 0, 2.5, color='k', head_length=0.4, head_width=0.1, clip_on=False)  #color='C2'
        ax_subplots[-1].arrow(1.4, 34, 0, -2.5, color='k', head_length=0.4, head_width=0.1, clip_on=False)  #color='C3'

        ax_subplots[-1].text(1.5, 35, 'Included in EE analysis', rotation=270, va='bottom', clip_on=False)
        ax_subplots[-1].text(1.5, 34, 'Excluded from EE analysis', rotation=270, ha='left', va='top', clip_on=False)

        ax_subplots[-1].set_xlim((0, remember_xlim[1]))

        plt.tight_layout()
        fig_subplots.savefig(r'../../../figures/combined_oat_ee_plot - {}.pdf'.format(label), bbox_inches='tight', dpi=600)
        plt.close(fig_subplots)
        #plt.show()


def combined_oat_ee_plot_paper_shortened(UQResultsAnalysis_OAT, UQResultsAnalysis_EE, extra_label=''):
    uq_results_oat = UQResultsAnalysis_OAT.get_UQ_results(UQResultsAnalysis_OAT.uncertainpy_results[0])
    uq_results_ee = UQResultsAnalysis_EE.get_UQ_results(UQResultsAnalysis_EE.uncertainpy_results[0])

    oat_max = dict()
    oat_mean = dict()

    for feature in uq_results_oat.data:
        if feature == 'CampbellDiagramModel':
            continue
        oat_max[feature] = uq_results_oat.data[feature].s_oat_max
        oat_mean[feature] = uq_results_oat.data[feature].s_oat_mean

    oat_max['total'] = np.max(np.vstack(([oat_max[feat] for feat in features])), axis=0)
    oat_mean['total'] = np.mean(np.vstack(([oat_mean[feat] for feat in features])), axis=0)

    ee_mean_max = dict()
    ee_mean_mean = dict()
    ee_std_max = dict()
    ee_std_mean = dict()

    for feature in uq_results_ee.data:
        if feature == 'CampbellDiagramModel':
            continue
        ee_mean_max[feature] = uq_results_ee.data[feature].ee_mean_max
        ee_mean_mean[feature] = uq_results_ee.data[feature].ee_mean_mean
        ee_std_max[feature] = uq_results_ee.data[feature].ee_std_max
        ee_std_mean[feature] = uq_results_ee.data[feature].ee_std_mean

    ee_mean_max['total'] = np.max(np.vstack(([ee_mean_max[feat] for feat in features])), axis=0)
    ee_std_max['total'] = np.max(np.vstack(([ee_std_max[feat] for feat in features])), axis=0)
    ee_mean_mean['total'] = np.mean(np.vstack(([ee_mean_mean[feat] for feat in features])), axis=0)
    ee_std_mean['total'] = np.mean(np.vstack(([ee_std_mean[feat] for feat in features])), axis=0)

    for label, oat_metric, ee_metric, ee_metric_std in zip(['mean', 'max'], [oat_mean, oat_max], [ee_mean_mean, ee_mean_max], [ee_std_mean, ee_std_max]):
        fig_subplots, ax_subplots = plt.subplots(num='Comparison OAT and EE metrics - All Features - {}'.format(label),
                                                 nrows=1, ncols=len(features), figsize=(8, 11), sharex=True)

        existing_sorting_index = None
        desired_features = ['total'] + features
        for ifeat, feature in enumerate(desired_features):

            metric_pd_dict = dict()  # key => legend label
            std_pd_dict = dict()

            oat_mean_nans_eliminated = oat_metric[feature]
            oat_mean_nans_eliminated[np.isnan(oat_metric[feature])] = -1
            metric_pd_dict['OAT'] = oat_mean_nans_eliminated
            std_pd_dict['OAT'] = np.zeros(oat_mean_nans_eliminated.shape)  # no standard deviation in OAT

            ee_mean_mean_nans_eliminated = ee_metric[feature]
            ee_mean_mean_nans_eliminated[np.isnan(ee_metric[feature])] = -1
            ee_values_all_oat_params = np.zeros(oat_mean_nans_eliminated.shape)
            ee_std_values_all_oat_params = np.zeros(oat_mean_nans_eliminated.shape)
            indices = [uq_results_oat.uncertain_parameters.index(ee_uncertain_param) for ee_uncertain_param in uq_results_ee.uncertain_parameters]
            ee_values_all_oat_params[indices] = ee_mean_mean_nans_eliminated
            ee_std_values_all_oat_params[indices] = ee_metric_std[feature]  # NANS are not checked for the STD here.
            metric_pd_dict['EE'] = ee_values_all_oat_params
            std_pd_dict['EE'] = ee_std_values_all_oat_params

            # CUT OFF THE SCALAR_PROPERTY FOR NOW
            index = [name[:-16] for name in uq_results_oat.uncertain_parameters]
            index = [uncertain_param_label_dict[idx] for idx in index]
            df1 = pd.DataFrame(metric_pd_dict, index=index)
            df_errors = pd.DataFrame(std_pd_dict, index=index)
            sorted_metrics = True
            if sorted_metrics is True:
                if existing_sorting_index is None:
                    df1 = df1.sort_values(by=['OAT'])
                    existing_sorting_index = df1.index
                else:
                    df1 = df1.reindex(existing_sorting_index)
                df_errors = df_errors.reindex(existing_sorting_index)

            if ifeat > 0:  # not plotting total

                df1.plot.barh(ax=ax_subplots[ifeat-1], xerr=df_errors, legend=False, error_kw=dict(lw=1), color=['#114360', '#faccfa'])  # , yerr=df_errors

                ax_subplots[ifeat-1].set_title(extra_short_feat_names[feature])
                ax_subplots[ifeat-1].grid(axis='y')
                ax_subplots[ifeat-1].set_xlabel('$\mathregular{S_{OAT/EE}}$')
                ax_subplots[ifeat-1].set_axisbelow(True)

                if ifeat > 1:
                    ax_subplots[ifeat-1].tick_params(labelleft=False)

        text = 'XXXXXXXXXXXXXXXXXXXXAll remaining parameters had an even smaller OAT sensitivity and are not visualized to save space.XXXX'
        t = ax_subplots[-1].text(-14, 29.5, text, va='bottom', clip_on=False, color='white')
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))

        text = 'All remaining parameters had an even smaller OAT sensitivity and are not visualized to save space.'
        t = ax_subplots[-1].text(-10, 29.5, text, va='bottom', clip_on=False)
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))

        h, l = ax_subplots[-1].get_legend_handles_labels()
        h.append(matplotlib.lines.Line2D([0], [0], color='k', lw=1))
        l.append('EE std')

        ax_subplots[-1].legend(handles=h, labels=l, loc='lower right', bbox_to_anchor=(1.05, 0.67), fontsize=9)


        ax_subplots[0].axhline(xmin=-2.9, xmax=8.93, color='k', y=34.5, zorder=0, clip_on=False)
        ax_subplots[0].axhline(xmin=-2.9, xmax=8.58, color='k', y=38.5, zorder=0, clip_on=False)
        for ax in ax_subplots[1:]:
            ax.axhline(xmin=0, xmax=1, color='k', y=34.5, zorder=0, clip_on=False)  #'C3'
            ax.axhline(xmin=0, xmax=1, color='k', y=38.5, zorder=0, clip_on=False)  #'C3'

        remember_xlim = ax_subplots[-1].get_xlim()
        ax_subplots[-1].arrow(1.85, 35, 0, 2.5, color='k', head_length=0.4, head_width=0.1, clip_on=False)  #color='C2'
        #ax_subplots[-1].arrow(1.4, 34, 0, -2.5, color='k', head_length=0.4, head_width=0.1, clip_on=False)  #color='C3'

        ax_subplots[-1].arrow(1.4, 39, 0, 2.5, color='k', head_length=0.4, head_width=0.1, clip_on=False)  #color='C2'
        #ax_subplots[-1].arrow(1.4, 34, 0, -2.5, color='k', head_length=0.4, head_width=0.1, clip_on=False)  #color='C3'

        ax_subplots[-1].text(1.95, 35, 'Included in EE', rotation=270, va='bottom', clip_on=False)
        #ax_subplots[-1].text(1.5, 34, 'Excluded', rotation=270, ha='left', va='top', clip_on=False)

        ax_subplots[-1].text(1.5, 39, 'Included in PCE', rotation=270, va='bottom', clip_on=False)
        #ax_subplots[-1].text(1.5, 34, 'Excluded', rotation=270, ha='left', va='top', clip_on=False)

        ax_subplots[-1].set_xlim((0, remember_xlim[1]))

        plt.tight_layout()
        fig_subplots.savefig(r'../../../figures/combined_oat_ee_plot - {} - shortened.pdf'.format(label), bbox_inches='tight', dpi=600)
        plt.close(fig_subplots)
        #plt.show()


def ee_plot_paper_mode_5_6(UQResultsAnalysis):
    """
    Compare oat metrics of multiple framework runs with each other. Assumed that those runs are made with the
    same uncertain parameter sets. Bar plot made with pandas.
    """
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])
    UQResultsAnalysis.uncertain_param_names[UQResultsAnalysis.uncertainpy_results[0]] = uq_results.uncertain_parameters

    # Mean of EE damping edge mode 5 & 6
    ee_mean_damp_mode_5_6 = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_damp_mode_5_6 = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    print('Last 8 indices in ee -> damping > 15 m/s')
    for feature in ['edge_mode_five', 'edge_mode_six']:
        ee_mean_damp_mode_5_6[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.mean(uq_results.data[feature].ee_mean[:, -8:], axis=1)
        ee_std_damp_mode_5_6[UQResultsAnalysis.uncertainpy_results[0]][feature] = np.mean(uq_results.data[feature].ee_std[:, -8:], axis=1)

    oat_metric = ee_mean_damp_mode_5_6
    std_metric = ee_std_damp_mode_5_6
    manual_features = ['edge_mode_five', 'edge_mode_six']

    fig_subplots, ax_subplots = plt.subplots(num='Comparison OAT metrics - Mode 5 - 6',
                                             nrows=1, ncols=2, figsize=(10, 5), sharex=True)

    for ifeat, feature in enumerate(manual_features):

        metric_pd_dict = dict()  # key => legend label
        metric_std_pd_dict = dict()
        for uq_result in oat_metric:
            oat_mean_nans_eliminated = oat_metric[uq_result][feature]
            oat_std_nans_eliminated = std_metric[uq_result][feature]

            oat_mean_nans_eliminated[np.isnan(oat_metric[uq_result][feature])] = -1
            oat_std_nans_eliminated[np.isnan(std_metric[uq_result][feature])] = -1

            metric_pd_dict[UQResultsAnalysis.run_label[uq_result]] = oat_mean_nans_eliminated
            metric_std_pd_dict[UQResultsAnalysis.run_label[uq_result]] = oat_std_nans_eliminated

        # CUT OFF THE SCALAR_PROPERTY FOR NOW
        index = [name[:-16] for name in UQResultsAnalysis.uncertain_param_names[uq_result]]
        index = [uncertain_param_label_dict[idx] for idx in index]
        df1 = pd.DataFrame(metric_pd_dict, index=index)
        df_std = pd.DataFrame(metric_std_pd_dict, index=index)

        df1 = df1.sort_values(by=[UQResultsAnalysis.run_label[UQResultsAnalysis.uncertainpy_results[0]]])
        existing_sorting_index = df1.index
        df_std = df_std.reindex(existing_sorting_index)

        df1.plot.barh(ax=ax_subplots[ifeat], xerr=df_std, color='#faccfa', legend=False)

        ax_subplots[ifeat].set_title(feat_names[feature])
        ax_subplots[ifeat].grid(axis='y')
        ax_subplots[ifeat].set_xlabel('$\mathregular{S^{EE}}$')
        ax_subplots[ifeat].set_axisbelow(True)

    fig_subplots.tight_layout()
    fig_subplots.savefig(os.path.join(UQResultsAnalysis.output_dir, 'ee_mean_damping_mode_5_6.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig_subplots)


def surrogate_model_verification_plot_paper(UQResultsAnalysis, uq_results, features, ws_range):
    fig, ax = plt.subplots(nrows=2, ncols=len(features), figsize=(9.5, 3))

    # a big subplot for a common ylabel (https://stackoverflow.com/questions/6963035/how-to-set-common-axes-labels-for-subplots)
    ax_big = fig.add_subplot(111, frameon=False)    # The big subplot
    ax_big.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_big.grid(False)
    ax_big.set_ylabel('Leave-one-out surrogate \n model evaluations')
    ax_big.set_xlabel('Model evaluations (training data)')
    ax_big.yaxis.labelpad = -15
    ax_big.xaxis.labelpad = -10

    for i_feat, feature in enumerate(features):

        evaluations_loo = uq_results.data[feature].evaluations_loo
        nr_nodes_for_loo = evaluations_loo.shape[0]
        evaluations = uq_results.data[feature].masked_evaluations

        ax[0, i_feat].scatter(evaluations[:nr_nodes_for_loo, :n_ws], evaluations_loo[:nr_nodes_for_loo, :n_ws], color='#828231', s=3, alpha=0.5, label='Frequency')
        ax[1, i_feat].scatter(evaluations[:nr_nodes_for_loo, n_ws:], evaluations_loo[:nr_nodes_for_loo, n_ws:], color='#f29d6d', s=3, alpha=0.5, label='Damping')

        ax[0, i_feat].grid(True)
        ax[1, i_feat].grid(True)
        ax[0, i_feat].set_axisbelow(True)
        ax[1, i_feat].set_axisbelow(True)

        ax[0, i_feat].tick_params(axis='both', direction='in', labelbottom=False, labelleft=False)
        ax[1, i_feat].tick_params(axis='both', direction='in', labelbottom=False, labelleft=False)

        ax[0, i_feat].set_title(extra_short_feat_names[feature])

    ax[0, 0].legend(loc='upper left', markerscale=3, fontsize=7.5)
    ax[1, 0].legend(loc='upper left', markerscale=3, fontsize=7.5)

    plt.tight_layout()
    fig.savefig(r'../../../figures/surrogate_model_verification_paper_plot.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(r'../../../figures/surrogate_model_verification_paper_plot.png', bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def sobol_plot_paper(UQResultsAnalysis, uq_results):

    param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]
    for feature in features:

        fig_subplots, ax_subplots = plt.subplots(num='Sobol indices - {}'.format(feature),
                                                 nrows=2, ncols=2, figsize=(12, 3.5), sharex=True)

        arr_first_order_freq, arr_first_order_damp = np.split(uq_results.data[feature].sobol_first_manual, 2, axis=1)
        arr_total_freq, arr_total_damp = np.split(uq_results.data[feature].sobol_total_manual, 2, axis=1)

        df_freq_first_order = pd.DataFrame(arr_first_order_freq, index=param_names, columns=ws_range)
        df_damp_first_order = pd.DataFrame(arr_first_order_damp, index=param_names, columns=ws_range)
        df_freq_total = pd.DataFrame(arr_total_freq, index=param_names, columns=ws_range)
        df_damp_total = pd.DataFrame(arr_total_damp, index=param_names, columns=ws_range)

        plot1 = sns.heatmap(df_freq_first_order, ax=ax_subplots[0, 0], cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
        plot2 = sns.heatmap(df_damp_first_order, ax=ax_subplots[1, 0], cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
        plot3 = sns.heatmap(df_freq_total, ax=ax_subplots[0, 1], cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
        plot4 = sns.heatmap(df_damp_total, ax=ax_subplots[1, 1], cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)

        # plot.set_xticklabels(plot.get_xticklabels(), rotation=65, ha='right', rotation_mode='anchor')
        ax_subplots[1, 0].set_xlabel('Wind speed / m/s')
        ax_subplots[1, 1].set_xlabel('Wind speed / m/s')

        ax_subplots[0, 0].set_title('First order Sobol index on frequency')
        ax_subplots[1, 0].set_title('First order Sobol index on damping')
        ax_subplots[0, 1].set_title('Total Sobol index on frequency')
        ax_subplots[1, 1].set_title('Total Sobol index on damping')

        ax_subplots[0, 1].tick_params(labelleft=False)
        ax_subplots[1, 1].tick_params(labelleft=False)

        plt.tight_layout()

        # fig_subplots.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=sns.cm.rocket_r), ax=ax_subplots.ravel().tolist())
        fig_subplots.subplots_adjust(right=0.94)
        cbar_ax = fig_subplots.add_axes([0.95, 0.2, 0.02, 0.7])
        fig_subplots.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=sns.cm.rocket_r), cax=cbar_ax)

        # fig = plot.get_figure()
        fig_subplots.savefig(os.path.join(UQResultsAnalysis.output_dir, 'sobol_indices_paper_{}.pdf'.format(feature)), dpi=300)

        # plt.show()
        plt.close()


def variance_decomposition_campbell_diagrams_with_nonlinearity_paper(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range):

    import matplotlib.gridspec as gridspec

    def plot_mean_and_variance(axx, ws_range, means, stds, feature):
        axx[0].plot(ws_range, means[:n_ws], color=feat_colors[feature], label=short_feat_names[feature] + ' (mean)')
        axx[1].plot(ws_range, means[n_ws:], color=feat_colors[feature], label=short_feat_names[feature] + ' (mean)')

        axx[0].fill_between(ws_range,
                            y1=means[:n_ws] - stds[:n_ws],
                            y2=means[:n_ws] + stds[:n_ws],
                            where=None, facecolor=feat_colors[feature], alpha=0.5,
                            label=short_feat_names[feature] + ' (std)')

        axx[1].fill_between(ws_range,
                            y1=means[n_ws:] - stds[n_ws:],
                            y2=means[n_ws:] + stds[n_ws:],
                            where=None, facecolor=feat_colors[feature], alpha=0.5,
                            label=short_feat_names[feature] + ' (std)')


    def plot_mean_and_variance_linear_nonlinear(axx, ws_range, means, stds_only_linear, stds_total, feature):
        axx[0].plot(ws_range, means[:n_ws], color=feat_colors[feature], label=short_feat_names[feature] + ' (mean)')
        axx[1].plot(ws_range, means[n_ws:], color=feat_colors[feature], label=short_feat_names[feature] + ' (mean)')

        axx[0].fill_between(ws_range,
                            y1=means[:n_ws] - stds_total[:n_ws],
                            y2=means[:n_ws] + stds_total[:n_ws],
                            where=None, facecolor='k', alpha=0.8,
                            label=short_feat_names[feature] + ' all (std)')

        axx[1].fill_between(ws_range,
                            y1=means[n_ws:] - stds_total[n_ws:],
                            y2=means[n_ws:] + stds_total[n_ws:],
                            where=None, facecolor='k', alpha=0.8,
                            label=short_feat_names[feature] + ' all (std)')

        axx[0].fill_between(ws_range,
                            y1=means[:n_ws] - stds_only_linear[:n_ws],
                            y2=means[:n_ws] + stds_only_linear[:n_ws],
                            where=None, facecolor=feat_colors[feature], alpha=1,
                            label=short_feat_names[feature] + ' only linear (std)')

        axx[1].fill_between(ws_range,
                            y1=means[n_ws:] - stds_only_linear[n_ws:],
                            y2=means[n_ws:] + stds_only_linear[n_ws:],
                            where=None, facecolor=feat_colors[feature], alpha=1,
                            label=short_feat_names[feature] + ' only linear (std)')




    uncertain_param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]

    #fig, axes = plt.subplots(nrows=6, ncols=3, num='Variance decomposed Campbell diagram PCE',
    #                         figsize=(15, 15), sharex=True)

    fig = plt.figure(num='Variance decomposed Campbell diagram PCE',
                     figsize=(14, 13), tight_layout=True)
    gs = gridspec.GridSpec(23, 3)

    freq_axes = [fig.add_subplot(gs[0:3, 0]), fig.add_subplot(gs[0:3, 1]), fig.add_subplot(gs[0:3, 2]),
                 fig.add_subplot(gs[7:10, 0]), fig.add_subplot(gs[7:10, 1]), fig.add_subplot(gs[7:10, 2]),
                 fig.add_subplot(gs[14:17, 0]), fig.add_subplot(gs[14:17, 1]), fig.add_subplot(gs[14:17, 2])]
    damp_axes = [fig.add_subplot(gs[3:6, 0]), fig.add_subplot(gs[3:6, 1]), fig.add_subplot(gs[3:6, 2]),
                 fig.add_subplot(gs[10:13, 0]), fig.add_subplot(gs[10:13, 1]), fig.add_subplot(gs[10:13, 2]),
                 fig.add_subplot(gs[17:20, 0]), fig.add_subplot(gs[17:20, 1]), fig.add_subplot(gs[17:20, 2])]

    for ax in freq_axes:
        ax.sharex(freq_axes[0])
    for ax in damp_axes:
        ax.sharex(freq_axes[0])

    ### Total Variance
    for feature in features:

        # uncertainpy values
        mean_all_params = uq_results.data[feature].mean
        variance_all_params = uq_results.data[feature].variance
        std_al_params = np.sqrt(variance_all_params)

        # determine variance from polynomial
        #polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)
        #samples_uncertain_param = uq_results.data[feature].nodes
        #evaluated_polynom = numpoly.call(polynom, samples_uncertain_param)
        #mean_evaluated_polynom = np.mean(evaluated_polynom, axis=1)
        #std_evaluated_polynom = np.std(evaluated_polynom, axis=1)
        #var_evaluated_polynom = np.var(evaluated_polynom, axis=1)
        #print(var_evaluated_polynom, uq_results.data[feature].variance)

        plot_mean_and_variance([freq_axes[0], damp_axes[0]], ws_range, mean_all_params, std_al_params, feature)


    ### Isolated Variance each parameter
    for iter, uncertain_param_name in enumerate(uncertain_param_names):

        for feature in features:

            mean_all_params = uq_results.data[feature].mean

            variance_decomposed_isolated_nonlinear_effects = uq_results.data[feature].variance_decomposed_isolated_nonlinear_effects
            std_decomposed_isolated_nonlinear_effects = np.sqrt(variance_decomposed_isolated_nonlinear_effects)

            variance_first_order_only_linear = variance_decomposed_isolated_nonlinear_effects[0, iter, :]
            std_first_order_only_linear = np.sqrt(variance_first_order_only_linear)

            variance_first_order = np.sum(variance_decomposed_isolated_nonlinear_effects[:, iter, :], axis=0)
            std_first_order = np.sqrt(variance_first_order)

            plot_mean_and_variance_linear_nonlinear([freq_axes[iter+1], damp_axes[iter+1]], ws_range, mean_all_params, std_first_order_only_linear, std_first_order, feature)


    ### Variance contribution interaction
    for feature in features:

        mean_all_params = uq_results.data[feature].mean
        variance_all_params = uq_results.data[feature].variance
        sobol_first_all_params = []

        for iter, uncertain_param_name in enumerate(uncertain_param_names):

            print('sobol first manual used instead of uncertainpy sobol')

            sobol_first_all_params.append(uq_results.data[feature].sobol_first_manual[iter, :])

        sobol_interaction = 1 - np.sum(sobol_first_all_params, axis=0)
        variance_interaction = sobol_interaction * variance_all_params
        std_interaction = np.sqrt(variance_interaction)
        plot_mean_and_variance([freq_axes[-1], damp_axes[-1]], ws_range, mean_all_params, std_interaction, feature)


    titles = ['Total'] + uncertain_param_names + ['Interaction']
    iter = 0
    for iter in range(9):
        damp_axes[iter].fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')
        freq_axes[iter].grid()
        damp_axes[iter].grid()
        freq_axes[iter].set_title(titles[iter])
        freq_axes[iter].set_xlim([min(ws_range)-1, max(ws_range)+1])
        damp_axes[iter].set_xlim([min(ws_range)-1, max(ws_range)+1])
        freq_axes[iter].set_ylim([0, 2.5])
        damp_axes[iter].set_ylim([-1, 5.1])

        if iter in [0, 3, 6]:
            freq_axes[iter].set_ylabel('Frequency / Hz')
            damp_axes[iter].set_ylabel('Damping / %')
        else:
            freq_axes[iter].tick_params(labelleft=False)
            damp_axes[iter].tick_params(labelleft=False)

        freq_axes[iter].tick_params(labelbottom=False)
        if iter > 5:
            damp_axes[iter].set_xlabel('Wind speed / m/s')
        else:
            damp_axes[iter].tick_params(labelbottom=False)


    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.05)

    #fig.subplots_adjust(bottom=0.2)  #, wspace=0.33)
    damp_axes[7].legend(loc='upper center',
                        bbox_to_anchor=(0.5, -0.35), ncol=4)  # , fancybox=False, shadow=False)

    fig.savefig(r'../../../figures/variance_decomposed_campbell_diagram_pce_std_linearity_effects.pdf', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


def variance_decomposition_campbell_diagrams_paper(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range):

    import matplotlib.gridspec as gridspec

    def plot_mean_and_variance(axx, ws_range, means, stds, feature):
        axx[0].plot(ws_range, means[:n_ws], color=feat_colors[feature], label=short_feat_names[feature] + ' (mean)')
        axx[1].plot(ws_range, means[n_ws:], color=feat_colors[feature], label=short_feat_names[feature] + ' (mean)')

        axx[0].fill_between(ws_range,
                            y1=means[:n_ws] - stds[:n_ws],
                            y2=means[:n_ws] + stds[:n_ws],
                            where=None, facecolor=feat_colors[feature], alpha=0.5,
                            label=short_feat_names[feature] + ' (std)')

        axx[1].fill_between(ws_range,
                            y1=means[n_ws:] - stds[n_ws:],
                            y2=means[n_ws:] + stds[n_ws:],
                            where=None, facecolor=feat_colors[feature], alpha=0.5,
                            label=short_feat_names[feature] + ' (std)')


    uncertain_param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]

    #fig, axes = plt.subplots(nrows=6, ncols=3, num='Variance decomposed Campbell diagram PCE',
    #                         figsize=(15, 15), sharex=True)

    fig = plt.figure(num='Variance decomposed Campbell diagram PCE',
                     figsize=(14, 13), tight_layout=True)
    gs = gridspec.GridSpec(23, 3)

    freq_axes = [fig.add_subplot(gs[0:3, 0]), fig.add_subplot(gs[0:3, 1]), fig.add_subplot(gs[0:3, 2]),
                 fig.add_subplot(gs[7:10, 0]), fig.add_subplot(gs[7:10, 1]), fig.add_subplot(gs[7:10, 2]),
                 fig.add_subplot(gs[14:17, 0]), fig.add_subplot(gs[14:17, 1]), fig.add_subplot(gs[14:17, 2])]
    damp_axes = [fig.add_subplot(gs[3:6, 0]), fig.add_subplot(gs[3:6, 1]), fig.add_subplot(gs[3:6, 2]),
                 fig.add_subplot(gs[10:13, 0]), fig.add_subplot(gs[10:13, 1]), fig.add_subplot(gs[10:13, 2]),
                 fig.add_subplot(gs[17:20, 0]), fig.add_subplot(gs[17:20, 1]), fig.add_subplot(gs[17:20, 2])]

    for ax in freq_axes:
        ax.sharex(freq_axes[0])
    for ax in damp_axes:
        ax.sharex(freq_axes[0])

    ### Total Variance
    for feature in features:

        # uncertainpy values
        mean_all_params = uq_results.data[feature].mean
        variance_all_params = uq_results.data[feature].variance
        std_al_params = np.sqrt(variance_all_params)

        # determine variance from polynomial
        #polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)
        #samples_uncertain_param = uq_results.data[feature].nodes
        #evaluated_polynom = numpoly.call(polynom, samples_uncertain_param)
        #mean_evaluated_polynom = np.mean(evaluated_polynom, axis=1)
        #std_evaluated_polynom = np.std(evaluated_polynom, axis=1)
        #var_evaluated_polynom = np.var(evaluated_polynom, axis=1)
        #print(var_evaluated_polynom, uq_results.data[feature].variance)

        plot_mean_and_variance([freq_axes[0], damp_axes[0]], ws_range, mean_all_params, std_al_params, feature)


    ### Isolated Variance each parameter
    for iter, uncertain_param_name in enumerate(uncertain_param_names):

        for feature in features:

            # uncertainpy values
            mean_all_params = uq_results.data[feature].mean
            sobol_first = uq_results.data[feature].sobol_first[iter, :]
            variance_all_params = uq_results.data[feature].variance
            first_order_variance = sobol_first * variance_all_params
            first_order_std = np.sqrt(first_order_variance)

            # directly from polynomial
            #polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)
            #samples_uncertain_param = uq_results.data[feature].nodes[iter, :]

            # mean values of all input distributions
            #print('Assumed that all distributions have mean = 0 for first order effect plotting!')
            #sample_means = np.zeros((len(uncertain_param_names), len(samples_uncertain_param)))

            # replace mean value of this iter with the sample points
            #sample_means[iter] = samples_uncertain_param

            #evaluated_polynom = numpoly.call(polynom, sample_means)
            #mean_evaluated_polynom = np.mean(evaluated_polynom, axis=1)
            #std_evaluated_polynom = np.std(evaluated_polynom, axis=1)

            plot_mean_and_variance([freq_axes[iter+1], damp_axes[iter+1]], ws_range, mean_all_params, first_order_std, feature)

    ### Variance contribution interaction
    for feature in features:

        mean_all_params = uq_results.data[feature].mean
        variance_all_params = uq_results.data[feature].variance
        sobol_first_all_params = []

        for iter, uncertain_param_name in enumerate(uncertain_param_names):

            sobol_first_all_params.append(uq_results.data[feature].sobol_first[iter, :])

        sobol_interaction = 1 - np.sum(sobol_first_all_params, axis=0)
        variance_interaction = sobol_interaction * variance_all_params
        std_interaction = np.sqrt(variance_interaction)
        plot_mean_and_variance([freq_axes[-1], damp_axes[-1]], ws_range, mean_all_params, std_interaction, feature)




    titles = ['Total'] + uncertain_param_names + ['Interaction']
    iter = 0
    for iter in range(9):
        damp_axes[iter].fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')
        freq_axes[iter].grid()
        damp_axes[iter].grid()
        freq_axes[iter].set_title(titles[iter])
        freq_axes[iter].set_xlim([min(ws_range)-1, max(ws_range)+1])
        damp_axes[iter].set_xlim([min(ws_range)-1, max(ws_range)+1])
        freq_axes[iter].set_ylim([0, 2.5])
        damp_axes[iter].set_ylim([-1, 5.1])

        if iter in [0, 3, 6]:
            freq_axes[iter].set_ylabel('Frequency / Hz')
            damp_axes[iter].set_ylabel('Damping / %')
        else:
            freq_axes[iter].tick_params(labelleft=False)
            damp_axes[iter].tick_params(labelleft=False)

        freq_axes[iter].tick_params(labelbottom=False)
        if iter > 5:
            damp_axes[iter].set_xlabel('Wind speed / m/s')
        else:
            damp_axes[iter].tick_params(labelbottom=False)


    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.05)

    #fig.subplots_adjust(bottom=0.2)  #, wspace=0.33)
    damp_axes[7].legend(loc='upper center',
                        bbox_to_anchor=(0.5, -0.35), ncol=4)  # , fancybox=False, shadow=False)

    fig.savefig(r'../../../figures/variance_decomposed_campbell_diagram_pce_std.pdf', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


def first_order_effects_paper(UQResultsAnalysis, uq_results, features, ws_range):

    uncertain_param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]
    show_op_points_indices = [1, 7, 17, 21, 24]

    for feature in features:
        polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)

        fig, axes = plt.subplots(nrows=2, ncols=len(show_op_points_indices), num='First order effects', figsize=(15, 5), sharex=True)
        #fig.tight_layout()

        #for col in range(np.shape(axes)[1]):
        #    axes[0, col].sharey(axes[0, 0])
        #    axes[1, col].sharey(axes[1, 1])

        for iter, uncertain_param_name in enumerate(uncertain_param_names):

            samples = uq_results.data[feature].nodes[iter, :]

            # scale_to_minus1_plus1 = np.linspace(-1, 1, nr_samples_plot_1D)

            # mean values of all input distributions
            print('Assumed that all distributions have mean = 0 for first order effect plotting!')
            sample_means = np.zeros((len(uncertain_param_names), len(samples)))

            # replace mean value of this iter with the sample points
            sample_means[iter] = samples

            evaluated_polynom = numpoly.call(polynom, sample_means)

            for ii, i_op_point in enumerate(show_op_points_indices):
                axes[0, ii].plot(samples, evaluated_polynom[i_op_point, :], '.', label=uncertain_param_name)
                axes[1, ii].plot(samples, evaluated_polynom[i_op_point+n_ws, :], '.', label=uncertain_param_name)

            #sample_means[iter] = samples_plot
            #evaluated_polynom = numpoly.call(U_hat[model_name], sample_means)
            #ax.plot(samples_plot, evaluated_polynom, label=uncertain_param_name + ': sampling within bounds training data')

            #ax2.plot(scale_to_minus1_plus1, evaluated_polynom, label=uncertain_param_name)



        for ii, i_op_point in enumerate(show_op_points_indices):
            axes[0, ii].set_title('wind speed = {} m/s'.format(ws_range[i_op_point]))

            axes[0, ii].grid(True)
            axes[1, ii].grid(True)

        axes[0, 0].set_ylabel('Frequency / Hz')
        axes[1, 0].set_ylabel('Damping / %')

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.23)  #, wspace=0.33)
        axes[1, 2].legend(loc='upper center',
                          bbox_to_anchor=(0.5, -0.2), ncol=3)  # , fancybox=False, shadow=False)


        # plt.tight_layout()
        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'first_order_effects_paper_{}.pdf'.format(feature)), bbox_inches='tight', dpi=300)
        # fig2.savefig(os.path.join(self.output_dir, 'first_order_effects_scaled_to_-1_to_1.png'), bbox_inches='tight')
        # plt.show()
        plt.close()


def reference_campbell_diagram_paper():

    fig = plt.figure('Reference Campbell Diagram', figsize=(9, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    picked_modes = [5, 6, 7, 11, 12, 13, 14]
    nmodes_shown = 19
    feature_ii = 0

    for mode_ii in range(19):

        if mode_ii in picked_modes:
            feature = features[feature_ii]
            ax1.plot(ws_range, result_dict['frequency'][mode_ii, :], color=feat_colors[feature], lw=1.5, marker='o', markersize=3, label=feat_names[feature])
            ax2.plot(ws_range, result_dict['damping'][mode_ii, :], color=feat_colors[feature], lw=1.5, marker='o', markersize=3, label=feat_names[feature])
            feature_ii += 1
        elif mode_ii == nmodes_shown - 1:  # add one label for the grey lines
            ax1.plot(ws_range, result_dict['frequency'][mode_ii, :], color='grey', lw=0.8, alpha=0.5, label='Excluded modes')
            ax2.plot(ws_range, result_dict['damping'][mode_ii, :], color='grey', lw=0.8, alpha=0.5)
        else:
            ax1.plot(ws_range, result_dict['frequency'][mode_ii, :], color='grey', lw=0.8, alpha=0.5)
            ax2.plot(ws_range, result_dict['damping'][mode_ii, :], color='grey', lw=0.8, alpha=0.5)


    # ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax1.set_xlim([min(ws_range)-1, max(ws_range)+1])
    ax1.set_ylim([0, 2.5])
    ax2.set_ylim([-1, 5])
    ax2.fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')

    ax1.grid()
    ax2.grid()
    ax1.set_ylabel('Frequency / Hz')
    ax2.set_ylabel('Damping Ratio / %')
    ax2.set_xlabel('Wind speed / m/s')
    fig.tight_layout()
    fig.subplots_adjust(right=0.6)  #, wspace=0.33)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.savefig(r'../../../figures/reference_campbell_diagram.pdf', bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def campbell_diagrams_percentiles_paper(UQResultsAnalysis, uq_results, features):

    fig = plt.figure('Campbell Diagram percentiles', figsize=(9, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    for feature in features:

        ax1.plot(ws_range, uq_results[feature].mean[:n_ws], color=feat_colors[feature], label=feat_names[feature])
        ax2.plot(ws_range, uq_results[feature].mean[n_ws:], color=feat_colors[feature], label=feat_names[feature])

        ax1.fill_between(ws_range,
                         y1=uq_results[feature].percentile_5[:n_ws],
                         y2=uq_results[feature].percentile_95[:n_ws],
                         where=None, facecolor=feat_colors[feature], alpha=0.5, label='5% - 95% percentile')

        ax2.fill_between(ws_range,
                         y1=uq_results[feature].percentile_5[n_ws:],
                         y2=uq_results[feature].percentile_95[n_ws:],
                         where=None, facecolor=feat_colors[feature], alpha=0.5)

    ax1.set_xlim([min(ws_range)-1, max(ws_range)+1])
    ax1.set_ylim([0, 2.5])
    ax2.set_ylim([-1, 5])
    ax2.fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')

    ax1.grid()
    ax2.grid()
    ax1.set_ylabel('Frequency / Hz')
    ax2.set_ylabel('Damping Ratio / %')
    ax2.set_xlabel('Wind speed / m/s')
    fig.tight_layout()
    fig.subplots_adjust(right=0.6)  #, wspace=0.33)
    ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.savefig(r'../../../figures/campbell_diagram_percentiles.pdf', bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def pce_plots_paper(UQResultsAnalysis):
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])
    for feature in features:
        print('Feature {}: {} / {} samples used'.format(feature,
                                                        len(uq_results.data[feature].masked_evaluations),
                                                        len(uq_results.data[feature].evaluations)))

    for feature in features:
        print('Feature {}: Average frequency MAE = {}, average damping MAE = {}'.format(feature,
                                                                                        np.mean(uq_results.data[feature].MAE_loo[:25]),
                                                                                        np.mean(uq_results.data[feature].MAE_loo[25:])))

    #sobol_plot_paper(UQResultsAnalysis, uq_results)

    surrogate_model_verification_plot_paper(UQResultsAnalysis, uq_results, features, ws_range)

    features_pce_analyse = ['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four', 'edge_mode_five', 'edge_mode_six', 'edge_mode_seven']

    campbell_diagrams_percentiles_paper(UQResultsAnalysis, uq_results, features_pce_analyse)

    variance_decomposition_campbell_diagrams_paper(UQResultsAnalysis, uq_results, features_pce_analyse, feat_colors, feat_names, ws_range)
    variance_decomposition_campbell_diagrams_with_nonlinearity_paper(UQResultsAnalysis, uq_results, features_pce_analyse, feat_colors, feat_names, ws_range)

    first_order_effects_paper(UQResultsAnalysis, uq_results, features_pce_analyse, ws_range)


def generate_paper_plots():

    # input('You are going to overwrite all plots of the Torque paper. Continue?')

    reference_campbell_diagram_paper()

    ###############
    ### OAT results
    ###############
    UQResultsAnalysis_OAT = UQResultsAnalysis([r'./results/IEA_15MW_oat_5E-3/uq_results/CampbellDiagramModel.h5'],
                                              ['hawcstab2'])

    ##############
    ### EE results
    ##############
    UQResultsAnalysis_EE = UQResultsAnalysis([r'./results/morris_convergence_study_5E-3_best-practice_skip61/IEA_15MW_morris_5E-3_50/uq_results/CampbellDiagramModel.h5'],
                                             ['hawcstab2'])
    ee_plot_paper_mode_5_6(UQResultsAnalysis_EE)
    combined_oat_ee_plot_paper_shortened(UQResultsAnalysis_OAT, UQResultsAnalysis_EE, extra_label='')

    ###############
    ### PCE results
    ###############
    uq_plots = UQResultsAnalysis([r'./results/IEA_15MW_pce_final-selection_best-practice/uq_results/CampbellDiagramModel.h5'],
                                 ['hawcstab2'])
    run_hawcstab2_for_specific_uncertain_params(uq_plots)
    analyse_pce(uq_plots, ['edge_mode_five', 'edge_mode_six'])
    pce_plots_paper(uq_plots)


if __name__ == '__main__':

    generate_paper_plots()

    ###############
    ### OAT results
    ###############
    #UQResultsAnalysis_OAT = UQResultsAnalysis([<path>/CampbellDiagramModel.h5'],
    #                                          ['hawcstab2'])
    #result_analysis_OAT(UQResultsAnalysis_OAT)

    ##############
    ### EE results
    ##############
    #UQResultsAnalysis_EE = UQResultsAnalysis([r'<path>/CampbellDiagramModel.h5'],
    #                                         ['hawcstab2'])
    #result_analysis_EE(UQResultsAnalysis_EE)

    ###############
    ### PCE results
    ###############
    #uq_plots = UQResultsAnalysis([r'<path>/CampbellDiagramModel.h5'],
    #                             ['hawcstab2'])
    #result_analysis_PCE(uq_plots)

    """
    for mac_op0 in [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.85, 0.75, 0.65, 0.55]:
        for mac_hs2 in [['nodamp', 0.7]]:
            for polynomial_order in [4]:
                for n_collocation_nodes in [1000]:
                    for mac_last_op in [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.85, 0.75, 0.65, 0.55]:

                        run_name = '{}_{}_{}_{}_{}_{}'.format(mac_op0, mac_hs2[1], mac_last_op, mac_hs2[0], polynomial_order, n_collocation_nodes)

                        uq_plots = UQResultsAnalysis([r'./results/pce_postpro_parameter_study/IEA_15MW_pce_7params2_{}/uq_results/CampbellDiagramModel.h5'.format(run_name)],
                                                     ['hawcstab2'])
                        uq_plots.run_name = run_name

                        result_analysis_PCE(uq_plots)
    """



