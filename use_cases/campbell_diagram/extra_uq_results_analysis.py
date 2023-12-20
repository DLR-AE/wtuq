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
                              'tower_damping_1': 'Damping 1st tower mode',
                              'tower_damping_2': 'Damping 2nd tower mode',
                              'blade_mass': 'Blade mass',
                              'blade_edge_stiffness': 'Blade edgewise stiffness',
                              'blade_flap_stiffness': 'Blade flapwise stiffness',
                              'blade_torsional_stiffness': 'Blade torsional stiffness',
                              'blade_pa_orientation': 'Blade principal axis orientation',
                              'blade_cog_x': r'Blade cog position $\parallel$ to chord',
                              'blade_cog_y': r'Blade cog position $\bot$ to chord',
                              'blade_sc_x': r'Blade shear center position $\parallel$ to chord',
                              'blade_sc_y': r'Blade shear center position $\bot$ to chord',
                              'blade_na_x': r'Blade prebend',
                              'blade_na_y': r'Blade sweep',
                              'blade_chord_tip': 'Blade chord length (tip)',
                              'blade_chord_root': 'Blade chord length (root)',
                              'blade_twist': 'Blade twist angle',
                              'blade_damping_1': 'Damping 1st blade mode',
                              'blade_damping_2': 'Damping 2nd blade mode',
                              'blade_damping_3': 'Damping 3rd blade mode',
                              'blade_damping_4': 'Damping 4th blade mode',
                              'cone_angle': 'Cone angle',
                              'polar_clalpha': r'Polar gradient $Cl_{\alpha}$',
                              'polar_alpha_max': r'Polar max. $\alpha$',
                              'polar_cd0': r'Polar $Cd_{0}$',
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
                              'ds_taupre': 'Dyn. stall param taupre',
                              'ds_taubly': 'Dyn. stall param taubly'}


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
    # mode_names = [mode_name.decode("ascii", "ignore") for mode_name in uq_results.data['CampbellDiagramModel'].mode_names[0 ,:].tolist()]

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

    oat_max[UQResultsAnalysis.uncertainpy_results[0]]['total'] = np.max(np.vstack((oat_max[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_bw'],
                                                                                   oat_max[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_fw'],
                                                                                   oat_max[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_bw'],
                                                                                   oat_max[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_fw'])), axis=0)
    oat_mean[UQResultsAnalysis.uncertainpy_results[0]]['total'] = np.mean(np.vstack((oat_mean[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_bw'],
                                                                                     oat_mean[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_fw'],
                                                                                     oat_mean[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_bw'],
                                                                                     oat_mean[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_fw'])), axis=0)

    compare_oat_metrics(UQResultsAnalysis, oat_max, ylabel='max. OAT metric')
    compare_oat_metrics(UQResultsAnalysis, oat_mean, ylabel='mean OAT metric')

    campbell_diagram_from_OAT(UQResultsAnalysis, uq_results)


def result_analysis_EE(UQResultsAnalysis):
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    UQResultsAnalysis.uncertain_param_names[UQResultsAnalysis.uncertainpy_results[0]] = uq_results.uncertain_parameters

    campbell_diagram_from_EE(UQResultsAnalysis, uq_results)

    ee_mean_max = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_mean_mean = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_max = {UQResultsAnalysis.uncertainpy_results[0]: {}}
    ee_std_mean = {UQResultsAnalysis.uncertainpy_results[0]: {}}

    for feature in uq_results.data:
        ee_mean_max[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_mean_max
        ee_mean_mean[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_mean_mean
        ee_std_max[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_std_max
        ee_std_mean[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].ee_std_mean

    for dicti in [ee_mean_max[UQResultsAnalysis.uncertainpy_results[0]],
                  ee_std_max[UQResultsAnalysis.uncertainpy_results[0]]]:
        dicti['total'] = np.max(
            np.vstack((dicti['first_edge_bw'],
                       dicti['first_edge_fw'],
                       dicti['second_edge_bw'],
                       dicti['second_edge_fw'])), axis=0)

    for dicti in [ee_mean_mean[UQResultsAnalysis.uncertainpy_results[0]],
                  ee_std_mean[UQResultsAnalysis.uncertainpy_results[0]]]:
        dicti['total'] = np.mean(
            np.vstack((dicti['first_edge_bw'],
                       dicti['first_edge_fw'],
                       dicti['second_edge_bw'],
                       dicti['second_edge_fw'])), axis=0)

    compare_oat_metrics(UQResultsAnalysis, ee_mean_max, ylabel='max. EE mean')
    compare_oat_metrics(UQResultsAnalysis, ee_mean_mean, ylabel='mean EE mean')
    compare_oat_metrics(UQResultsAnalysis, ee_std_max, ylabel='max. EE std')
    compare_oat_metrics(UQResultsAnalysis, ee_std_mean, ylabel='mean EE std')



    for feature in uq_results.data:

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


def result_analysis_PCE(UQResultsAnalysis):
    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    features = ['first_edge_bw', 'first_edge_fw', 'second_edge_bw', 'second_edge_fw']
    feat_colors = {'first_edge_bw': 'C0',
                   'first_edge_fw': 'C1',
                   'second_edge_bw': 'C3',
                   'second_edge_fw': 'C4'}
    feat_names = {'first_edge_bw': 'Rotor 1st edgewise backward whirl',
                  'first_edge_fw': 'Rotor 1st edgewise forward whirl',
                  'second_edge_bw': 'Rotor 2nd edgewise backward whirl',
                  'second_edge_fw': 'Rotor 2nd edgewise forward whirl'}

    # ws_range = np.linspace(3, 25, 12)
    ws_range = np.array([3, 5, 7, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 17,
                         19, 21, 23, 25])

    compare_QoI_evaluations_per_iter_with_mac_values(UQResultsAnalysis, uq_results, features, ws_range)
    compare_macs(UQResultsAnalysis, uq_results, features, ws_range)
    sobol_index_plots2(UQResultsAnalysis, uq_results, ws_range)
    # first_order_effects(UQResultsAnalysis, uq_results, features, ws_range)
    compare_QoI_evaluations_per_iter(UQResultsAnalysis, uq_results, features, ws_range)
    surrogate_model_verification(UQResultsAnalysis, uq_results, features, ws_range)

    campbell_diagram_from_PCE(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range)
    first_order_campbell_diagrams(UQResultsAnalysis, uq_results, features, feat_colors, feat_names, ws_range)


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

        plt.show()


def compare_QoI_evaluations_per_iter(UQResultsAnalysis, uq_results, features, ws_range):

    fig1 = plt.figure('Comparison QoI evaluations')
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

        evaluations_loo = uq_results.data[feature].evaluations_loo
        nr_nodes_for_loo = evaluations_loo.shape[0]
        if 'masked_evaluations' in uq_results.data[feature]:
            evaluations = uq_results.data[feature].masked_evaluations
        else:
            evaluations = uq_results.data[feature].evaluations

        n_ws = len(ws_range)

        ax1.plot(ws_range, evaluations[:nr_nodes_for_loo, :n_ws].T, markersize=10, label=feature)
        ax2.plot(ws_range, evaluations[:nr_nodes_for_loo, n_ws:].T, markersize=10, label=feature)

        #ax1.plot(ws_range, evaluations_loo[:nr_nodes_for_loo, :n_ws].T, linestyle='--', markersize=10, label=feature)
        #ax2.plot(ws_range, evaluations_loo[:nr_nodes_for_loo, n_ws:].T, linestyle='--', markersize=10, label=feature)

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig1.savefig(os.path.join(UQResultsAnalysis.output_dir, 'QoI_evaluations.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig1)


def compare_QoI_evaluations_per_iter_with_mac_values(UQResultsAnalysis, uq_results, features, ws_range):

    fig1 = plt.figure('Comparison QoI evaluations')
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    plt.subplots_adjust(hspace=0)
    ax1.set_ylabel('Frequency in Hz')
    ax2.set_xlabel('Wind speed in m/s')
    ax2.set_ylabel('Damping ratio in %')

    n_ws = len(ws_range)

    from matplotlib import colors
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    # cmap = plt.cm.cool
    # cmap = sns.color_palette("coolwarm", as_cmap=True)
    color_norm = colors.Normalize(vmin=0.8, vmax=0.95)

    for feature in features:

        if 'masked_evaluations' in uq_results.data[feature]:
            evaluations = uq_results.data[feature].masked_evaluations
        else:
            evaluations = uq_results.data[feature].evaluations

        ax1.plot(ws_range, evaluations[:, :n_ws].T, markersize=10, label=feature)
        ax2.plot(ws_range, evaluations[:, n_ws:].T, markersize=10, label=feature)

        mac = uq_results.data[uq_results.model_name]['mac_{}'.format(feature)]

        for mac_eval_ii, ws in enumerate(ws_range[1:]):
            for sample_ii, freq_eval in enumerate(evaluations[:, mac_eval_ii+1]):
                ax1.text(ws, freq_eval, '{:.2f}'.format(mac[sample_ii, mac_eval_ii]), fontsize='small', weight="bold",
                         ha='center', va='center', color=cmap(color_norm(mac[sample_ii, mac_eval_ii])))

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig1.savefig(os.path.join(UQResultsAnalysis.output_dir, 'QoI_evaluations_with_mac.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig1)


def surrogate_model_verification(UQResultsAnalysis, uq_results, features, ws_range):
    fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(12, 8))

    for i_feat, feature in enumerate(features):

        evaluations_loo = uq_results.data[feature].evaluations_loo
        nr_nodes_for_loo = evaluations_loo.shape[0]
        if 'masked_evaluations' in uq_results.data[feature]:
            evaluations = uq_results.data[feature].masked_evaluations
        else:
            evaluations = uq_results.data[feature].evaluations

        n_ws = len(ws_range)

        axes[i_feat].scatter(evaluations[:nr_nodes_for_loo, :n_ws], evaluations_loo[:nr_nodes_for_loo, :n_ws], s=3, alpha=0.5, label='Frequency')
        axes[i_feat].scatter(evaluations[:nr_nodes_for_loo, n_ws:], evaluations_loo[:nr_nodes_for_loo, n_ws:], s=3, alpha=0.5, label='Damping')

        axes[i_feat].set_ylabel('Leave-one-out surrogate model evaluations')
        axes[i_feat].grid(True)
        axes[i_feat].set_xlabel('Model evaluations (training data)')
        axes[i_feat].set_axisbelow(True)
        axes[i_feat].legend(loc='upper left', markerscale=4)

    plt.tight_layout()
    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'surrogate_model_verification.png'), bbox_inches='tight', dpi=300)
    plt.show()


def first_order_effects(UQResultsAnalysis, uq_results, features, ws_range):

    uncertain_param_names = uq_results.uncertain_parameters
    nr_op_points = len(ws_range)

    for feature in features:
        polynom = cp.load(os.path.join(UQResultsAnalysis.output_dir, feature+'.npy'), allow_pickle=True)

        fig, axes = plt.subplots(nrows=2, ncols=nr_op_points, num='First order effects', figsize=(22, 7))
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

            for i_op_point in range(nr_op_points):
                axes[0, i_op_point].plot(samples, evaluated_polynom[i_op_point, :], '.', label=uncertain_param_name)
                axes[1, i_op_point].plot(samples, evaluated_polynom[i_op_point+nr_op_points, :], '.', label=uncertain_param_name)

            #sample_means[iter] = samples_plot
            #evaluated_polynom = numpoly.call(U_hat[model_name], sample_means)
            #ax.plot(samples_plot, evaluated_polynom, label=uncertain_param_name + ': sampling within bounds training data')

            #ax2.plot(scale_to_minus1_plus1, evaluated_polynom, label=uncertain_param_name)

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

    for feature in uq_results.data:
        if feature == uq_results.model_name:  # and uq_results.model_ignore == True:
            continue

        for which_sobol, sobol_arr in zip(['First order Sobol index', 'Total Sobol index'],
                                          [uq_results.data[feature].sobol_first_manual, uq_results.data[feature].sobol_total_manual]):

            fig_subplots, ax_subplots = plt.subplots(num='{} - {} - vertical'.format(which_sobol, feature),
                                                     nrows=2, ncols=1, figsize=(10, 10), sharex=True)

            param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]
            sobol_arr_freq, sobol_arr_damp = np.split(sobol_arr, 2, axis=1)
            df_freq = pd.DataFrame(sobol_arr_freq, index=param_names, columns=ws_range)
            df_damp = pd.DataFrame(sobol_arr_damp, index=param_names, columns=ws_range)

            plot1 = sns.heatmap(df_freq, ax=ax_subplots[0], annot=True, fmt=".3f", cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
            plot2 = sns.heatmap(df_damp, ax=ax_subplots[1], annot=True, fmt=".3f", cbar=False, cmap=sns.cm.rocket_r, vmin=0, vmax=1)
            # plot.set_xticklabels(plot.get_xticklabels(), rotation=65, ha='right', rotation_mode='anchor')
            ax_subplots[1].set_xlabel('Wind speed / m/s')
            ax_subplots[0].set_title('Frequency sensitivity')
            ax_subplots[1].set_title('Damping sensitivity')
            plt.tight_layout()

            # fig = plot.get_figure()
            fig_subplots.savefig(os.path.join(UQResultsAnalysis.output_dir, '{} - {}.png'.format(which_sobol, feature)), dpi=300)

            plt.show()


def compare_oat_metrics(UQResultsAnalysis, oat_metric, ylabel):
    """
    Compare oat metrics of multiple framework runs with each other. Assumed that those runs are made with the
    same uncertain parameter sets. Bar plot made with pandas.
    """
    feat_to_label = {'first_edge_bw': '1st edge BW',
                     'first_edge_fw': '1st edge FW',
                     'second_edge_bw': '2nd edge BW',
                     'second_edge_fw': '2nd edge FW',
                     'total': 'Total'}

    existing_sorting_index = None
    for visualization_mode in ['unsorted', 'sorted']:
        fig_subplots, ax_subplots = plt.subplots(num='Comparison OAT metrics - All Features - {}'.format(ylabel),
                                                 nrows=1, ncols=5, figsize=(14, 10.5), sharex=True)

        for ifeat, feature in enumerate(['total', 'first_edge_bw', 'first_edge_fw', 'second_edge_bw', 'second_edge_fw']):
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
                ax.set_title(feat_to_label[feature])
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


def campbell_diagram_from_OAT_old(UQResultsAnalysis, uq_results):

    # 51 Section Blade
    #reference_frequency_progression = {
    #    'Rotor 1st edgewise backward whirl': [0.576163, 0.569978, 0.570617, 0.571181, 0.56783, 0.56467],
    #    'Rotor 1st edgewise forward whirl': [0.810334, 0.821295, 0.822507, 0.819621, 0.815166, 0.811032],
    #    'Rotor 2nd edgewise backward whirl': [2.0494, 2.0184, 2.0423, 2.0542, 2.0583, 2.0614],
    #    'Rotor 2nd edgewise forward whirl': [2.24289, 2.25599, 2.24389, 2.21513, 2.19358, 2.17557]}

    #reference_damping_progression = {
    #    'Rotor 1st edgewise backward whirl': np.array([0.028861, 0.015432, 0.008905, 0.007159, 0.007289, 0.007924])*100,
    #    'Rotor 1st edgewise forward whirl': np.array([0.029442, 0.017596, 0.0115, 0.00923, 0.008381, 0.008059])*100,
    #    'Rotor 2nd edgewise backward whirl': np.array([0.0436, 0.0233, 0.0139, 0.0134, 0.0154, 0.0172])*100,
    #    'Rotor 2nd edgewise forward whirl': np.array([0.020073, 0.014623, 0.010538, 0.012653, 0.018184, 0.026091])*100}

    # 26 Section Blade
    reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.579, 0.572, 0.572, 0.570, 0.569, 0.563],
                                       'Rotor 1st edgewise forward whirl': [0.813, 0.823, 0.824, 0.822, 0.817, 0.816],
                                       'Rotor 2nd edgewise backward whirl': [2.053, 2.034, 2.050, 2.054, 2.061, 2.058],
                                       'Rotor 2nd edgewise forward whirl': [2.257, 2.265, 2.253, 2.235, 2.208, 2.205]}

    reference_damping_progression = {'Rotor 1st edgewise backward whirl': [2.927, 1.638, 0.947, 0.760, 0.707, 0.769],
                                     'Rotor 1st edgewise forward whirl': [2.997, 1.838, 1.195, 0.937, 0.831, 0.746],
                                     'Rotor 2nd edgewise backward whirl': [3.143, 1.986, 1.259, 1.307, 1.354, 1.679],
                                     'Rotor 2nd edgewise forward whirl': [1.823, 1.408, 1.033, 1.163, 1.625, 2.070]}

    # nr_wind_speeds = 6  # uq_results.data['CampbellDiagramModel'].evaluations.shape[1]
    # start_wind_speed = 10
    ws_range = [10, 12, 14, 16, 18, 20]

    for ii, param in enumerate(uq_results.uncertain_parameters):
        print(param)

        low_freq = uq_results.data[uq_results.model_name].frequency[ii * 2]
        high_freq = uq_results.data[uq_results.model_name].frequency[ii * 2 + 1]
        low_damp = uq_results.data[uq_results.model_name].damping[ii * 2]
        high_damp = uq_results.data[uq_results.model_name].damping[ii * 2 + 1]

        mode_names_low = uq_results.data[uq_results.model_name].mode_names[ii * 2]
        mode_names_high = uq_results.data[uq_results.model_name].mode_names[ii * 2 + 1]
        if not np.all(mode_names_low == mode_names_high):
            print('Postprocessing failed for {}, Campbell diagram with uncertainty bands can not be made'.format(param))
            print('This is either because the postprocessing of at least one of the simulations failed, or because '
                  'the extracted modes do not have the same name.')
            mode_names = [b'Failed'] * len(mode_names_low)
        else:
            mode_names = mode_names_low

        #if mode_names[0] == 'Failed':
        #    print('The postprocessing of uncertain parameter {} failed'.format(param))
        #    continue

        fig = plt.figure('Campbell Diagram with Uncertainty Bands - {}'.format(param), figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for ii, mode_name in enumerate(mode_names):
            mode_name = mode_name.decode('utf8')

            if mode_name != 'Failed':
                line = ax1.plot(ws_range, reference_frequency_progression[mode_name],
                                marker='o', markerfacecolor='white', linewidth=1,
                                label=mode_name + ' - reference')
                line = ax2.plot(ws_range, reference_damping_progression[mode_name],
                                marker='o', markerfacecolor='white', linewidth=1)

            ax1.fill_between(ws_range, low_freq[:, ii], high_freq[:, ii], color=line[0].get_color(), alpha=0.5, label=mode_name)
            ax2.fill_between(ws_range, low_damp[:, ii], high_damp[:, ii], color=line[0].get_color(), alpha=0.5)

        ax1.set_xlim([min(ws_range)-1, max(ws_range)+1])
        ax1.set_ylim([0, 2.5])
        ax2.set_ylim([-1, 5])

        ax2.fill_between([-10, 100], y1=0, y2=-10, where=None, facecolor='grey', alpha=0.1, hatch='/')
        ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax1.grid()
        ax2.grid()
        ax1.set_ylabel('Frequency / Hz')
        ax2.set_ylabel('Damping Ratio / %')
        ax2.set_xlabel('Wind speed / m/s')
        fig.tight_layout()

        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'Campbell diagram '+param+'.png'), dpi=300)
        plt.close()
        # plt.show()


def campbell_diagram_from_OAT(UQResultsAnalysis, uq_results):

    # 26 Section Blade
    reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.579, 0.572, 0.572, 0.570, 0.569, 0.563],
                                       'Rotor 1st edgewise forward whirl': [0.813, 0.823, 0.824, 0.822, 0.817, 0.816],
                                       'Rotor 2nd edgewise backward whirl': [2.053, 2.034, 2.050, 2.054, 2.061, 2.058],
                                       'Rotor 2nd edgewise forward whirl': [2.257, 2.265, 2.253, 2.235, 2.208, 2.205]}

    reference_damping_progression = {'Rotor 1st edgewise backward whirl': [2.927, 1.638, 0.947, 0.760, 0.707, 0.769],
                                     'Rotor 1st edgewise forward whirl': [2.997, 1.838, 1.195, 0.937, 0.831, 0.746],
                                     'Rotor 2nd edgewise backward whirl': [3.143, 1.986, 1.259, 1.307, 1.354, 1.679],
                                     'Rotor 2nd edgewise forward whirl': [1.823, 1.408, 1.033, 1.163, 1.625, 2.070]}

    # nr_wind_speeds = 6  # uq_results.data['CampbellDiagramModel'].evaluations.shape[1]
    # start_wind_speed = 10
    ws_range = np.arange(3, 26)
    n_ws = len(ws_range)

    nr_params = len(uq_results.uncertain_parameters)
    nr_iters_per_param = 2  # mean - range and mean + range

    colors_features = [None, 'C0', 'C1', 'C2', 'C3']
    for param_idx in range(nr_params):

        param_name = uq_results.uncertain_parameters[param_idx]

        fig = plt.figure('Campbell Diagram OAT - Param {}'.format(param_name), figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for feature, feat_color in zip(uq_results, colors_features):

            if feature == 'CampbellDiagramModel':
                continue

            for iteration in range(nr_iters_per_param * param_idx, nr_iters_per_param * (1 + param_idx)):
                freq_and_damp = uq_results[feature].evaluations[iteration]
                if np.any(np.isnan(freq_and_damp)):
                    continue

                ax1.plot(ws_range, freq_and_damp[:n_ws], color=feat_color)
                ax2.plot(ws_range, freq_and_damp[n_ws:], color=feat_color)

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


def campbell_diagram_from_EE_old(UQResultsAnalysis, uq_results):

    # 51 Section Blade
    #reference_frequency_progression = {
    #    'Rotor 1st edgewise backward whirl': [0.576163, 0.569978, 0.570617, 0.571181, 0.56783, 0.56467],
    #    'Rotor 1st edgewise forward whirl': [0.810334, 0.821295, 0.822507, 0.819621, 0.815166, 0.811032],
    #    'Rotor 2nd edgewise backward whirl': [2.0494, 2.0184, 2.0423, 2.0542, 2.0583, 2.0614],
    #    'Rotor 2nd edgewise forward whirl': [2.24289, 2.25599, 2.24389, 2.21513, 2.19358, 2.17557]}

    #reference_damping_progression = {
    #    'Rotor 1st edgewise backward whirl': np.array([0.028861, 0.015432, 0.008905, 0.007159, 0.007289, 0.007924])*100,
    #    'Rotor 1st edgewise forward whirl': np.array([0.029442, 0.017596, 0.0115, 0.00923, 0.008381, 0.008059])*100,
    #    'Rotor 2nd edgewise backward whirl': np.array([0.0436, 0.0233, 0.0139, 0.0134, 0.0154, 0.0172])*100,
    #    'Rotor 2nd edgewise forward whirl': np.array([0.020073, 0.014623, 0.010538, 0.012653, 0.018184, 0.026091])*100}

    # 26 Section Blade
    reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.579, 0.572, 0.572, 0.570, 0.569, 0.563],
                                       'Rotor 1st edgewise forward whirl': [0.813, 0.823, 0.824, 0.822, 0.817, 0.816],
                                       'Rotor 2nd edgewise backward whirl': [2.053, 2.034, 2.050, 2.054, 2.061, 2.058],
                                       'Rotor 2nd edgewise forward whirl': [2.257, 2.265, 2.253, 2.235, 2.208, 2.205]}

    reference_damping_progression = {'Rotor 1st edgewise backward whirl': [2.927, 1.638, 0.947, 0.760, 0.707, 0.769],
                                     'Rotor 1st edgewise forward whirl': [2.997, 1.838, 1.195, 0.937, 0.831, 0.746],
                                     'Rotor 2nd edgewise backward whirl': [3.143, 1.986, 1.259, 1.307, 1.354, 1.679],
                                     'Rotor 2nd edgewise forward whirl': [1.823, 1.408, 1.033, 1.163, 1.625, 2.070]}

    # nr_wind_speeds = 6  # uq_results.data['CampbellDiagramModel'].evaluations.shape[1]
    # start_wind_speed = 10
    ws_range = [10, 12, 14, 16, 18, 20]

    nr_params = len(uq_results.uncertain_parameters)
    nr_iters_per_repetition = nr_params + 1
    nr_morris_repetitions = int(len(uq_results[uq_results.model_name].evaluations) / nr_iters_per_repetition)

    colors_features = [None, 'C0', 'C1', 'C2', 'C3']
    for repetition in range(nr_morris_repetitions):

        fig = plt.figure('Campbell Diagram EE - Repetition {}'.format(repetition+1), figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for feature, feat_color in zip(uq_results, colors_features):

            if feature == 'CampbellDiagramModel':
                continue

            for iteration in range(nr_iters_per_repetition * repetition, nr_iters_per_repetition * (1 + repetition)):
                freq_and_damp = uq_results[feature].evaluations[iteration]
                if np.any(np.isnan(freq_and_damp)):
                    continue

                ax1.plot(ws_range, freq_and_damp[:6], color=feat_color)
                ax2.plot(ws_range, freq_and_damp[6:], color=feat_color)

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


def campbell_diagram_from_EE(UQResultsAnalysis, uq_results):

    # 51 Section Blade
    #reference_frequency_progression = {
    #    'Rotor 1st edgewise backward whirl': [0.576163, 0.569978, 0.570617, 0.571181, 0.56783, 0.56467],
    #    'Rotor 1st edgewise forward whirl': [0.810334, 0.821295, 0.822507, 0.819621, 0.815166, 0.811032],
    #    'Rotor 2nd edgewise backward whirl': [2.0494, 2.0184, 2.0423, 2.0542, 2.0583, 2.0614],
    #    'Rotor 2nd edgewise forward whirl': [2.24289, 2.25599, 2.24389, 2.21513, 2.19358, 2.17557]}

    #reference_damping_progression = {
    #    'Rotor 1st edgewise backward whirl': np.array([0.028861, 0.015432, 0.008905, 0.007159, 0.007289, 0.007924])*100,
    #    'Rotor 1st edgewise forward whirl': np.array([0.029442, 0.017596, 0.0115, 0.00923, 0.008381, 0.008059])*100,
    #    'Rotor 2nd edgewise backward whirl': np.array([0.0436, 0.0233, 0.0139, 0.0134, 0.0154, 0.0172])*100,
    #    'Rotor 2nd edgewise forward whirl': np.array([0.020073, 0.014623, 0.010538, 0.012653, 0.018184, 0.026091])*100}

    # 26 Section Blade
    reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.579, 0.572, 0.572, 0.570, 0.569, 0.563],
                                       'Rotor 1st edgewise forward whirl': [0.813, 0.823, 0.824, 0.822, 0.817, 0.816],
                                       'Rotor 2nd edgewise backward whirl': [2.053, 2.034, 2.050, 2.054, 2.061, 2.058],
                                       'Rotor 2nd edgewise forward whirl': [2.257, 2.265, 2.253, 2.235, 2.208, 2.205]}

    reference_damping_progression = {'Rotor 1st edgewise backward whirl': [2.927, 1.638, 0.947, 0.760, 0.707, 0.769],
                                     'Rotor 1st edgewise forward whirl': [2.997, 1.838, 1.195, 0.937, 0.831, 0.746],
                                     'Rotor 2nd edgewise backward whirl': [3.143, 1.986, 1.259, 1.307, 1.354, 1.679],
                                     'Rotor 2nd edgewise forward whirl': [1.823, 1.408, 1.033, 1.163, 1.625, 2.070]}

    # nr_wind_speeds = 6  # uq_results.data['CampbellDiagramModel'].evaluations.shape[1]
    # start_wind_speed = 10
    ws_range = np.arange(3, 26)
    n_ws = len(ws_range)

    nr_params = len(uq_results.uncertain_parameters)
    nr_iters_per_repetition = nr_params + 1
    nr_morris_repetitions = int(len(uq_results[uq_results.model_name].evaluations) / nr_iters_per_repetition)

    colors_features = [None, 'C0', 'C1', 'C2', 'C3']
    for repetition in range(nr_morris_repetitions):

        fig = plt.figure('Campbell Diagram EE - Repetition {}'.format(repetition+1), figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for feature, feat_color in zip(uq_results, colors_features):

            if feature == 'CampbellDiagramModel':
                continue

            for iteration in range(nr_iters_per_repetition * repetition, nr_iters_per_repetition * (1 + repetition)):
                freq_and_damp = uq_results[feature].evaluations[iteration]
                if np.any(np.isnan(freq_and_damp)):
                    continue

                ax1.plot(ws_range, freq_and_damp[:n_ws], color=feat_color)
                ax2.plot(ws_range, freq_and_damp[n_ws:], color=feat_color)

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
            axes[1, iter].plot(ws_range, mean_evaluated_polynom[nr_ws:], color=feat_colors[feature])

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

    axes[0, -1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    axes[0, 0].set_ylabel('Frequency / Hz')
    axes[1, 0].set_ylabel('Damping Ratio / %')

    fig.tight_layout()

    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'first_order_campbell_diagram_pce.png'), bbox_inches='tight', dpi=300)
    plt.show()


def input_params_rejected_samples(dir, UQResultsAnalysis):
    import json
    from run_analysis import CampbellDiagramModel
    from tool_interfaces.hawcstab2_interface import HAWCStab2Model

    uq_results = UQResultsAnalysis.get_UQ_results(UQResultsAnalysis.uncertainpy_results[0])

    input_params_accepted = []
    input_params_rejected_edge_mode_order_not_correct = []
    input_params_rejected_mode_tracking_not_correct = []
    input_params_rejected_mode_at_first_op_point_too_different_from_ref_mode = []

    accepted_iteration_run_directories = [os.path.split(run_dir_path)[-1].decode('utf8') for run_dir_path in uq_results.data[uq_results.model_name].iteration_run_directory]

    for dir in glob.glob(dir):

        if dir[-10:] == 'uq_results':
            continue

        try:
            with open(os.path.join(dir, 'input_parameters.txt')) as f:
                data = f.read()
            input_params = json.loads(data)

            simulation_tool = HAWCStab2Model(dir, {'cmb_filename': 'IEA_15MW_RWT_Onshore.cmb'})

            result_dict = load_dict_h5py(os.path.join(dir, 'result_dict.hdf5'))
            success, reason_for_failure = CampbellDiagramModel._postprocessor_hs2(None, result_dict, simulation_tool,
                                                                                  postpro_method='strict name and order based postpro')

        except FileNotFoundError:
            raise FileNotFoundError('File not found in directory', dir)

        dir_id = os.path.split(dir)[-1]

        if dir_id in accepted_iteration_run_directories:
            input_params_accepted.append(input_params)
        else:
            if reason_for_failure == 'edge_mode_order_not_correct':
                input_params_rejected_edge_mode_order_not_correct.append(input_params)
            elif reason_for_failure == 'mode_tracking_not_correct':
                input_params_rejected_mode_tracking_not_correct.append(input_params)
            elif reason_for_failure == 'mode_at_first_op_point_too_different_from_ref_mode':
                input_params_rejected_mode_at_first_op_point_too_different_from_ref_mode.append(input_params)


    fig = plt.figure('1D input parameters visualization')
    ax = fig.gca()

    for iparam, param in enumerate(uq_results.uncertain_parameters):

        for accepted_input_params in input_params_accepted:
            ax.plot(accepted_input_params[param], iparam, marker="o", color='C0')

        for rejected_input_params in input_params_rejected_edge_mode_order_not_correct:
            ax.plot(rejected_input_params[param], iparam, marker="s", color='C3')

        for rejected_input_params in input_params_rejected_mode_tracking_not_correct:
            ax.plot(rejected_input_params[param], iparam, marker="x", color='C1')

        for rejected_input_params in input_params_rejected_mode_at_first_op_point_too_different_from_ref_mode:
            ax.plot(rejected_input_params[param], iparam, marker="+", color='C2')

    ax.yaxis.set_ticks(np.arange(len(uq_results.uncertain_parameters)))
    ax.set_yticklabels(uq_results.uncertain_parameters)

    point_accepted = matplotlib.lines.Line2D([0], [0], label='Accepted samples', marker="o", color='C0', linestyle='')
    point_rejected1 = matplotlib.lines.Line2D([0], [0], label='Rejected samples (edge mode order)', marker="s", color='C3', linestyle='')
    point_rejected2 = matplotlib.lines.Line2D([0], [0], label='Rejected samples (mode tracking)', marker="x", color='C1', linestyle='')
    point_rejected3 = matplotlib.lines.Line2D([0], [0], label='Rejected samples (mode too different from reference)', marker="+", color='C2', linestyle='')

    plt.legend(handles=[point_accepted, point_rejected1, point_rejected2, point_rejected3])
    plt.tight_layout()
    fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'rejected_samples.png'), dpi=300)
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


def combined_oat_ee_plot(UQResultsAnalysis_OAT, UQResultsAnalysis_EE, extra_label=''):
    uq_results_oat = UQResultsAnalysis_OAT.get_UQ_results(UQResultsAnalysis_OAT.uncertainpy_results[0])
    uq_results_ee = UQResultsAnalysis_EE.get_UQ_results(UQResultsAnalysis_EE.uncertainpy_results[0])

    oat_max = dict()
    oat_mean = dict()

    for feature in uq_results_oat.data:
        if feature == 'CampbellDiagramModel':
            continue
        oat_max[feature] = uq_results_oat.data[feature].s_oat_max
        oat_mean[feature] = uq_results_oat.data[feature].s_oat_mean

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

    for dicti in [oat_max, ee_mean_max, ee_std_max]:
        dicti['total'] = np.max(
            np.vstack((dicti['first_edge_bw'],
                       dicti['first_edge_fw'],
                       dicti['second_edge_bw'],
                       dicti['second_edge_fw'])), axis=0)

    for dicti in [oat_mean, ee_mean_mean, ee_std_mean]:
        dicti['total'] = np.mean(
            np.vstack((dicti['first_edge_bw'],
                       dicti['first_edge_fw'],
                       dicti['second_edge_bw'],
                       dicti['second_edge_fw'])), axis=0)

    feat_to_label = {'first_edge_bw': '1st edge BW',
                     'first_edge_fw': '1st edge FW',
                     'second_edge_bw': '2nd edge BW',
                     'second_edge_fw': '2nd edge FW',
                     'total': 'Total'}

    for label, oat_metric, ee_metric, ee_metric_std in zip(['mean', 'max'], [oat_mean, oat_max], [ee_mean_mean, ee_mean_max], [ee_std_mean, ee_std_max]):
        fig_subplots, ax_subplots = plt.subplots(num='Comparison OAT and EE metrics - All Features - {}'.format(label),
                                                 nrows=1, ncols=5, figsize=(9, 7), sharex=True)

        existing_sorting_index = None
        for ifeat, feature in enumerate(['total', 'first_edge_bw', 'first_edge_fw', 'second_edge_bw', 'second_edge_fw']):

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
                    df1 = df1.sort_values(by=['EE'])
                    existing_sorting_index = df1.index
                else:
                    df1 = df1.reindex(existing_sorting_index)
                df_errors = df_errors.reindex(existing_sorting_index)

            if ifeat == 0:
                legend_flag = True
            else:
                legend_flag = False

            df1.plot.barh(ax=ax_subplots[ifeat], legend=legend_flag)  # , yerr=df_errors

            ax_subplots[ifeat].set_title(feat_to_label[feature])
            ax_subplots[ifeat].grid(axis='y')
            ax_subplots[ifeat].set_xlabel('EE value')
            ax_subplots[ifeat].set_axisbelow(True)

            if ifeat > 0:
                ax_subplots[ifeat].tick_params(labelleft=False)

        fig_subplots.tight_layout()
        fig_subplots.savefig(r'C:\Users\verd_he\projects\Torque2024\abstract\All features - {} - sorted - {}.png'.format(label, extra_label), bbox_inches='tight', dpi=300)
        # plt.close(fig_subplots)
        plt.show()



if __name__ == '__main__':

    #UQResultsAnalysis_OAT = UQResultsAnalysis([r'/work/verd_he/projects/torque2024_hs2/wtuq/use_cases/campbell_diagram/results/IEA_15MW_oat_1E-1_first_run/uq_results/CampbellDiagramModel.h5'],
    #                                          ['hawcstab2'])
    #result_analysis_OAT(UQResultsAnalysis_OAT)

    #transform_to_campbellviewer_database(
    #    r'/work/verd_he/projects/torque2024_hs2/wtuq/use_cases/campbell_diagram/results/IEA_15MW_morris_1E-1/*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_morris_1E-1', get_run_name_from='from_dir_name')

    #UQResultsAnalysis_EE = UQResultsAnalysis([r'/work/verd_he/projects/torque2024_hs2/wtuq/use_cases/campbell_diagram/results/IEA_15MW_morris_1E-1/uq_results/CampbellDiagramModel.h5'],
    #                                         ['hawcstab2'])
    #result_analysis_EE(UQResultsAnalysis_EE)

    #for morris_repetitions in range(30, 31):
    #    UQResultsAnalysis_EE = UQResultsAnalysis([r'Z:\projects\Torque2024\IEA_15MW_morris_convergence_study_2\IEA_15MW_morris_postpro_{}\uq_results\CampbellDiagramModel.h5'.format(morris_repetitions)],
    #                                             ['bladed-lin'])
    #    combined_oat_ee_plot(UQResultsAnalysis_OAT, UQResultsAnalysis_EE, extra_label=morris_repetitions)



    #uq_plots = UQResultsAnalysis([r'/work/verd_he/projects/torque2024_hs2/wtuq/use_cases/campbell_diagram/results/IEA_15MW_pce/uq_results/CampbellDiagramModel.h5'],
    #                             ['hawcstab2'])
    #result_analysis_PCE(uq_plots)

    uq_plots = UQResultsAnalysis([r'/work/verd_he/projects/torque2024_hs2/wtuq/use_cases/campbell_diagram/results/IEA_15MW_pce_4params_25points_NEW/uq_results/CampbellDiagramModel.h5'],
                                 ['hawcstab2'])
    result_analysis_PCE(uq_plots)
    input_params_rejected_samples(r'/work/verd_he/projects/torque2024_hs2/wtuq/use_cases/campbell_diagram/results/IEA_15MW_pce_4params_25points_NEW/*', uq_plots)
