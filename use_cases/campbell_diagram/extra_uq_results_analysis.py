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
                              'blade_cog_x': r'Blade cog position $\bot$ to chord',
                              'blade_cog_y': r'Blade cog position $\parallel$ to chord',
                              'blade_sc_x': r'Blade shear center position $\bot$ to chord',
                              'blade_sc_y': r'Blade shear center position $\parallel$ to chord',
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
                              'LSS_damping': 'Drivetrain damping'}


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


            run_name = dir.split('\\')[-1]  # default name if specific run name can not be found
            if get_run_name_from == 'from_input_params':
                for param, value in input_params.items():
                    if value < 0:
                        # CUTOUT 'SCALAR PROPERTY'
                        run_name = param[:-16] + '_low'
                    if value > 0:
                        # CUTOUT 'SCALAR PROPERTY'
                        run_name = param[:-16] + '_high'

            database.add_data(run_name, 'bladed-lin',
                              tool_specific_info={'result_dir': dir, 'result_prefix': 'stab_analysis_run'})
            nr_datasets_added += 1

            if nr_datasets_added == max_nr_datasets:
                database.save(fname='{}_{}.nc'.format(base_output_name, database_nr))
                database['Bladed (lin.)'] = dict()
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

    # UQResultsAnalysis.uncertain_param_names[UQResultsAnalysis.uncertainpy_results[0]] = uq_results.uncertain_parameters

    sobol_index_plots2(UQResultsAnalysis, uq_results)

    campbell_diagram_from_PCE(UQResultsAnalysis, uq_results)


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


def sobol_index_plots2(UQResultsAnalysis, uq_results):

    for feature in uq_results.data:
        if feature == uq_results.model_name and uq_results.model_ignore == True:
            continue

        for which_sobol, sobol_arr in zip(['First order Sobol index', 'Total Sobol index'],
                                          [uq_results.data[feature].sobol_first, uq_results.data[feature].sobol_total]):

            fig_subplots, ax_subplots = plt.subplots(num='{} - {} - vertical'.format(which_sobol, feature),
                                                     nrows=2, ncols=1, figsize=(10, 4), sharex=True)

            param_names = [uncertain_param_label_dict[param_name[:-16]] for param_name in uq_results.uncertain_parameters]
            sobol_arr_freq, sobol_arr_damp = np.split(sobol_arr, 2, axis=1)
            df_freq = pd.DataFrame(sobol_arr_freq, index=param_names, columns=np.arange(2, 22, 2))
            df_damp = pd.DataFrame(sobol_arr_damp, index=param_names, columns=np.arange(2, 22, 2))

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
                    df1 = df1.sort_values(by=['bladed-lin'])
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


def campbell_diagram_from_OAT(UQResultsAnalysis, uq_results):

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


def campbell_diagram_from_PCE(UQResultsAnalysis, uq_results):

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
    #reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.579, 0.572, 0.572, 0.570, 0.569, 0.563],
    #                                   'Rotor 1st edgewise forward whirl': [0.813, 0.823, 0.824, 0.822, 0.817, 0.816],
    #                                   'Rotor 2nd edgewise backward whirl': [2.053, 2.034, 2.050, 2.054, 2.061, 2.058],
    #                                   'Rotor 2nd edgewise forward whirl': [2.257, 2.265, 2.253, 2.235, 2.208, 2.205]}

    #reference_damping_progression = {'Rotor 1st edgewise backward whirl': [2.927, 1.638, 0.947, 0.760, 0.707, 0.769],
    #                                 'Rotor 1st edgewise forward whirl': [2.997, 1.838, 1.195, 0.937, 0.831, 0.746],
    #                                 'Rotor 2nd edgewise backward whirl': [3.143, 1.986, 1.259, 1.307, 1.354, 1.679],
    #                                 'Rotor 2nd edgewise forward whirl': [1.823, 1.408, 1.033, 1.163, 1.625, 2.070]}

    # 26 Section Blade - 2-20-2
    reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.611, 0.613, 0.618, 0.608, 0.579, 0.571, 0.572, 0.570, 0.569, 0.563],
                                       'Rotor 1st edgewise forward whirl': [0.781, 0.784, 0.789, 0.800, 0.814, 0.823, 0.824, 0.822, 0.820, 0.816],
                                       'Rotor 2nd edgewise backward whirl': [2.063, 2.072, 2.082, 2.079, 2.054, 2.036, 2.050, 2.054, 2.056, 2.058],
                                       'Rotor 2nd edgewise forward whirl': [2.194, 2.206, 2.218, 2.238, 2.257, 2.267, 2.253, 2.235, 2.220, 2.206]}

    reference_damping_progression = {'Rotor 1st edgewise backward whirl': [1.09, 1.23, 1.88, 2.80, 3.17, 1.50, 0.947, 0.760, 0.707, 0.769],
                                     'Rotor 1st edgewise forward whirl': [1.17, 1.57, 2.41, 3.22, 3.17, 1.74, 1.22, 0.93, 0.81, 0.76],
                                     'Rotor 2nd edgewise backward whirl': [1.09, 1.14, 1.21, 1.58, 3.10, 2.02, 1.26, 1.31, 1.48, 1.68],
                                     'Rotor 2nd edgewise forward whirl': [0.77, 0.65, 0.57, 0.78, 1.78, 1.41, 1.03, 1.16, 1.53, 2.07]}

    # nr_wind_speeds = 6  # uq_results.data['CampbellDiagramModel'].evaluations.shape[1]
    # start_wind_speed = 10
    ws_range = [10, 12, 14, 16, 18, 20]
    feat_colors = {'first_edge_bw': 'C0',
                   'first_edge_fw': 'C1',
                   'second_edge_bw': 'C2',
                   'second_edge_fw': 'C3'}
    feat_names = {'first_edge_bw': 'Rotor 1st edgewise backward whirl',
                  'first_edge_fw': 'Rotor 1st edgewise forward whirl',
                  'second_edge_bw': 'Rotor 2nd edgewise backward whirl',
                  'second_edge_fw': 'Rotor 2nd edgewise forward whirl'}

    nr_iterations = len(uq_results[uq_results.model_name].evaluations)
    for iteration in range(nr_iterations):

        fig = plt.figure('Campbell Diagram PCE - Iteration {}'.format(iteration), figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for feature in uq_results:

            if feature == 'CampbellDiagramModel':
                continue

            ax1.plot(ws_range, reference_frequency_progression[feat_names[feature]], linestyle=':', color=feat_colors[feature])
            ax2.plot(ws_range, reference_damping_progression[feat_names[feature]], linestyle=':', color=feat_colors[feature])

            freq_and_damp = uq_results[feature].evaluations[iteration]
            if np.any(np.isnan(freq_and_damp)):
                continue

            ax1.plot(ws_range, freq_and_damp[:6], color=feat_colors[feature])
            ax2.plot(ws_range, freq_and_damp[6:], color=feat_colors[feature])


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

        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, 'Campbell diagram - Iteration {}.png'.format(iteration)), dpi=50)
        #plt.close(fig)
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

    #UQResultsAnalysis_OAT = UQResultsAnalysis([r'Z:\projects\Torque2024\IEA_15MW_oat_linear_postpro_new\uq_results\CampbellDiagramModel.h5'],
    #                                          ['bladed-lin'])
    #for morris_repetitions in range(30, 31):
    #    UQResultsAnalysis_EE = UQResultsAnalysis([r'Z:\projects\Torque2024\IEA_15MW_morris_convergence_study_2\IEA_15MW_morris_postpro_{}\uq_results\CampbellDiagramModel.h5'.format(morris_repetitions)],
    #                                             ['bladed-lin'])
    #    combined_oat_ee_plot(UQResultsAnalysis_OAT, UQResultsAnalysis_EE, extra_label=morris_repetitions)



    uq_plots = UQResultsAnalysis([r'Z:\projects\Torque2024\IEA_15MW_pce_2-20-2_postpro_final\uq_results\CampbellDiagramModel.h5'],
                                 ['bladed-lin'])
    result_analysis_PCE(uq_plots)




    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\probably_morris_24_2\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_morris_probably_24_2', get_run_name_from='from_dir_name')

    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\IEA_15MW_pce_2-20-2_first_run\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_pce_2-20-2', get_run_name_from='from_dir_name')

    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\IEA_15MW_pce_first_run\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_pce', get_run_name_from='from_dir_name')

    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\IEA_15MW_MC\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_MC', get_run_name_from='from_dir_name')
    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\IEA_15MW_MC_conservative\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_MC_conservative', get_run_name_from='from_dir_name')
    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\IEA_15MW_oat\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_oat', get_run_name_from='from_input_params')
    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\IEA_15MW_oat_linear\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_oat_linear', get_run_name_from='from_input_params')
    # transform_to_campbellviewer_database(r'E:\IEA_15MW_large_uncertainties\*', base_output_name='CampbellViewerDatabase_IEA_15MW_large_uncertainties')
    # transform_to_campbellviewer_database(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\IEA_15MW\*',
    #                                      base_output_name='CampbellViewerDatabase_IEA_15MW')
    # transform_to_campbellviewer_database(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\IEA_15MW_fast_large_uncertainties\*', base_output_name='CampbellViewerDatabase_IEA_15MW_fast_large_uncertainties')
    # transform_to_campbellviewer_database(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\demo_a_testing_full_FIRST_RUN\*')