import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['hatch.color']     = 'grey'
matplotlib.rcParams['hatch.linewidth'] = 0.2
import numpy as np
import os
import glob
import pandas as pd

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


def transform_to_campbellviewer_database(dir, base_output_name='CampbellViewerDatabase_', run_name='from_input_params'):
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
            if run_name == 'from_input_params':
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
        oat_max[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].s_oat_max
        oat_mean[UQResultsAnalysis.uncertainpy_results[0]][feature] = uq_results.data[feature].s_oat_mean

    oat_max[UQResultsAnalysis.uncertainpy_results[0]]['total'] = np.max(np.vstack((oat_max[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_bw'],
                                                                                   oat_max[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_fw'],
                                                                                   oat_max[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_bw'],
                                                                                   oat_max[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_fw'])), axis=0)
    oat_mean[UQResultsAnalysis.uncertainpy_results[0]]['total'] = np.mean(np.vstack((oat_max[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_bw'],
                                                                                     oat_max[UQResultsAnalysis.uncertainpy_results[0]]['first_edge_fw'],
                                                                                     oat_max[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_bw'],
                                                                                     oat_max[UQResultsAnalysis.uncertainpy_results[0]]['second_edge_fw'])), axis=0)

    compare_oat_metrics(UQResultsAnalysis, oat_max, ylabel='max. OAT metric')
    compare_oat_metrics(UQResultsAnalysis, oat_mean, ylabel='mean OAT metric')

    campbell_diagram_from_OAT(UQResultsAnalysis, uq_results)


def compare_oat_metrics(UQResultsAnalysis, oat_max, ylabel):
    """
    Compare oat metrics of multiple framework runs with each other. Assumed that those runs are made with the
    same uncertain parameter sets. Bar plot made with pandas.
    """
    for feature in ['first_edge_bw', 'first_edge_fw', 'second_edge_bw', 'second_edge_fw', 'total']:
        fig = plt.figure('Comparison OAT metrics - {} - {}'.format(feature, ylabel))
        ax1 = fig.gca()

        metric_pd_dict = dict()  # key => legend label
        color_list = list()
        for uq_result in oat_max:
            oat_max_nans_eliminated = oat_max[uq_result][feature]
            oat_max_nans_eliminated[np.isnan(oat_max[uq_result][feature])] = -1
            metric_pd_dict[UQResultsAnalysis.run_label[uq_result]] = oat_max_nans_eliminated
            color_list.append(UQResultsAnalysis.run_color[uq_result])

        # CUT OFF THE SCALAR_PROPERTY FOR NOW
        index = [name[:-16] for name in UQResultsAnalysis.uncertain_param_names[uq_result]]
        df1 = pd.DataFrame(metric_pd_dict, index=index)
        if color_list[0] == None:
            df1.plot.bar(ax=ax1, rot=90)
        else:
            df1.plot.bar(ax=ax1, rot=90, color=color_list)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax1.set_ylabel(ylabel)
        ax1.grid(axis='y')
        ax1.set_axisbelow(True)
        fig.tight_layout()

        fig.savefig(os.path.join(UQResultsAnalysis.output_dir, '{} - {}.png'.format(feature, ylabel)), bbox_inches='tight', dpi=300)
        plt.show()
        # plt.close(fig)


def campbell_diagram_from_OAT(UQResultsAnalysis, uq_results):

    reference_frequency_progression = {
        'Rotor 1st edgewise backward whirl': [0.576163, 0.569978, 0.570617, 0.571181, 0.56783, 0.56467],
        'Rotor 1st edgewise forward whirl': [0.810334, 0.821295, 0.822507, 0.819621, 0.815166, 0.811032],
        'Rotor 2nd edgewise backward whirl': [2.0494, 2.0184, 2.0423, 2.0542, 2.0583, 2.0614],
        'Rotor 2nd edgewise forward whirl': [2.24289, 2.25599, 2.24389, 2.21513, 2.19358, 2.17557]}

    reference_damping_progression = {
        'Rotor 1st edgewise backward whirl': np.array([0.028861, 0.015432, 0.008905, 0.007159, 0.007289, 0.007924])*100,
        'Rotor 1st edgewise forward whirl': np.array([0.029442, 0.017596, 0.0115, 0.00923, 0.008381, 0.008059])*100,
        'Rotor 2nd edgewise backward whirl': np.array([0.0436, 0.0233, 0.0139, 0.0134, 0.0154, 0.0172])*100,
        'Rotor 2nd edgewise forward whirl': np.array([0.020073, 0.014623, 0.010538, 0.012653, 0.018184, 0.026091])*100}


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


if __name__ == '__main__':
    #transform_to_campbellviewer_database(
    #    r'Z:\projects\Torque2024\IEA_15MW_MC\*',
    #    base_output_name='CampbellViewerDatabase_IEA_15MW_MC', run_name='from_dir_name')
    transform_to_campbellviewer_database(
        r'Z:\projects\Torque2024\IEA_15MW_MC_conservative\*',
        base_output_name='CampbellViewerDatabase_IEA_15MW_MC_conservative', run_name='from_dir_name')
    # transform_to_campbellviewer_database(r'E:\IEA_15MW_large_uncertainties\*', base_output_name='CampbellViewerDatabase_IEA_15MW_large_uncertainties')
    # transform_to_campbellviewer_database(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\IEA_15MW\*',
    #                                      base_output_name='CampbellViewerDatabase_IEA_15MW')
    # transform_to_campbellviewer_database(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\IEA_15MW_fast_large_uncertainties\*', base_output_name='CampbellViewerDatabase_IEA_15MW_fast_large_uncertainties')
    # transform_to_campbellviewer_database(r'C:\Users\verd_he\projects\Torque2024\wtuq\use_cases\campbell_diagram\reference_data\bladed-lin\demo_a_testing_full_FIRST_RUN\*')