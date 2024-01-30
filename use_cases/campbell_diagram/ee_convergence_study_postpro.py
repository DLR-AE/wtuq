"""
Postprocess the EE convergence results
"""
import matplotlib.pyplot as plt
import numpy as np

from wtuq_framework.uq_results_analysis import UQResultsAnalysis

features = ['edge_mode_one', 'edge_mode_two', 'edge_mode_three', 'edge_mode_four',
            'edge_mode_five', 'edge_mode_six', 'edge_mode_seven']
nr_params = 11
total_nr_repetitions = 50
nr_of_qoi = 50  # nr of qoi per feature

ee = np.zeros((len(features), total_nr_repetitions, nr_params, nr_of_qoi))
ee_mean_max = np.zeros((len(features), total_nr_repetitions, nr_params))
ee_mean_mean = np.zeros((len(features), total_nr_repetitions, nr_params))
ee_std_max = np.zeros((len(features), total_nr_repetitions, nr_params))
ee_std_mean = np.zeros((len(features), total_nr_repetitions, nr_params))

for i_morris_repetition in range(0, total_nr_repetitions):
    uq_plots = UQResultsAnalysis([r'./results/morris_convergence_study_5E-3_best-practice_skip61/IEA_15MW_morris_5E-3_{}/uq_results/CampbellDiagramModel.h5'.format(i_morris_repetition+1)],
                                 ['hawcstab2'])

    uq_results = uq_plots.get_UQ_results(uq_plots.uncertainpy_results[0])

    uq_plots.uncertain_param_names[uq_plots.uncertainpy_results[0]] = uq_results.uncertain_parameters

    for i_feat, feature in enumerate(features):

        ee[i_feat, i_morris_repetition, :, :] = uq_results.data[feature].ee[:, :, -1]
        ee_mean_max[i_feat, i_morris_repetition, :] = uq_results.data[feature].ee_mean_max
        ee_mean_mean[i_feat, i_morris_repetition, :] = uq_results.data[feature].ee_mean_mean
        ee_std_max[i_feat, i_morris_repetition, :] = uq_results.data[feature].ee_std_max
        ee_std_mean[i_feat, i_morris_repetition, :] = uq_results.data[feature].ee_std_mean


ee_convergence_fig, ee_convergence_ax = plt.subplots(len(features), 4, figsize=(19, 11), sharex=True)

for i_feat, feature in enumerate(features):

    ee_convergence_ax[i_feat, 0].plot(np.arange(1, total_nr_repetitions + 1), ee_mean_max[i_feat, :, :])
    ee_convergence_ax[i_feat, 1].plot(np.arange(1, total_nr_repetitions + 1), ee_mean_mean[i_feat, :, :])
    ee_convergence_ax[i_feat, 2].plot(np.arange(1, total_nr_repetitions + 1), ee_std_max[i_feat, :, :])
    ee_convergence_ax[i_feat, 3].plot(np.arange(1, total_nr_repetitions + 1), ee_std_mean[i_feat, :, :])

    ee_convergence_ax[i_feat, 0].set_ylabel(feature)

    ee_convergence_ax[i_feat, 0].grid()
    ee_convergence_ax[i_feat, 1].grid()
    ee_convergence_ax[i_feat, 2].grid()
    ee_convergence_ax[i_feat, 3].grid()

ee_convergence_ax[0, 0].set_title('Mean Max. EE')
ee_convergence_ax[0, 1].set_title('Mean Mean EE')
ee_convergence_ax[0, 2].set_title('Std Max. EE')
ee_convergence_ax[0, 3].set_title('Std Mean EE')

ee_convergence_ax[3, 0].set_xlabel('Nr. of EE repetitions')

ee_convergence_fig.tight_layout()
# ee_convergence_fig.savefig(os.path.join(uq_plots.output_dir, 'EE - {}.png'.format(feature)), bbox_inches='tight', dpi=300)
# plt.close(ee_convergence_fig)
plt.show()



# EE convergence fig per uncertain param
for uncertain_param_idx in range(nr_params):
    ee_convergence_fig, ee_convergence_ax = plt.subplots(len(features), 4, figsize=(19, 11), sharex=True)

    for i_feat, feature in enumerate(features):

        ee_convergence_ax[i_feat, 0].plot(np.arange(1, total_nr_repetitions + 1), ee[i_feat, :, uncertain_param_idx, :], '--')
        ee_convergence_ax[i_feat, 1].plot(np.arange(1, total_nr_repetitions + 1), ee[i_feat, :, uncertain_param_idx, :], '--')

        ee_convergence_ax[i_feat, 0].plot(np.arange(1, total_nr_repetitions + 1), ee_mean_max[i_feat, :, uncertain_param_idx])
        ee_convergence_ax[i_feat, 1].plot(np.arange(1, total_nr_repetitions + 1), ee_mean_mean[i_feat, :, uncertain_param_idx])
        ee_convergence_ax[i_feat, 2].plot(np.arange(1, total_nr_repetitions + 1), ee_std_max[i_feat, :, uncertain_param_idx])
        ee_convergence_ax[i_feat, 3].plot(np.arange(1, total_nr_repetitions + 1), ee_std_mean[i_feat, :, uncertain_param_idx])

        ee_convergence_ax[i_feat, 0].set_ylabel(feature)

        ee_convergence_ax[i_feat, 0].grid()
        ee_convergence_ax[i_feat, 1].grid()
        ee_convergence_ax[i_feat, 2].grid()
        ee_convergence_ax[i_feat, 3].grid()

    ee_convergence_ax[0, 0].set_title('Mean Max. EE')
    ee_convergence_ax[0, 1].set_title('Mean Mean EE')
    ee_convergence_ax[0, 2].set_title('Std Max. EE')
    ee_convergence_ax[0, 3].set_title('Std Mean EE')

    ee_convergence_ax[3, 0].set_xlabel('Nr. of EE repetitions')

    ee_convergence_fig.tight_layout()
    # ee_convergence_fig.savefig(os.path.join(uq_plots.output_dir, 'EE - {}.png'.format(feature)), bbox_inches='tight', dpi=300)
    # plt.close(ee_convergence_fig)
    plt.show()




