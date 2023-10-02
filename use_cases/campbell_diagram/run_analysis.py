"""
Campbell diagram use case main module
"""
import numpy as np
import os
import logging

# from preprocessor import PreProcessor
from wtuq_framework.helperfunctions import save_dict_json, load_dict_json, save_dict_h5py, load_dict_h5py, equal_dicts, \
    get_CPU

from wtuq_framework.uq_framework import Model, UQFramework, ReferenceRunExit
from extra_uq_results_analysis import result_analysis_PCE, result_analysis_OAT, result_analysis_EE, campbell_diagram_with_uncertainty_bands, campbell_diagram_from_OAT


class CampbellDiagramModel(Model):
    """
    Common interface class for stability analysis simulations for all simulation tools, method run() is called
    by uncertainpy.

    Parameters
    ----------
    tool_config : {dict, ConfigObj}
        Tool specific user settings, subsection of the config file
    preprocessor_config : {dict, ConfigObj}
        Preprocessor specific user settings, subsection of the config file
    model_inputs : dict
        Dictionary with content:

        key: run_directory, value: directory where all run information will be assembled, name specified in config
        key: restart_directory, value: pattern for directories with existing results, has to end with //*
        key: restart_h5, value: path to existing uncertainpy result file (.h5)
        key: run_type, value: flag to indicate which kind of simulation (reference, test or full)

    Attributes
    ----------
    simulation_tool : SimulationModel
        Interface to aeroelastic WT simulation tool
    tool_name : string
        Name of the aeroelastic WT simulation tool
    tool_config : {dict, ConfigObj}
        Tool specific user settings, subsection of the config file
    preprocessor_config : {dict, ConfigObj}
        Preprocessor specific user settings, subsection of the config file
    """

    def __init__(self, tool_config, preprocessor_config, model_inputs):
        super(CampbellDiagramModel, self).__init__(model_inputs)

        # variables for the model
        self.tool_name = tool_config['tool_name']
        self.simulation_tool = None
        self.tool_config = tool_config

        # numerical blade data for the simulation tool (specified via arguments to run() )
        self.blade_data = dict()

        self.preprocessor_config = preprocessor_config

    def _preprocessor(self):
        """
        Converts reference structural blade data and spline objects to modified blade input data
        """
        # preprocessor = PreProcessor(self.preprocessor_config)
        # if self.run_type != 'reference':
        #     preprocessor.apply_structural_modifications(self.interpolation_objects)
        # preprocessor.plot_distribution(self.run_directory, True)
        # self.blade_data = preprocessor.export_data()

        self.preprocessed_data = dict()
        if self.run_type != 'reference':
            for param, value in self.input_parameters.items():
                if 'scalar_property' in param:
                    uncertain_param_prefix = param.partition("scalar_property")[0][:-1]
                    self.preprocessed_data[uncertain_param_prefix] = value
                else:
                    print('do something with the interpolation objects')

    def _postprocessor_bladedlin(self, result_dict, variable_mode_tracking=True):
        """
        Select the damping and frequency content of specific mode names
        """
        """
        desired_modes = ['Tower 1st fore-aft mode', 'Tower 1st side-side mode', 'Rotor 1st edgewise backward whirl',
                         'Rotor 1st edgewise forward whirl', 'Rotor 3rd flapwise cosine cyclic',
                         'Rotor 3rd flapwise sine cyclic', 'Low-speed Shaft', 'Rotor 2nd edgewise forward whirl']
        """
        print('Postprocessing is based on hardcoded Campbell diagram reference data')
        # DEMO-A reference Campbell data
        #desired_modes = ['Rotor 1st edgewise backward whirl', 'Rotor 1st edgewise forward whirl',
        #                 'Rotor 2nd edgewise backward whirl', 'Rotor 2nd edgewise forward whirl']
        #reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [1.1947, 1.19734, 1.19349, 1.18938, 1.18822, 1.18762],
        #                                   'Rotor 1st edgewise forward whirl': [1.80307, 1.80559, 1.80177, 1.79754, 1.79616, 1.79535],
        #                                   'Rotor 2nd edgewise backward whirl': [4.60805, 4.60812, 4.6081, 4.60801, 4.60793, 4.60788],
        #                                   'Rotor 2nd edgewise forward whirl': [5.21335, 5.21341, 5.21339, 5.21326, 5.21313, 5.213]}

        # IEA 15 MW reference Campbell data (51 Sections)
        #desired_modes = ['Rotor 1st edgewise backward whirl', 'Rotor 1st edgewise forward whirl',
        #                 'Rotor 2nd edgewise backward whirl', 'Rotor 2nd edgewise forward whirl']
        #reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.576163, 0.569978, 0.570617, 0.571181, 0.56783, 0.56467],
        #                                   'Rotor 1st edgewise forward whirl': [0.810334, 0.821295, 0.822507, 0.819621, 0.815166, 0.811032],
        #                                   'Rotor 2nd edgewise backward whirl': [2.0494, 2.0184, 2.0423, 2.0542, 2.0583, 2.0614],
        #                                   'Rotor 2nd edgewise forward whirl': [2.24289, 2.25599, 2.24389, 2.21513, 2.19358, 2.17557]}

        # IEA 15 MW reference Campbell data (26 Sections) - also suffices for 10 Perturbation Points simulation
        desired_modes = ['Rotor 1st edgewise backward whirl', 'Rotor 1st edgewise forward whirl',
                         'Rotor 2nd edgewise backward whirl', 'Rotor 2nd edgewise forward whirl']
        reference_frequency_progression = {'Rotor 1st edgewise backward whirl': [0.579, 0.572, 0.572, 0.570, 0.569, 0.563],
                                           'Rotor 1st edgewise forward whirl': [0.813, 0.823, 0.824, 0.822, 0.817, 0.816],
                                           'Rotor 2nd edgewise backward whirl': [2.053, 2.034, 2.050, 2.054, 2.061, 2.058],
                                           'Rotor 2nd edgewise forward whirl': [2.257, 2.265, 2.253, 2.235, 2.208, 2.205]}

        # 26 Section Blade - 2-20-2
        #desired_modes = ['Rotor 1st edgewise backward whirl', 'Rotor 1st edgewise forward whirl',
        #                 'Rotor 2nd edgewise backward whirl', 'Rotor 2nd edgewise forward whirl']
        #reference_frequency_progression = {
        #    'Rotor 1st edgewise backward whirl': [0.611, 0.613, 0.618, 0.608, 0.579, 0.571, 0.572, 0.570, 0.569, 0.563],
        #    'Rotor 1st edgewise forward whirl': [0.781, 0.784, 0.789, 0.800, 0.814, 0.823, 0.824, 0.822, 0.820, 0.816],
        #    'Rotor 2nd edgewise backward whirl': [2.063, 2.072, 2.082, 2.079, 2.054, 2.036, 2.050, 2.054, 2.056, 2.058],
        #    'Rotor 2nd edgewise forward whirl': [2.194, 2.206, 2.218, 2.238, 2.257, 2.267, 2.253, 2.235, 2.220, 2.206]}

        reference_frequency_progression_deltas = dict()
        for mode in reference_frequency_progression:
            reference_frequency_progression_deltas[mode] = np.diff(reference_frequency_progression[mode])

        available_mode_names = result_dict['mode_names']
        nr_ws = result_dict['Frequency'].shape[0]

        subset_result_dict = dict()
        subset_result_dict['postprocessing_successful'] = []
        subset_result_dict['postprocessing_failed'] = []
        subset_result_dict['mode_names'] = []
        subset_result_dict['damping'] = np.zeros((nr_ws, len(desired_modes)))
        subset_result_dict['frequency'] = np.zeros((nr_ws, len(desired_modes)))
        for mode_ii, desired_mode_name in enumerate(desired_modes):
            matching_mode_name_indices = np.where(np.array(available_mode_names) == desired_mode_name)[0]
            if desired_mode_name == 'Rotor 2nd edgewise backward whirl':
                matching_mode_name_indices2 = np.where(np.array(available_mode_names) == 'Rotor 2nd edgewise sine cyclic')[0]
                matching_mode_name_indices = np.hstack((matching_mode_name_indices, matching_mode_name_indices2))

            if len(matching_mode_name_indices) == 0:
                print('{} is not available in the result_dict, this mode is no longer used'.format(desired_mode_name))
                subset_result_dict['postprocessing_failed'].append(desired_mode_name)
                subset_result_dict['frequency'][:, mode_ii] = np.ones(nr_ws) * -999
                subset_result_dict['damping'][:, mode_ii] = np.ones(nr_ws) * -999
                continue
            elif len(matching_mode_name_indices) == 1:
                start_col = matching_mode_name_indices[0]
                # start_col = available_mode_names.index(desired_mode_name)
                subset_result_dict['mode_names'].append(desired_mode_name)
            else:
                print('Mode: {} is found multiple times in the result_dict. The last mode will be used as start'.format(desired_mode_name))
                start_col = max(matching_mode_name_indices)  # matching_mode_name_indices[-1]
                subset_result_dict['mode_names'].append(desired_mode_name)

            if variable_mode_tracking:
                subset_result_dict['frequency'][0, mode_ii] = result_dict['Frequency'][0, start_col]
                subset_result_dict['damping'][0, mode_ii] = result_dict['Damping'][0, start_col]

                freq = result_dict['Frequency'][0, start_col]
                current_col = start_col
                # follow this mode progressively
                for ws_ii in range(1, nr_ws):
                    freq_next = result_dict['Frequency'][ws_ii, current_col]
                    freq_expected = freq + reference_frequency_progression_deltas[desired_mode_name][ws_ii-1]
                    if abs(freq_expected-freq_next) > 0.1:
                        print('Probably a problem in mode tracking for mode={} at ws_ii={} , looking for a better frequency branch...'.format(desired_mode_name, ws_ii))

                        # option 1: finding frequency value closest to expected value
                        absolute_freq_diff = np.abs(result_dict['Frequency'][ws_ii, :] - freq_expected)
                        current_col = (absolute_freq_diff).argmin()

                        # option 2: finding branch where 3 next values are closest to expected values
                        print('THIS METHOD HAS TO BE REVIEWED')
                        extra_ws_steps = 3
                        extra_ws_steps = min(extra_ws_steps, nr_ws-ws_ii-1)
                        for ii in range(1, extra_ws_steps+1):
                            freq_expected += reference_frequency_progression_deltas[desired_mode_name][ws_ii-1 + ii]
                            absolute_freq_diff += np.abs(result_dict['Frequency'][ws_ii+ii, :] - freq_expected)

                        current_col = (absolute_freq_diff).argmin()

                        if absolute_freq_diff[current_col] > 0.1:
                            # stop this has to be updated see below
                            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            print('Mode tracking not precise enough for mode: {}'.format(desired_mode_name))
                            print('This iteration is skipped')
                            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            return False, {'frequency': np.ones((nr_ws, len(desired_modes))) * -999,  #np.array([[-999] * nr_ws] * len(desired_modes)),  #   np.[-999]*(nr_ws*len(desired_modes)),
                                           'damping': np.ones((nr_ws, len(desired_modes))) * -999,   # [-999]*(nr_ws*len(desired_modes)),
                                           'mode_names': ['Failed']*len(desired_modes)}

                        print('Continuing on branch of mode={}'.format(available_mode_names[current_col]))

                    freq = result_dict['Frequency'][ws_ii, current_col]

                    subset_result_dict['frequency'][ws_ii, mode_ii] = result_dict['Frequency'][ws_ii, current_col]
                    subset_result_dict['damping'][ws_ii, mode_ii] = result_dict['Damping'][ws_ii, current_col]

            else:
                # last check for mode jumps -> frequency can not jump more than 0.1 Hz more or less than the expected
                # jump between operating points
                freqs = result_dict['Frequency'][:, start_col]
                freqs_expected = np.zeros(freqs.shape)
                freqs_expected[0] = freqs[0]
                freqs_expected[1:] = freqs[:-1] + reference_frequency_progression_deltas[desired_mode_name][:].reshape(freqs[:-1].shape)
                if not np.all(np.abs(freqs_expected - freqs) < 0.1):
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('There seem to be some jumps in the frequency for mode: {}'.format(desired_mode_name))
                    print('This iteration is skipped')
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # subset_result_dict['postprocessing_success'][desired_mode_name] = False
                    subset_result_dict['postprocessing_failed'].append(desired_mode_name)
                    subset_result_dict['frequency'][:, mode_ii] = np.ones(nr_ws) * -999
                    subset_result_dict['damping'][:, mode_ii] = np.ones(nr_ws) * -999

                    #return False, {'frequency': np.ones((nr_ws, len(desired_modes))) * -999,  # np.array([[-999] * nr_ws] * len(desired_modes)),  #   np.[-999]*(nr_ws*len(desired_modes)),
                    #               'damping': np.ones((nr_ws, len(desired_modes))) * -999,  # [-999]*(nr_ws*len(desired_modes)),
                    #               'mode_names': ['Failed'] * len(desired_modes)}
                else:
                    subset_result_dict['postprocessing_successful'].append(desired_mode_name)
                    subset_result_dict['frequency'][:, mode_ii] = result_dict['Frequency'][:, start_col].squeeze()
                    subset_result_dict['damping'][:, mode_ii] = result_dict['Damping'][:, start_col].squeeze()

        subset_result_dict['total_series'] = np.concatenate((subset_result_dict['frequency'].flatten(),
                                                             subset_result_dict['damping'].flatten()))
        return subset_result_dict

    def run(self, **kwargs):
        """
        Method which is executed by uncertainpy with varying sets of input parameters. This is the general interface
        method for a single parameter set. Is run in parallel by default!

        Parameters
        ----------
        kwargs : dict
            Input parameters for a specific iteration of the framework

        Returns
        -------
        time : {float, array, None}
            Time instances where PCE model has to be made
        damping: {float, array, None}
            Damping value (QoI) for the uncertainty quantification
        """

        logger = logging.getLogger('cd.uq.model.run')

        mp_lock = kwargs.pop('mp_lock')  # multi-processing lock
        self.input_parameters = kwargs  # (uncertain + fixed) input_parameters

        # check if restart based on uncertainpy result file is an option
        if self.restart_h5 is True:
            success, damping = self._check_restart_h5(type(self).__name__)
            if success is True:
                return None, damping

        # check if restart based on available iteration_run_directory is an option
        restart_available = False
        if self.restart is True:
            restart_available, result_dict = self._check_restart_options()

        if not restart_available:
            # make unique iteration run directory for this framework iteration
            iteration_run_directory, iteration_id = self._setup_iter_run_dir(logger)

            # convert spline parameters to physical parameters
            logger.debug('Convert parameters to splines')
            self._convert_parameters_to_splines()

            # make blade model
            logger.debug('Preprocess structural parameters')
            self._preprocessor()

            # save_dict_h5py(os.path.join(iteration_run_directory, 'blade_data.hdf5'), self.blade_data)

            logger.debug('Initialize tool interface')
            if mp_lock is not None:
                mp_lock.acquire()
            if self.tool_name == 'bladed-lin':
                from tool_interfaces.bladed_lin_interface import BladedLinModel
                self.simulation_tool = BladedLinModel(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'hawcstab2':
                from tool_interfaces.hawcstab2_interface import HAWCStab2Model
                self.simulation_tool = HAWCStab2Model(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'dummy-tool':
                from tool_interfaces.dummy_tool_interface import DummyToolModel
                self.simulation_tool = DummyToolModel(iteration_run_directory, self.tool_config)
            if mp_lock is not None:
                mp_lock.release()

            try:
                logger.debug('Create tool simulation')
                self.simulation_tool.create_simulation(self.preprocessed_data)

                logger.debug('Run tool simulation')
                self.simulation_tool.run_simulation(mp_lock)

                try:
                    logger.debug('Extract tool results')
                    result_dict = self.simulation_tool.extract_results()
                except IOError as exc:
                    logger.exception('Failed to extract results')
                    return None, None

                # save result dict for possible later use, this could be skipped if we are not interested in re-using
                # this run or in case of memory issues.
                # result_dict could be large depending on how many variables are saved
                save_dict_h5py(os.path.join(iteration_run_directory, 'result_dict.hdf5'), {key: value for key, value in result_dict.items() if key != 'cv_bladedlin_data'})

            except Exception as exc:
                # framework continuation is guaranteed no matter which failure appears in the setup, execution and data
                # extraction of the simulation model for this iteration.
                logger.exception('Iteration uuid {}: Unknown exception in tool interface'.format(iteration_id))
                return None, None

        if self.tool_name == 'dummy-tool':
            time = np.array([1, 3, 4])
            damping = np.array([1, 3, 4])  # dummy damping
            #time = None
            #damping = 1
            campbell_dict = {'TEST': np.array([1, 3, 4]), 'TEST2': ['list'], 'TEST3': 'text'}
            return time, damping, campbell_dict

        if self.tool_name == 'bladed-lin' or self.tool_name == 'hawcstab2':
            try:
                campbell_dict = self._postprocessor_bladedlin(result_dict, variable_mode_tracking=False)
                # campbell_dict['postprocessing_success'] = postprocessing_success
                # if not postprocessing_success:
                #     return None, None, campbell_dict
                time = np.arange(campbell_dict['damping'].shape[0]*2)
            except Exception as exc:
                # framework continuation is guaranteed no matter which failure appears in the postprocessing of the
                # simulation model for this iteration.
                logger.exception('Iteration uuid {}: Unknown exception in postprocessing -> {}'.format(iteration_id, exc))

        if self.run_type == 'reference' or self.run_type == 'test':
            raise ReferenceRunExit

        return time, None, campbell_dict

    @staticmethod
    def first_edge_bw(time, dummy_model_output, campbell_dict):
        if 'Rotor 1st edgewise backward whirl' in campbell_dict['postprocessing_successful']:
            freq_and_damp_joined = np.hstack((campbell_dict['frequency'][:, 0], campbell_dict['damping'][:, 0]))
            return time, freq_and_damp_joined
        else:
            return time, None

    @staticmethod
    def first_edge_fw(time, dummy_model_output, campbell_dict):
        if 'Rotor 1st edgewise forward whirl' in campbell_dict['postprocessing_successful']:
            freq_and_damp_joined = np.hstack((campbell_dict['frequency'][:, 1], campbell_dict['damping'][:, 1]))
            return time, freq_and_damp_joined
        else:
            return time, None

    @staticmethod
    def second_edge_bw(time, dummy_model_output, campbell_dict):
        if 'Rotor 2nd edgewise backward whirl' in campbell_dict['postprocessing_successful']:
            freq_and_damp_joined = np.hstack((campbell_dict['frequency'][:, 2], campbell_dict['damping'][:, 2]))
            return time, freq_and_damp_joined
        else:
            return time, None

    @staticmethod
    def second_edge_fw(time, dummy_model_output, campbell_dict):
        if 'Rotor 2nd edgewise forward whirl' in campbell_dict['postprocessing_successful']:
            freq_and_damp_joined = np.hstack((campbell_dict['frequency'][:, 3], campbell_dict['damping'][:, 3]))
            return time, freq_and_damp_joined
        else:
            return time, None



if __name__ == '__main__':
    """
    Program start
    """
    try:
        use_case_configspec = os.path.join(os.path.dirname(__file__), 'config', 'use_case_config.spec')
    except IOError:
        print('use_case_config.spec has to be defined for the use case')

    framework = UQFramework(use_case_configspec)

    model = CampbellDiagramModel(tool_config=framework.config['use_case']['tool'],
                                 preprocessor_config=framework.config['use_case']['preprocessor'],
                                 model_inputs=framework.give_standard_model_inputs())

    features = [model.first_edge_bw, model.first_edge_fw, model.second_edge_bw, model.second_edge_fw]
    UQResultsAnalysis = framework.main(model, features=features, return_postprocessor=True)

    result_analysis_EE(UQResultsAnalysis)
    # result_analysis_OAT(UQResultsAnalysis)
    # result_analysis_PCE(UQResultsAnalysis)

    # campbell_diagram_from_OAT(UQResultsAnalysis)
    # campbell_diagram_with_uncertainty_bands(UQResultsAnalysis)
