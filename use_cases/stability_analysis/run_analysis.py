"""
Stability analysis use case main module
"""
import numpy as np
import os
import logging

from preprocessor import PreProcessor
from postprocessor import PostProcessor
from libDMD import DMDanalysis
from libMBC import MBCTransformation
from wtuq_framework.helperfunctions import save_dict_json, load_dict_json, save_dict_h5py, load_dict_h5py, equal_dicts, \
    get_CPU

from wtuq_framework.uq_framework import Model, UQFramework, ReferenceRunExit


class StabilityAnalysisModel(Model):
    """
    Common interface class for stability analysis simulations for all simulation tools, method run() is called
    by uncertainpy.

    Parameters
    ----------
    tool_config : {dict, ConfigObj}
        Tool specific user settings, subsection of the config file
    preprocessor_config : {dict, ConfigObj}
        Preprocessor specific user settings, subsection of the config file
    postprocessor_config : {dict, ConfigObj}
        Postprocessor specific user settings, subsection of the config file
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
    postprocessor_config : {dict, ConfigObj}
        Postprocessor specific user settings, subsection of the config file
    blade_data : dict
        Dictionary with structural blade data
    """

    def __init__(self, tool_config, preprocessor_config, postprocessor_config, model_inputs):
        super(StabilityAnalysisModel, self).__init__(model_inputs)

        # variables for the model
        self.tool_name = tool_config['tool_name']
        self.simulation_tool = None
        self.tool_config = tool_config

        # numerical blade data for the simulation tool (specified via arguments to run() )
        self.blade_data = dict()

        self.preprocessor_config = preprocessor_config
        self.postprocessor_config = postprocessor_config

    def _preprocessor(self):
        """
        Converts reference structural blade data and spline objects to modified blade input data
        """
        preprocessor = PreProcessor(self.preprocessor_config)
        if self.run_type != 'reference':
            preprocessor.apply_structural_modifications(self.interpolation_objects)
        preprocessor.plot_distribution(self.run_directory, True)
        self.blade_data = preprocessor.export_data()

    def _postprocessor(self, resultdict, iteration_run_directory):
        """
        Defines damping from signals in resultdict, settings defined in config file

        Parameters
        ----------
        resultdict : dict
            Collected simulation results. Key names have to match with var_names specified in the postprocessor
            settings in the config
        iteration_run_directory : string
            Path to directory of this framework iteration

        Returns
        -------
        damping : float
            Damping ratio of the most critical mode, based on time domain signals in resultdict and postprocessor
            settings in postprocessor_config

        Notes
        -----
        IMPORTANT: this method can not be called _postprocess!
        """
        # it seems to be impossible to give default integer or float lists to the configobj
        # therefore the default string lists have to be converted to floats here...
        interp_radpos = [float(radpos) for radpos in self.postprocessor_config['interp_radpos']]
        crop_window = [float(timestamp) for timestamp in self.postprocessor_config['crop_window']]
        window_method = int(self.postprocessor_config['window_method'])

        # initialize postprocessing object
        postprocess = PostProcessor(resultdict, result_dir=iteration_run_directory)

        # define which signal will be analyzed
        # args:
        # - variable name (key of resultdict),
        # - interp_radpos = optional radpos positions for interpolation of 2D array
        # - downsampling = downsample time series
        # - downsampling_frequency = frequency to downsample to
        postprocess.read_results(self.postprocessor_config['var_name'],
                                 interp_radpos=interp_radpos,
                                 downsampling_flag=self.postprocessor_config['downsampling_flag'],
                                 downsampling_frequency=self.postprocessor_config['downsampling_freq'])

        # do some pre-processing on the signals: de-mean, crop_window
        postprocess.prepare_signals(window_method=window_method,
                                    crop_window=crop_window)

        if self.postprocessor_config['MBC_transf'] is True:
            if self.tool_name == 'alaska':  # only alaska has CCW blade numbering (opposite to rotor rotation)
                mbc_transfo = MBCTransformation(postprocess.time, postprocess.signal,
                                                self.postprocessor_config['var_name'], blade_order='CCW')
            else:
                mbc_transfo = MBCTransformation(postprocess.time, postprocess.signal,
                                                self.postprocessor_config['var_name'], blade_order='CW')
            postprocess.signal = mbc_transfo.do_MBC()

        if self.postprocessor_config['damping_method'] == 'DMD':
            # initialize DMD object
            # args:
            # required: time array
            # required: sampling_time
            # required: signal array (1D or 2D, time on axis=0)
            # optional: plotting -> True or False
            # optional: result_dir
            # optional: svd_rank for DMD determination (0 -> let DMD module decide)
            DMDobj = DMDanalysis(postprocess.time,
                                 postprocess.sampling_time,
                                 postprocess.signal,
                                 plotting=self.postprocessor_config['plotting'],
                                 result_dir=iteration_run_directory,
                                 svd_rank=self.postprocessor_config['svd_rank'])

            # do dmd analysis
            DMDobj.do_full_hodmd_analysis()
            damping = -DMDobj.DMDresult.DMDdamping_criticalmode
        else:
            # compare the different damping determination methods (based on peaks, linear fitting, or exponential fitting)
            postprocess.compare_analysis_methods_wrap(plotting=self.postprocessor_config['plotting'])
            if self.postprocessor_config['damping_method'] == 'log_decr':
                damping = -postprocess.damp_ratio_log_decr
            elif self.postprocessor_config['damping_method'] == 'linear_fit':
                damping = -postprocess.damp_ratio_lin_fit
            elif self.postprocessor_config['damping_method'] == 'exp_fit':
                damping = -postprocess.damp_ratio_exp_fit

        return damping

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

        logger = logging.getLogger('quexus.uq.model.run')

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

        # make unique iteration run directory for this framework iteration
        iteration_run_directory, iteration_id = self._setup_iter_run_dir(logger)

        if not restart_available:
            # convert spline parameters to physical parameters
            logger.debug('Convert parameters to splines')
            self._convert_parameters_to_splines()

            # make blade model
            logger.debug('Preprocess structural parameters')
            self._preprocessor()
            save_dict_h5py(os.path.join(iteration_run_directory, 'blade_data.hdf5'), self.blade_data)

            logger.debug('Initialize tool interface')
            if self.tool_name == 'alaska':
                from tool_interfaces.alaska_interface import AlaskaModel
                self.simulation_tool = AlaskaModel(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'bladed':
                from tool_interfaces.bladed_interface import BladedModel
                self.simulation_tool = BladedModel(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'bladed-lin':
                from tool_interfaces.bladed_lin_interface import BladedLinModel
                self.simulation_tool = BladedLinModel(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'hawc2':
                from tool_interfaces.hawc2_interface import HAWC2Model
                self.simulation_tool = HAWC2Model(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'hawcstab2':
                from tool_interfaces.hawcstab2_interface import HAWCStab2Model
                self.simulation_tool = HAWCStab2Model(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'openfast':
                from tool_interfaces.openfast_interface import OpenFASTModel
                self.simulation_tool = OpenFASTModel(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'simpack':
                from tool_interfaces.simpack_interface import SimpackModel
                self.simulation_tool = SimpackModel(iteration_run_directory, self.tool_config)
            elif self.tool_name == 'dummy-tool':
                from tool_interfaces.dummy_tool_interface import DummyToolModel
                self.simulation_tool = DummyToolModel(iteration_run_directory, self.tool_config)

            try:
                logger.debug('Create tool simulation')
                self.simulation_tool.create_simulation(self.blade_data)

                logger.debug('Run tool simulation')
                self.simulation_tool.run_simulation()

                try:
                    logger.debug('Extract tool results')
                    result_dict = self.simulation_tool.extract_results()
                except IOError as exc:
                    logger.exception('Failed to extract results')
                    return None, None

                # save result dict for possible later use, this could be skipped if we are not interested in re-using
                # this run or in case of memory issues.
                # result_dict could be large depending on how many variables are saved
                save_dict_h5py(os.path.join(iteration_run_directory, 'result_dict.hdf5'), result_dict)

            except Exception as exc:
                # framework continuation is guaranteed no matter which failure appears in the setup, execution and data
                # extraction of the simulation model for this iteration.
                logger.exception('Iteration uuid {}: Unknown exception in tool interface'.format(iteration_id))
                return None, None

        # evaluate damping
        if self.tool_name == 'dummy-tool':
            damping = 1  # dummy damping
        elif self.tool_name == 'bladed-lin' or self.tool_name == 'hawcstab2':
            damping = result_dict['damping_criticalmode']
        else:
            try:
                logger.debug('Start postprocessor')
                damping = self._postprocessor(result_dict, iteration_run_directory)
            except Exception as exc:  # todo: make a specific exception
                logger.exception(
                    'Iteration uuid {}: Error in _postprocessor, continue framework execution'.format(iteration_id))
                logger.exception(exc)
                damping = None

        # save damping for easy debugging and analysis
        if damping is not None:
            np.savetxt(os.path.join(iteration_run_directory, 'damping.txt'), np.array([damping]))

        if self.run_type == 'reference' or self.run_type == 'test':
            raise ReferenceRunExit

        return None, damping


if __name__ == '__main__':
    """
    Program start
    """
    try:
        use_case_configspec = os.path.join(os.path.dirname(__file__), 'config', 'use_case_config.spec')
    except IOError:
        print('use_case_config.spec has to be defined for the use case')

    framework = UQFramework(use_case_configspec)

    model = StabilityAnalysisModel(tool_config=framework.config['use_case']['tool'],
                                   preprocessor_config=framework.config['use_case']['preprocessor'],
                                   postprocessor_config=framework.config['use_case']['postprocessor'],
                                   model_inputs=framework.give_standard_model_inputs())

    framework.main(model)
