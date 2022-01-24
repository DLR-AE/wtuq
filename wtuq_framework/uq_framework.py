"""
Uncertainty Quantification framework main module
"""
import argparse
import datetime
import os
import uuid
import glob
from configobj import ConfigObj
from validate import Validator
import logging

import uncertainpy as un
import chaospy as cp

from .splining import QuexusSpline, LinearInterp
from .uq_results_analysis import UQResultsAnalysis
from .helperfunctions import save_dict_json, load_dict_json, load_dict_h5py, equal_dicts, get_CPU


class ReferenceRunExit(Exception):
    """
    Will be raised to exit the uncertainty loop if a reference run is requested.
    """
    pass


class Model(un.Model):
    """
    Uncertainpy Model object.

    Parameters
    ----------
    model_inputs : dict
        Dictionary with content:

        key: run_directory, value: directory where all run information will be assembled, name specified in config
        key: restart_directory, value: pattern for directories with existing results, has to end with //*
        key: restart_h5, value: path to existing uncertainpy result file (.h5)
        key: run_type, value: flag to indicate which kind of simulation (reference, test or full)

    Attributes
    ----------
    interpolation_objects : dict
        Dictionary with interpolation object (NURBS curve or fixed interpolation) for each of the uncertain parameters
    input_parameters : dict
        Uncertain parameter input given by the user in the config file
    run_directory : str
        Directory where all run information will be assembled, name specified in config
    restart : bool
        Flag if restart based on existing run directory has to be attempted
    restart_h5 : bool
        Flag if restart based on uncertainpy result file has to be attempted
    restart_directory : str
        Directory with pre-computed results which should be used for restart of the framework
    restart_h5_file : str
        Path to uncertainpy result file which can be used for re-run
    run_type : str
        'full': standard simulation of the framework
        'reference': only one iteration of the framework, without application of the uncertain parameter
        'test': only one iteration of the framework, with application of the uncertain parameter
    """

    def __init__(self, model_inputs=dict()):
        # mandatory
        super(Model, self).__init__(labels=["Time (s)", "Damping (%)"])

        run_directory = model_inputs['run_directory']
        restart_directory = model_inputs['restart_directory']
        restart_h5 = model_inputs['restart_h5']
        run_type = model_inputs['run_type']

        # splines or linear interpolation objects containing information of scaling of reference data
        self.interpolation_objects = dict()

        # arguments to run()
        self.input_parameters = None

        # directory that collects all iterations
        self.run_directory = run_directory

        # restart based on individual iteration run_directories -> use result_dict.hdf5
        self.restart = False
        self.restart_directory = restart_directory
        if restart_directory is not None:
            self.restart = True
        # restart based on uncertainpy result file (.h5)
        self.restart_h5 = False
        self.restart_h5_file = restart_h5
        if restart_h5 is not None:
            self.restart_h5 = True

        self.run_type = run_type

    def _convert_parameters_to_splines(self):
        """
        Converts user-defined spline/interpolation settings to interpolation objects.

        Input: self.input_parameters, Output: self.interpolation_objects

        Notes
        -----
        Example:
        The parameters
            'mass_distribution_cp01_x': 0.3
            'mass_distribution_cp01_y': 1.05
            'mass_distribution_cp01_a': 0.0
            'shear_center_x_cp01_y': 1.1
            'cog_x_fixed_distr': 0.8
            'cog_x_cp00_fix': 0.0

        end up in an entry
        uncertain_params_dict[mass_distribution] = {'cp01_x': 0.3, 'cp01_y': 1.05, 'cp01_a': 0.0}
        uncertain_params_dict[shear_center_x] = {'cp01_y': 1.1}
        uncertain_params_dict[cog_x] = {'fixed_distr': 0.8, 'cp00_fix': 0.0}

        -> used as input for QuexusSpline and LinearInterp

        See Also
        --------
        wtuq_framework.splining.QuexusSpline
        wtuq_framework.splining.LinearInterp
        """
        uncertain_params_dict = dict()
        for param, value in self.input_parameters.items():
            if 'fixed_distr' in param:
                uncertain_param_prefix = param.partition("fixed_distr")[0][:-1]         # f.ex. mass_distribution
            elif 'cp0' in param:
                uncertain_param_prefix = param.partition("cp0")[0][:-1]                 # f.ex. mass_distribution
            uncertain_param_suffix = param.replace(uncertain_param_prefix + '_', '')    # f.ex. cp01_x
            if uncertain_param_prefix not in uncertain_params_dict:
                uncertain_params_dict[uncertain_param_prefix] = dict()
            uncertain_params_dict[uncertain_param_prefix][uncertain_param_suffix] = value

        for uncertain_param_prefix in uncertain_params_dict:
            if 'fixed_distr' in uncertain_params_dict[uncertain_param_prefix]:
                self.interpolation_objects[uncertain_param_prefix] = LinearInterp(
                    uncertain_params_dict[uncertain_param_prefix])
            else:
                self.interpolation_objects[uncertain_param_prefix] = QuexusSpline(
                    uncertain_params_dict[uncertain_param_prefix])
            # self.interpolation_objects[uncertain_param_prefix].plot()  # for debugging/understanding splines

    def _preprocessor(self):
        """
        preprocessing methods, preprocessor will be case study specific
        """
        pass

    def _postprocessor(self):
        """
        postprocessing methods, postprocessor will be case study specific
        """
        pass

    def _check_restart_options(self):
        """
        Checks if the same uncertain input_parameters have been used earlier somewhere in the restart_directory.
        If yes, load the result_dict

        Returns
        -------
        simulation_found : bool
            Flag if a simulation result was found from which a rerun is possible
        result_dict : dict
            Dictionary with simulation results for rerun
        """
        logger = logging.getLogger('quexus.uq.model._check_restart_options')
        logger.info('Checking restart possibility')
        for dir in glob.glob(self.restart_directory):
            if os.path.abspath(dir) == os.path.abspath(self.run_directory):
                continue
            else:
                try:
                    available_data = load_dict_json(os.path.join(dir, 'input_parameters.txt'))
                except FileNotFoundError:
                    logger.warning('input_parameters.txt not found in {}'.format(dir))
                    available_data = dict()
                # the simple check dict == dict does not seem to be sufficient
                if equal_dicts(self.input_parameters, available_data, precision=1e-7):
                    logger.info('Reusable simulation result found')
                    result_dict = load_dict_h5py(os.path.join(dir, 'result_dict.hdf5'))
                    return True, result_dict
        logger.info('NO reusable simulation result found')
        return False, dict()

    def _check_restart_h5(self, model_name='Model'):
        """
        Checks if the same input_parameters have been used in referenced uncertainpy.h5 file
        If yes, load the uncertainpy evaluation

        Parameters
        ----------
        model_name : string, optional
            Name of the feature which should be found in the uncertainpy.h5 file.
            Default is 'Model'

        Returns
        -------
        result_found : bool
            Flag if a uncertainpy result was found from which a rerun is possible
        eval : {list, None}
            QoI evaluations which were available in the uncertainpy result file
        """
        logger = logging.getLogger('quexus.uq.model._check_restart_h5')
        logger.info('Checking restart possibility baded on uncertainpy .h5 file')

        try:
            uq_results = UQResultsAnalysis.get_UQ_results(self.restart_h5_file)
        except OSError:
            logger.info('Restart file: {} could not be found'.format(self.restart_h5_file))
            logger.info('NO reusable simulation result found')
            return False, None

        uncertain_input_parameters = dict((key, self.input_parameters[key]) for key in uq_results.uncertain_parameters)
        for nodes, eval in zip(uq_results.data[model_name].nodes.T, uq_results.data[model_name].evaluations):
            nodes_as_dict = dict((key, nodes[i]) for i, key in enumerate(uq_results.uncertain_parameters))
            if equal_dicts(uncertain_input_parameters, nodes_as_dict, precision=1e-7):
                logger.info('Reusable uncertainpy result found')
                return True, eval
        logger.info('NO reusable simulation result found')
        return False, None

    def _setup_iter_run_dir(self, logger):
        """
        Setup of a unique directory for a framework iteration.

        Parameters
        ----------
        logger : logger object

        Returns
        -------
        iteration_run_directory : string
            Path to directory for this iteration
        iteration_id : string
            Iteration directory name (last element of path)
        """
        # define iteration run_directory
        # this directory will hold details about the parameters, splining, etc.
        # but it could f.ex. also be the standard directory for the corresponding modified models
        iteration_id = uuid.uuid1()  # id containing a time attribute
        logger.debug('Iteration uuid: {}'.format(iteration_id))

        iteration_run_directory = os.path.join(self.run_directory, str(iteration_id))
        # This function makes directories recursively (the older os.mkdir doesn't.
        # It is only available for Python > v3.2
        os.makedirs(iteration_run_directory, exist_ok=True)

        # copy the dict and save in run_directory
        # This information is necessary for re-runs of the framework without an actual uncertainpy result file. This
        # can be useful to test different postprocessing settings without a full model recomputation or it can be
        # useful to retry a framework run if a number iterations failed.
        save_dict_json(os.path.join(iteration_run_directory, 'input_parameters.txt'), self.input_parameters)

        return iteration_run_directory, iteration_id


class UQFramework():
    """
    Wrapping class around uncertainpy

    Parameters
    ----------
    use_case_configspec : dict
        Specifications for use case config file

    Attributes
    ----------
    config : dict
        Parsed config file
    run_directory : string
        Directory where all run information will be assembled, name specified in config
    logger : logger object
    parameters : dict
        Uncertain input parameters for uncertainpy
    """
    def __init__(self, use_case_configspec):
        args = UQFramework.parse_args()
        self.config = UQFramework.parse_config(args.config, use_case_configspec)

        # add the restart directory pattern to the config
        self.config['framework']['uncertainty']['restart_directories'] = args.reuse
        self.config['framework']['uncertainty']['restart_h5'] = args.reuse_h5

        self.run_directory = UQFramework.setup_run_dir(self.config, args)

        UQFramework.setup_logger(self.config['framework'], self.run_directory)
        self.logger = logging.getLogger('quexus.uq')
        self.logger.info('Model set up')

        # create distributions and collect in a dict + add modification methods to preprocessor_config
        self.parameters, self.config['use_case']['preprocessor']['modification_method'] = UQFramework.convert_parameters(
            param_definition=self.config['framework']['parameters'])

    def give_standard_model_inputs(self):
        """
        Return use-case-independent Model inputs. They will be used to initialize the use case specific Model instance.

        Returns
        -------
        model_inputs : dict
            Dictionary with content:

            key: run_directory, value: directory where all run information will be assembled, name specified in config
            key: restart_directory, value: pattern for directories with existing results, has to end with //*
            key: restart_h5, value: path to existing uncertainpy result file (.h5)
            key: run_type, value: flag to indicate which kind of simulation (reference, test or full)
        """
        return {'run_directory': self.run_directory,
                'restart_directory': self.config['framework']['uncertainty']['restart_directories'],
                'restart_h5': self.config['framework']['uncertainty']['restart_h5'],
                'run_type': self.config['framework']['uncertainty']['run_type']}

    def main(self, model):
        """
        Main function for the sensitivity analysis. Runs uncertainpy uncertainty quantification for the model
        according to uncertainty settings given in the config

        Parameters
        ----------
        model : uncertainpy model object (un.Model)
            Model which is used for the uncertainty quantification

        Raises
        ------
        ReferenceRunExit
            If the run_type=reference or run_type=test, the framework exits after one iteration
        """
        # set up UQ
        uq = un.UncertaintyQuantification(model=model, parameters=self.parameters,
                                          CPUs=get_CPU(self.config['framework']['uncertainty']['n_CPU'],
                                                       self.config['framework']['uncertainty']['run_type']))
        self.logger.info('UQ set up')

        # run it
        try:
            result, U_hat, distribution = uq.quantify(data_folder=os.path.join(self.run_directory, 'uq_results'),
                                                      figure_folder=os.path.join(self.run_directory, 'uq_results'),
                                                      polynomial_order=self.config['framework']['uncertainty'][
                                                          'polynomial_order'],
                                                      nr_collocation_nodes=self.config['framework']['uncertainty'][
                                                          'nr_collocation_nodes'])
            # todo: U_hat and distribution are currently not saved, so a comparison between different runs of the
            # framework is not possible. Two options for future: 1) Save these objects, 2) move enable comparison of
            # different runs within uncertainpy, such that these objects do not have to be saved, but can be recomputed.
            uq.plotting.plot_all(sensitivity=u'total')

            uq_plots = UQResultsAnalysis([os.path.join(self.run_directory, 'uq_results', result.model_name + '.h5')],
                                         [self.config['use_case']['tool']['tool_name']])
            uq_plots.read_results(model_name=result.model_name)
            uq_plots.compare_sobol_indices()
            uq_plots.show_input_parameters_per_iter()
            uq_plots.compare_QoI_evaluations_per_iter()
            uq_plots.surrogate_model_verification()
            uq_plots.plot_surface(U_hat, distribution, model_name=result.model_name)
            uq_plots.plot_distributions(U_hat, distribution, model_name=result.model_name)
        except ReferenceRunExit:
            # makes the framework exit after the first iteration
            print("Reference or test run completed, forced exit from uncertainty framework")
        except Exception as exc:
            self.logger.exception('Unknown exception')
        finally:
            # this is the right place for code that will be executed in any case (e.g. writing a summary file)
            pass

        print('Successfully finished')

    @staticmethod
    def convert_parameters(param_definition):
        """
        Takes the [parameters] section from the config (a nested dict) and creates a flattened dictionary as input for
        uncertainpy. Parameters are either uncertain or fixed according to the config. The nomenclature depends on the
        radial distribution type (fixed distribution or spline).

        Parameters
        ----------
        param_definition : dict
            user defined control points for the radial uncertain parameter distributions

        Returns
        -------
        parameters : dict
            Flattened dictionary as input for uncertainpy
        modification_method : dict
            Specification for each uncertain parameter whether the values are supposed to be added or multiplied with
            the reference value

        Raises
        ------
        ValueError
            If the user specified control points in the config file are not according to convention
        """
        parameters = dict()
        modification_method = dict()
        for uncertain_parameter in param_definition.keys():
            modification_method[uncertain_parameter] = param_definition[uncertain_parameter]['method']
            if param_definition[uncertain_parameter]['radial_distribution'] == 'fixed':
                parameter_name = uncertain_parameter + '_fixed_distr'
                if param_definition[uncertain_parameter]['param_distribution'] == 'uniform':
                    parameters[parameter_name] = cp.Uniform(0, 1)
                elif param_definition[uncertain_parameter]['param_distribution'] == 'normal':
                    parameters[parameter_name] = cp.Normal(mu=0.5, sigma=0.5)
                else:
                    raise ValueError('Only <uniform> or <normal> parameter distributions are allowed')
            for control_point in param_definition[uncertain_parameter]:
                if control_point[:2] != 'cp':
                    continue
                for values in param_definition[uncertain_parameter][control_point]:
                    mini = None
                    maxi = None
                    fixi = None
                    if param_definition[uncertain_parameter][control_point][values].keys() == ['max', 'min'] or\
                       param_definition[uncertain_parameter][control_point][values].keys() == ['min', 'max'] or\
                       param_definition[uncertain_parameter][control_point][values].keys() == ['fix']:

                        for value_type in param_definition[uncertain_parameter][control_point][values]:
                            if value_type == 'min':
                                mini = float(param_definition[uncertain_parameter][control_point][values][value_type])
                            if value_type == 'max':
                                maxi = float(param_definition[uncertain_parameter][control_point][values][value_type])
                            if value_type == 'fix':
                                fixi = float(param_definition[uncertain_parameter][control_point][values][value_type])
                    else:
                        raise ValueError(control_point + '_' + values
                                         + ' not specified correctly, options are: min, max, fix')

                    if param_definition[uncertain_parameter]['radial_distribution'] == 'spline':
                        parameter_name = uncertain_parameter + '_' + control_point + '_' + values[0]
                        if mini is not None and maxi is not None:
                            if param_definition[uncertain_parameter]['param_distribution'] == 'uniform':
                                parameters[parameter_name] = cp.Uniform(mini, maxi)
                            elif param_definition[uncertain_parameter]['param_distribution'] == 'normal':
                                parameters[parameter_name] = cp.Normal(mu=(mini+maxi)/2, sigma=maxi/2-mini/2)
                            else:
                                raise ValueError('Only <uniform> or <normal> parameter distributions are allowed')

                        elif (mini is not None) and maxi is None:
                            raise ValueError('Minimum value min given but not maximal value, specify max for '
                                             + control_point + '_' + values)
                        elif (maxi is not None) and mini is None:
                            raise ValueError('Maximum value max given but not minimal value, specify min for '
                                             + control_point + '_' + values)
                        else:
                            parameters[parameter_name] = fixi
                    elif param_definition[uncertain_parameter]['radial_distribution'] == 'fixed':
                        if mini is not None and maxi is not None:
                            parameter_name = uncertain_parameter + '_' + control_point + '_' + values[0] + '_min'
                            parameters[parameter_name] = mini
                            parameter_name = uncertain_parameter + '_' + control_point + '_' + values[0] + '_max'
                            parameters[parameter_name] = maxi
                        elif (mini is not None) and maxi is None:
                            raise ValueError('Minimum value min given but not maximal value, specify max for '
                                             + control_point + '_' + values)
                        elif (maxi is not None) and mini is None:
                            raise ValueError('Maximum value max given but not minimal value, specify min for '
                                             + control_point + '_' + values)
                        else:
                            parameter_name = uncertain_parameter + '_' + control_point + '_' + values[0] + '_fix'
                            parameters[parameter_name] = fixi

        parameters = un.Parameters(parameters=parameters)
        return parameters, modification_method

    @staticmethod
    def parse_args():
        """
        Parse command line arguments

        Returns
        -------
        args : argparse object
            Command line arguments
        """
        parser = argparse.ArgumentParser(
                            description='Run QuexUS uncertainty quantification')
        parser.add_argument('-c', '--config', metavar='cfg', type=str, nargs='?',
                            help='Config file for simulation parameters', default='config.txt')
        parser.add_argument('-r', '--reuse', type=str, metavar='restart directory pattern', nargs='?',
                            help='Directory name pattern with existing simulation result to reuse')
        parser.add_argument('-rh5', '--reuse_h5', type=str, metavar='restart .h5 file', nargs='?',
                            help='Path name to existing simulation result (.h5) to reuse')
        parser.add_argument('-f', '--force_overwrite', default=False, action='store_true',
                            help='Force writing to run_directory if it already exists')
        args = parser.parse_args()

        return args

    @staticmethod
    def parse_config(config_file, use_case_configspec):
        """
        Parse config file

        Parameters
        ----------
        config_file : string
            Path to user specified config settings
        use_case_configspec : dict
            Use case specific config specification

        Returns
        -------
        config : ConfigObj
            Parsed config file

        Raises
        ------
        ValueError
            If the validation of the config file failed
        """
        with open(os.path.join(os.path.dirname(__file__), 'config.spec')) as f_framework_configspec:
            lines = f_framework_configspec.readlines()

        with open(use_case_configspec) as f_use_case_configspec:
            lines = lines + f_use_case_configspec.readlines()

        # read config (simulation tool, parameters), validate against config.spec
        config_spec = ConfigObj(infile=lines,
                                interpolation=False,
                                list_values=False,
                                _inspec=True,
                                file_error=True)

        if not os.path.isfile(config_file):
            raise FileNotFoundError('Config file - {} - not found, uq framework run aborted'.format(config_file))

        config = ConfigObj(config_file, configspec=config_spec)
        validation_result = config.validate(Validator())
        if validation_result is not True:
            raise ValueError('Validation failed (result: {0})'.format(validation_result))

        return config

    @staticmethod
    def setup_run_dir(config=dict(), args=None):
        """
        Setup working directory for this framework run

        Parameters
        ----------
        config : dict
            User specified config settings
        args : argparse object
            Command line arguments

        Returns
        -------
        run_directory : string
            Path to the directory for this framework execution

        Raises
        ------
        KeyError
            If config does not have required data to set up run_directory
        """
        try:
            base_dir = os.path.join(os.path.abspath(config['framework']['uncertainty']['base_directory']))
            run_directory = os.path.join(base_dir, config['framework']['uncertainty']['run_directory'])
        except KeyError:
            print('Config file not complete enough for setting up run directory')
            raise

        # create the new run_directory for this framework
        # if a directory with the same name exists
        # - it will be overwritten if the force_overwrite argument is given in the command line
        # - the results will be written to <run_directory>_NEW
        if os.path.isdir(run_directory) is True and args.force_overwrite is False:
            run_directory = run_directory + '_NEW'
        os.makedirs(run_directory, exist_ok=True)
        os.makedirs(os.path.join(run_directory, 'uq_results'), exist_ok=True)
        # save config to run_directory
        config_file_to_save = datetime.datetime.now().isoformat().replace(':', '-') + ".config"
        with open(os.path.join(run_directory, 'uq_results', config_file_to_save), 'wb') as config_out_file:
            config.write(config_out_file)

        return run_directory

    @staticmethod
    def setup_logger(config=dict(), run_directory='.'):
        """
        Setup of logger object

        Parameters
        ----------
        config : dict
            User specified config settings
        run_directory : string, optional
            Path to the directory for this framework execution
            Default is '.'

        Returns
        -------
        uncertain_parameters : list
            A list with the name of all uncertain parameters.

        Raises
        ------
        KeyError
            If config does not contain required information to setup logger
        """
        try:
            log_level = config['uncertainty']['log_level']
        except KeyError:
            print('Config file not complete enough for setting up logger')
            raise

        levels = {
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warn': logging.WARNING,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG
        }

        log_file = os.path.join(run_directory, 'uq_results', 'quexus_uncertainpy.log')
        print('Logging to {}'.format(log_file))
        logger = logging.getLogger('quexus.uq')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(levels.get(log_level))
        logger.addHandler(file_handler)

        logger.info('Starting uncertainty analysis')
