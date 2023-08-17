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
import numpy as np

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
            if 'scalar_property' not in param:
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
        logger = logging.getLogger('wtuq.uq.model._check_restart_options')
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
        logger = logging.getLogger('wtuq.uq.model._check_restart_h5')
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
        self.logger = logging.getLogger('wtuq.uq')
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

    def main(self, model, features=None, return_postprocessor=False):
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
        if self.config['framework']['uncertainty']['uq_method'] == 'morris':
            custom_uq_method = morris_screening
            method = 'custom'
        elif self.config['framework']['uncertainty']['uq_method'] == 'oat':
            custom_uq_method = oat_screening
            method = 'custom'
        elif self.config['framework']['uncertainty']['uq_method'] == 'pc' or \
             self.config['framework']['uncertainty']['uq_method'] == 'mc':
            custom_uq_method = None
            method = self.config['framework']['uncertainty']['uq_method']
        else:
            raise ValueError('UQ method chosen in the config file is not known')

        # set up UQ
        uq = un.UncertaintyQuantification(model=model, parameters=self.parameters, features=features,
                                          CPUs=get_CPU(self.config['framework']['uncertainty']['n_CPU'],
                                                       self.config['framework']['uncertainty']['run_type']),
                                          custom_uncertainty_quantification=custom_uq_method)
        self.logger.info('UQ set up')

        # run it
        try:
            result, U_hat, distribution = uq.quantify(method=method,
                                                      data_folder=os.path.join(self.run_directory, 'uq_results'),
                                                      figure_folder=os.path.join(self.run_directory, 'uq_results'),
                                                      polynomial_order=self.config['framework']['uncertainty'][
                                                          'polynomial_order'],
                                                      nr_collocation_nodes=self.config['framework']['uncertainty'][
                                                          'nr_collocation_nodes'],
                                                      nr_mc_samples=self.config['framework']['uncertainty'][
                                                          'nr_mc_samples'],
                                                      morris_nr_of_repetitions=self.config['framework']['uncertainty'][
                                                          'morris_nr_of_repetitions'],
                                                      morris_oat_linear_disturbance =self.config['framework']['uncertainty'][
                                                          'morris_oat_linear_disturbance'],
                                                      morris_oat_linear_disturbance_factor=self.config['framework']['uncertainty'][
                                                          'morris_oat_linear_disturbance_factor'])

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

            print('Successfully finished')
            if return_postprocessor:
                return uq_plots

        except ReferenceRunExit:
            # makes the framework exit after the first iteration
            print("Reference or test run completed, forced exit from uncertainty framework")
        except Exception as exc:
            self.logger.exception('Unknown exception')
        finally:
            # this is the right place for code that will be executed in any case (e.g. writing a summary file)
            pass

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

            if param_definition[uncertain_parameter]['radial_distribution'] == 'none':

                parameter_name = uncertain_parameter + '_scalar_property'
                mini = float(param_definition[uncertain_parameter]['min'])
                maxi = float(param_definition[uncertain_parameter]['max'])
                if param_definition[uncertain_parameter]['param_distribution'] == 'uniform':
                    parameters[parameter_name] = cp.Uniform(mini, maxi)
                elif param_definition[uncertain_parameter]['param_distribution'] == 'normal':
                    parameters[parameter_name] = cp.Normal(mu=(mini + maxi)/2, sigma=maxi/2-mini/2)

            else:

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

        log_file = os.path.join(run_directory, 'uq_results', 'wtuq_uncertainpy.log')
        print('Logging to {}'.format(log_file))
        logger = logging.getLogger('wtuq.uq')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(levels.get(log_level))
        logger.addHandler(file_handler)

        logger.info('Starting uncertainty analysis')


def morris_screening(self, **kwargs):
    """
    http://www.andreasaltelli.eu/file/repository/Screening_CPC_2011.pdf

    Morris = Local derivative at multiple points
        -> If the local derivatives are large -> large mean value -> large influence of the parameter
        -> If the local derivatives change a lot -> large std -> non-linear fie or interaction with other parameters
    """
    from scipy.stats import qmc
    import numpy as np
    import matplotlib.pyplot as plt

    nr_of_repetitions = kwargs['morris_nr_of_repetitions']  # = Number of starting conditions from which OAT is done
    all_distributions = self.create_distribution()
    sample_means = [distr.mom([1])[0] for distr in all_distributions]

    uncertain_params = self.convert_uncertain_parameters()
    nr_params = len(uncertain_params)
    sampler = qmc.Sobol(d=nr_params)
    sample = sampler.random(n=nr_of_repetitions*2)
    # -> array with :nr_of_repetitions of the reference coordinates and nr_of_repetitions: values as disturbance

    # if linear disturbances are preferred, the nr_of_repetitions: samples are modified
    if kwargs['morris_linear_disturbance'] is True:
        sample[nr_of_repetitions:] = sample[:nr_of_repetitions] + \
                                     kwargs['morris_linear_disturbance_factor'] * sample[nr_of_repetitions:]

    # build the normalized_nodes array, which will has following format:
    # row 1: unmodified reference coordinates 1
    # row 2: disturbed reference coordinate 1, unmodified reference coordinates 1 for all other parameters
    # row 3: disturbed reference coordinate 2, unmodified reference coordinates 1 for all other parameters
    # ...
    # row nr_parameters+1: unmodified reference coordinates 2
    # row nr_parameters+2: disturbed reference coordinate 1, unmodified reference coordinates 2 for all other parameters
    # ...
    normalized_nodes = np.repeat(sample[:nr_of_repetitions], nr_params+1, axis=0)
    for repetition_idx, b_values in enumerate(sample[nr_of_repetitions:, :]):
        np.fill_diagonal(normalized_nodes[repetition_idx * (nr_params+1) + 1:
                                          (repetition_idx+1) * (nr_params+1), :], b_values)

    # scale the nodes with the input distributions
    normalized_nodes = normalized_nodes.T
    nodes = np.zeros(normalized_nodes.shape)
    distr_ranges = [[distr.lower[0], distr.upper[0]] for distr in all_distributions]
    for idx, distr_range in enumerate(distr_ranges):
        nodes[idx, :] = distr_range[0] + (distr_range[1]-distr_range[0]) * normalized_nodes[idx, :]

    # run the simulations
    data = self.runmodel.run(nodes, uncertain_params)

    for feature in data.data:

        """
        # masking 
        # check which iterations are not none or do not have any nan values in the qoi
        # mask the evaluations
        # mask the nodes
        # mask the nr_of_repetitions
        not_nan_mask = []
        for i_eval, evaluation in enumerate(data.data[feature].evaluations):
            if np.any(np.isnan(evaluation)) == False:
                not_nan_mask.append(i_eval)

        if not_nan_mask == []:
            continue"""

        evaluations = np.array(data.data[feature].evaluations[not_nan_mask])
        if evaluations.ndim > 1:
            nr_of_qoi = evaluations.shape[1]
        else:
            nr_of_qoi = 1
        ee = np.zeros((nr_params, nr_of_qoi, nr_of_repetitions))

        for repetition in range(nr_of_repetitions):
            # evaluation difference between disturbed and reference computation
            y_delta = evaluations[repetition * (nr_params+1) + 1: (repetition+1) * (nr_params+1)] - evaluations[repetition * (nr_params+1)]
            # normalized input disturbance
            x_delta = normalized_nodes[:, repetition * (nr_params+1) + 1: (repetition+1) * (nr_params+1)] - normalized_nodes[:, [repetition * (nr_params+1)]]
            if nr_of_qoi > 1:
                ee[:, :, repetition] = (y_delta / x_delta.sum(axis=0).reshape((y_delta.shape[0], 1))).reshape((nr_params, nr_of_qoi))
            else:
                ee[:, :, repetition] = (y_delta / x_delta.sum(axis=0)).reshape((nr_params, nr_of_qoi))

        data.data[feature].ee = ee
        data.data[feature].ee_mean = np.mean(np.abs(ee), axis=-1)
        data.data[feature].ee_std = np.std(np.abs(ee), axis=-1)


    """fig, ax = plt.subplots()

    for idx in range(nr_params):
        print('MEAN {}: {}'.format(idx, data.data[data.model_name].ee_mean[idx]))
        print('STD {}: {}'.format(idx, data.data[data.model_name].ee_std[idx]))

        if idx < 10:
            marker = 'o'
        else:
            marker = '+'
        ax.plot(data.data[data.model_name].ee_mean[idx], data.data[data.model_name].ee_std[idx],
                marker=marker, label='Uncertain param #{}'.format(idx))

    ax.set_xlabel('Mean')
    ax.set_ylabel('Standard Dev.')
    ax.grid()
    ax.legend()
    plt.show()"""


    return data, None, None


def oat_screening(self, **kwargs):
    """
    Hübler:
    "The general concept is to vary one parameter while freezing all others. In most cases, only two values
    (maximum and minimum) are tested. Non-linear effects and interactions between inputs are neglected.
    A sensitivity index for the OAT method for the kth input factor can be defined as the (symmetric) derivative
    with respect to the kth input factor"
    """
    logger = logging.getLogger('wtuq.uq')

    uncertain_params = self.convert_uncertain_parameters()
    nr_params = len(uncertain_params)
    nr_samples_per_param = 2  # lower and upper
    all_distributions = self.create_distribution()
    sample_means = [distr.mom([1])[0] for distr in all_distributions]

    nodes = np.tile(np.array(sample_means).reshape(len(sample_means), 1), nr_params*nr_samples_per_param)
    nodes_mean = np.copy(nodes)

    for idx_param in range(nr_params):
        if kwargs['morris_oat_linear_disturbance'] is True:
            logger.info('The uncertain parameters are disturbed by {} x Distribution.lower and '
                        '{} x Distribution.upper. This makes sure only the linear effect of the uncertain parameter '
                        'is computed.'.format(kwargs['morris_linear_disturbance_factor'],
                                              kwargs['morris_linear_disturbance_factor']))
            nodes[idx_param, idx_param*nr_samples_per_param] = all_distributions[idx_param].lower[0] * kwargs['morris_oat_linear_disturbance_factor']
            nodes[idx_param, idx_param * nr_samples_per_param + 1] = all_distributions[idx_param].upper[0] * kwargs['morris_oat_linear_disturbance_factor']
        else:
            logger.info('The uncertain parameters are disturbed by -Distribution.lower and +Distribution.upper. '
                        'This could give false results for nonlinear functions.')
            nodes[idx_param, idx_param*nr_samples_per_param] = all_distributions[idx_param].lower[0]
            nodes[idx_param, idx_param * nr_samples_per_param + 1] = all_distributions[idx_param].upper[0]

    deltas = np.sum(nodes - nodes_mean, axis=0)

    data = self.runmodel.run(nodes, uncertain_params)

    for feature in data.data:
        s_oat = [(data.data[feature].evaluations[i] - data.data[feature].evaluations[i+1]) / (deltas[i] - deltas[i+1])
                 for i in range(0, nr_params*nr_samples_per_param, 2)]

        # if both low and high were nans, the s_oat will be nan instead of array with nans
        nr_campbell_diagram_points = len(data.data[feature].time)
        for ii in range(len(s_oat)):
            if np.any(np.isnan(s_oat[ii])) == True:
                print('These s_oat result are assumed to be nan: ', s_oat[ii])
                s_oat[ii] = np.ones(nr_campbell_diagram_points) * np.nan

        data.data[feature].nodes = nodes
        data.data[feature].s_oat = s_oat
        data.data[feature].s_oat_mean = np.mean(np.array(s_oat), axis=1)
        data.data[feature].s_oat_max = np.max(np.array(s_oat), axis=1)

    return data, None, None
