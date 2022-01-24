"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 25.11.2020
"""
import subprocess
import os
import numpy as np
from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError

from simpack_postprocess import extract_data
from simpack_preprocess import preprocess_simpack


def parse_h2_st(h2_st_file):
    """
    Read a HAWC2 .st file

    Parameters
    ----------
    h2_st_file : string
        Path to HAWC2 structural blade data file

    Returns
    -------
    h2_st : dict
        HAWC2 blade data

    Notes
    -----
    HAWC2 structural parameters:
        arc
        m_pm
        x_cg
        y_cg
        ri_x
        ri_y
        x_sc
        y_sc
        E
        G
        I_x
        I_y
        I_T
        k_x
        k_y
        A
        theta_s
        x_ec
        y_ec
    """
    h2_st = dict()
    table_struct_h2 = np.loadtxt(h2_st_file, skiprows=8)
    h2_st['arc'] = table_struct_h2[:, 0]
    h2_st['m_pm'] = table_struct_h2[:, 1]
    h2_st['x_cg'] = table_struct_h2[:, 2]
    h2_st['y_cg'] = table_struct_h2[:, 3]
    h2_st['ri_x'] = table_struct_h2[:, 4]
    h2_st['ri_y'] = table_struct_h2[:, 5]
    h2_st['x_sc'] = table_struct_h2[:, 6]
    h2_st['y_sc'] = table_struct_h2[:, 7]
    h2_st['E'] = table_struct_h2[:, 8]
    h2_st['G'] = table_struct_h2[:, 9]
    h2_st['I_x'] = table_struct_h2[:, 10]
    h2_st['I_y'] = table_struct_h2[:, 11]
    h2_st['I_T'] = table_struct_h2[:, 12]
    h2_st['k_x'] = table_struct_h2[:, 13]
    h2_st['k_y'] = table_struct_h2[:, 14]
    h2_st['A'] = table_struct_h2[:, 15]
    h2_st['theta_s'] = table_struct_h2[:, 16]
    h2_st['x_ec'] = table_struct_h2[:, 17]
    h2_st['y_ec'] = table_struct_h2[:, 18]

    return h2_st


class SimpackModel(SimulationModel):
    """
    Bladed model interface

    Parameters
    ----------
    run_directory : string
        Path to directory of this framework iteration
    config : dict
        User-specified settings for this tool

    Attributes
    ----------
    ref_model : string
        Path to reference model directory
    main_model_name : string
        Name of the main model
    run_directory : string
        Path to directory of this framework iteration
    hawc2_to_simbeam_ref_excel : string
        Path reference excel file with hawc2 to simbeam conversion
    simbeamgen_dir : string
        Path to SIMBEAM-generator directory
    path_to_simbeam_config : string
        Path to SIMBEAM config file
    simpack_slv_exe : string
        Path to Simpack-solve executable
    run_directory : string
        Path to directory of this framework iteration
    run_scripts_directory : string
        Path to directory with additional Simpack pre- and post-processor scripts
    """

    def __init__(self, run_directory, config):
        # these paths have to go in the config
        self.ref_model = config['reference_model']
        self.main_model_name = config['main_model_name']
        self.hawc2_to_simbeam_ref_excel = config['hawc2_to_simbeam_ref_excel']
        self.simbeamgen_dir = config['simbeamgen_dir']
        self.path_to_simbeam_config = config['path_to_simbeam_config']
        self.simpack_slv_exe = config['simpack_slv_exe']
        self.run_directory = run_directory
        self.run_scripts_directory = config['run_scripts_directory']

    def create_simulation(self, blade_data):
        """
        Creates a new Simpack model (identified via path and file name).

        Takes the necessary information from blade_data (which is redundant)

        Parameters
        ----------
        blade_data : dict
            Dictionary with structural blade data
        """
        # copy reference model and modified .subvar to run_directory
        preprocess_simpack(work_dir=self.run_directory,
                           ref_model=self.ref_model,
                           destination_dir=self.run_directory,
                           blade_data=blade_data['h2_beam'],
                           simbeamgen_config_template=self.path_to_simbeam_config,
                           hawc2_to_simbeam_ref_excel=self.hawc2_to_simbeam_ref_excel,
                           simbeamgen_dir=self.simbeamgen_dir)

    def run_simulation(self):
        """
        Model interface for a single execution of the model created before.
        """
        main_model_file = os.path.abspath(os.path.join(self.run_directory,
                                                       'main_model',
                                                       self.main_model_name))
        set_aerodyn_run_equilibrium_sjs = os.path.abspath(os.path.join(self.run_scripts_directory,
                                                       'set_aerodyn_input_dir_run_static_equilibrium.sjs'))        
        set_aerodyn_sjs = os.path.abspath(os.path.join(self.run_scripts_directory,
                                                       'set_aerodyn_input_dir.sjs')) 

        set_steady_aerodyn_run_equilibrium_command = '{} -s {} {} "\\"steady12\\""'.format(self.simpack_slv_exe, set_aerodyn_run_equilibrium_sjs, main_model_file)
        set_beddoes_aerodyn_command = '{} -s {} {} "\\"dynamic12\\""'.format(self.simpack_slv_exe, set_aerodyn_sjs, main_model_file)
        static_equil_command = '{} --static-equilibrium {}'.format(self.simpack_slv_exe, main_model_file)
        time_int_command = '{} {}'.format(self.simpack_slv_exe, main_model_file)

        # run static equilibrium (AeroDyn model has to be STEADY)
        subprocess.check_call(set_steady_aerodyn_run_equilibrium_command, shell=True)
        # subprocess.check_call(static_equil_command, shell=True)
        # run time integration (AeroDyn model returned to BEDDOES)
        subprocess.check_call(set_beddoes_aerodyn_command, shell=True)
        subprocess.check_call(time_int_command, shell=True)       

    def extract_results(self, save_results=False):
        """
        Read the result file (.mat) and extract the relevant information (time series).

        Returns
        -------
        result_dict : dict
            Collected simulation results. Key names have to match with var_names specified in the postprocessor
            settings in the config
        """
        print('start_reading_mat')	
        full_result_dict = extract_data(os.path.join(self.run_directory,
                                                     'main_model',
                                                     'results',
                                                     'IWT_75_164.mat'), save_results)
        print('finished reading mat')						     
        result_dict = dict()
        result_dict['time'] = full_result_dict['time']
        result_dict['radpos'] = full_result_dict['radpos']
        result_dict['torsion_b1'] = full_result_dict['torsion_b1']
        result_dict['torsion_b2'] = full_result_dict['torsion_b2']
        result_dict['torsion_b3'] = full_result_dict['torsion_b3']
        result_dict['deflectionInPlane_b1'] = full_result_dict['deflectionInPlane_b1']
        result_dict['deflectionInPlane_b2'] = full_result_dict['deflectionInPlane_b2']
        result_dict['deflectionInPlane_b3'] = full_result_dict['deflectionInPlane_b3']
        result_dict['deflectionOutOfPlane_b1'] = full_result_dict['deflectionOutOfPlane_b1']
        result_dict['deflectionOutOfPlane_b2'] = full_result_dict['deflectionOutOfPlane_b2']
        result_dict['deflectionOutOfPlane_b3'] = full_result_dict['deflectionOutOfPlane_b3']
        result_dict['towertop_fa'] = full_result_dict['towertop_fa']
        result_dict['towertop_ss'] = full_result_dict['towertop_ss']

        return result_dict