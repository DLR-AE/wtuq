"""
@author: Otto Braun 
@date: 05.01.2021
"""

import numpy as np
import os
from os import path
import glob
import shutil
from subprocess import call
import logging

from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError


class HAWC2Model(SimulationModel):
    """
    HAWC2 model interface

    Parameters
    ----------
    iteration_run_directory : string
        Path to directory of this framework iteration
    config : dict
        User-specified settings for this tool

    Attributes
    ----------
    config : dict
        User-specified settings for this tool
    iteration_run_directory : string
        Path to directory of this framework iteration
    """
    def __init__(self, iteration_run_directory, config):
        self.iteration_run_directory = iteration_run_directory
        self.config = config
        
    def create_simulation(self, blade_data):
        """
        Setup modified HAWC2 model

        Parameters
        ----------
        blade_data : dict
            Dictionary with structural blade data
        """
        logger = logging.getLogger('quexus.uq.model.run.hawc2model.create_simulation')

        self.htc_master_file_path = self.config['htc_master_file_path']
        self.master_file_path=self.config['master_file_path']

        # copy the master model data
        source_path = self.master_file_path+'\ModelData' 
        destination_path = self.iteration_run_directory+'\ModelData'
        if path.exists(destination_path):
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        
        source_path = self.master_file_path+'\htc_Snippets' 
        destination_path = self.iteration_run_directory+'\htc_Snippets'
        if path.exists(destination_path):
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        
        source_path = self.master_file_path+'\OutputSensors' 
        destination_path = self.iteration_run_directory+'\OutputSensors'
        if path.exists(destination_path):
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        
        source_path = self.master_file_path+'\Wind' 
        destination_path = self.iteration_run_directory+'\Wind'
        if path.exists(destination_path):
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        
        source_path = self.master_file_path+'\Control' 
        destination_path = self.iteration_run_directory+'\Control'
        if path.exists(destination_path):
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        
        # copy batchfiles from master model
        for file in glob.glob(self.master_file_path+'/*.bat'):
            shutil.copy(file, self.iteration_run_directory)
        
        with open(self.htc_master_file_path, 'r') as htc_master_file:
            htc_content = htc_master_file.readlines()
        self.write_st(blade_data)            
        self.modify_htc(htc_content)
        logger.info('Hawc2 Simulation created sucessfully')

    def write_st(self, blade_data):
        """
        Write blade_data to HAWC2 .st file

        Parameters
        ----------
        blade_data : dict
            Dictionary with structural blade data
        """
        logger = logging.getLogger('quexus.uq.model.run.hawc2model.create_simulation.write_st')
        
        if self.config['fpm_bool'] == '0':
        
            st_file= np.c_[blade_data['h2_beam']['arc'],
                           blade_data['h2_beam']['m_pm'],
                           blade_data['h2_beam']['x_cg'],
                           blade_data['h2_beam']['y_cg'],
                           blade_data['h2_beam']['ri_x'],
                           blade_data['h2_beam']['ri_y'],
                           blade_data['h2_beam']['x_sc'],
                           blade_data['h2_beam']['y_sc'],
                           blade_data['h2_beam']['E'],
                           blade_data['h2_beam']['G'],
                           blade_data['h2_beam']['I_x'],
                           blade_data['h2_beam']['I_y'],
                           blade_data['h2_beam']['I_T'],
                           blade_data['h2_beam']['k_x'],
                           blade_data['h2_beam']['k_y'],
                           blade_data['h2_beam']['A'],
                           blade_data['h2_beam']['theta_s'],
                           blade_data['h2_beam']['x_ec'],
                           blade_data['h2_beam']['y_ec']]
                
            header = open(os.path.join(os.getcwd(), 'tool_interfaces', 'header_st_file_hawc2.txt'), 'r')
            header_content = header.read()
            header.close()
            np.savetxt(os.path.join(self.iteration_run_directory, 'st_file.inp'), st_file, newline="\n      ",
                       delimiter="      ", header=header_content, comments='', fmt='%1.12e')
        
        elif self.config['fpm_bool'] == '1':
            
            st_file= np.c_[blade_data['h2_FPM']['arc'],
                           blade_data['h2_FPM']['m_pm'],
                           blade_data['h2_FPM']['x_cg'],
                           blade_data['h2_FPM']['y_cg'],
                           blade_data['h2_FPM']['ri_x'],
                           blade_data['h2_FPM']['ri_y'],
                           blade_data['h2_FPM']['theta_s'],
                           blade_data['h2_FPM']['x_ec'],
                           blade_data['h2_FPM']['y_ec'],
                           blade_data['h2_FPM']['K_11'],
                           blade_data['h2_FPM']['K_12'],
                           blade_data['h2_FPM']['K_13'],
                           blade_data['h2_FPM']['K_14'],
                           blade_data['h2_FPM']['K_15'],
                           blade_data['h2_FPM']['K_16'],
                           blade_data['h2_FPM']['K_22'],
                           blade_data['h2_FPM']['K_23'],
                           blade_data['h2_FPM']['K_24'],
                           blade_data['h2_FPM']['K_25'],
                           blade_data['h2_FPM']['K_26'],
                           blade_data['h2_FPM']['K_33'],
                           blade_data['h2_FPM']['K_34'],
                           blade_data['h2_FPM']['K_35'],
                           blade_data['h2_FPM']['K_36'],
                           blade_data['h2_FPM']['K_44'],
                           blade_data['h2_FPM']['K_45'],
                           blade_data['h2_FPM']['K_46'],
                           blade_data['h2_FPM']['K_55'],
                           blade_data['h2_FPM']['K_56'],
                           blade_data['h2_FPM']['K_66']]

            header_6x6 = open(os.path.join(os.getcwd(),'tool_interfaces','header_st_file_hawc2_6x6.txt'), 'r')
            header_6x6_content = header_6x6.read()
            header_6x6.close()
            np.savetxt(os.path.join(self.iteration_run_directory, 'st_file.inp'), st_file, newline="\n      ",
                       delimiter="      ", header=header_6x6_content, comments='', fmt='%1.12e')

        else:
            logger.exception('fpm_bool not assigned correctly, options are 0 and 1')
            raise ValueError('fpm_bool not assigned correctly, options are 0 and 1')
            
    def modify_htc(self, htc_content):
        """ Modify htc file """
        keywords = self.config["keywords"]
        for i, key in enumerate(keywords):
            key = key.replace('absoulte_path+', self.config["master_file_path"])
            key = key.replace('\\', '/')
            keywords[i] = key

        replacements = self.config["replacements"]
        if self.config['fpm_bool'] == '1':
            for i, key in enumerate(self.config["fpm_keywords"]):
                keywords.append(self.config["fpm_keywords"][i])
                replacements.append(self.config["fpm_replacements"][i])

        for i, key in enumerate(replacements):
            key = key.replace('absoulte_path+', self.config["master_file_path"])
            key = key.replace('\\', '/')
            if key == './st_file.inp;' and self.config['fpm_bool'] == '1':
                key = './st_file.inp;'
                key = key+'\n\t\t\tFPM   1 ;'
            replacements[i] = key
            
        with open(os.path.join(self.iteration_run_directory, "HAWC2.htc"), 'w') as output:
            for idx, lines in enumerate(htc_content):
                for i in range(len(keywords)):
                    if keywords[i] in lines:
                        lines = lines.replace(keywords[i], replacements[i])
                        htc_content[idx] = lines
                output.write(str(lines))

    def run_simulation(self):
        """
        Execute HAWC2 simulation
        """
        exe_homedir = self.config['exe_path']
        exe_name = self.config['exe_name']
        call(exe_homedir+exe_name+' HAWC2.htc', cwd=self.iteration_run_directory)

    def extract_results(self):
        """
        Extract the useful, desired signals which can be used for damping determination.

        Returns
        -------
        result_dict : dict
            Collected simulation results. Key names have to match with var_names specified in the postprocessor
            settings in the config

        Raises
        ------
        FileNotFoundError
            If a .sel file could not be found in the iteration_run_directory

        Notes
        -----
        Takes the .sel file in the iteration_run_directory to create sensor_key,
        these keys are use to look up time series in the .dat-file in the iteration run_directory based on indices
        extracted from the .sel file

        resulting dictionary result dict:
        result_dict[sensor_key]=# time steps
        """

        logger = logging.getLogger('quexus.uq.model.run.hawc2model.extract_results')   
        try:
            idents = []
            sensor_key_list_file = glob.glob(self.iteration_run_directory +  "/**/*.sel", recursive=True)

            with open(sensor_key_list_file[0]) as hawc_keys:
                hawc_keys_content = hawc_keys.readlines()
                hawc_keys_content = hawc_keys_content[12:-1]
                for line in hawc_keys_content:
                    ident = line.split(maxsplit=1)[1].replace('\t', ' ')
                    while '  ' in ident:
                        ident = ident.replace('  ', ' ')
                    idents.append(ident)   
                    
            time_key = 'Time s Time\n'
            tower_ss_key = "Dx coo: tower m elastic Deflection tower Dx Mbdy:tower s= 116.00[m] s/S= 1.00 coo: tower center:default\n"
            tower_fa_key = "Dy coo: tower m elastic Deflection tower Dy Mbdy:tower s= 116.00[m] s/S= 1.00 coo: tower center:default\n"

            torsion_keys = []
            InPlane_keys = []
            OutOfPlane_keys = []
            for i in[1, 2, 3]:
                torsion_keys.append('Rz coo: blade' + str(i) + ' deg elastic Rotation blade' + str(i) + ' Rz Mbdy:blade'+ str(i))
                InPlane_keys.append('Dx coo: blade' + str(i) + ' m elastic Deflection blade' + str(i) + ' Dx Mbdy:blade' + str(i))
                OutOfPlane_keys.append('Dy coo: blade' + str(i) + ' m elastic Deflection blade' + str(i) + ' Dy Mbdy:blade' + str(i))
                
            torsion_all_key = dict()
            InPlane_all_key = dict()
            OutOfPlane_all_key = dict()
            
            for key in torsion_keys:
                torsion_all_key[key] = []
            
            for key in InPlane_keys:
                InPlane_all_key[key] = []
            
            for key in OutOfPlane_keys:
               OutOfPlane_all_key[key] = []
    
            for lines in idents:
                for key in torsion_keys:
                    if key in lines:
                        torsion_all_key[key].append(lines)
                for key in InPlane_keys:
                    if key in lines:
                        InPlane_all_key[key].append(lines)
                for key in OutOfPlane_all_key:
                    if key in lines:
                        OutOfPlane_all_key[key].append(lines)     
            
            radpos = []
            for line in OutOfPlane_all_key[key]:  
                ident = line.split('s=', maxsplit=1)[1]
                ident = float(ident.split('[m]', maxsplit=1)[0])
                radpos.append(ident)
     
            sensor_data_file = glob.glob(self.iteration_run_directory + "/**/*.dat", recursive=True)
            if len(sensor_data_file) > 1:
                raise ValueError("More than one .dat file was found in " + self.iteration_run_directory)
            if len(sensor_data_file) < 1:
                raise ValueError("No .dat file was found in " + self.iteration_run_directory)

            hawc2data = np.loadtxt(sensor_data_file[0])
           
            result_dict = dict()
            result_dict['time'] = hawc2data[:, idents.index(time_key)]
            result_dict['radpos'] = np.transpose(np.asarray(radpos))
            result_dict['towertop_fa'] = hawc2data[:, idents.index(tower_fa_key)]
            result_dict['towertop_ss'] = hawc2data[:, idents.index(tower_ss_key)]
            
            for i, key in enumerate(torsion_all_key):
                result_dict['torsion_b'+str(i+1)] = []
                for single_key in torsion_all_key[key]:
                    result_dict['torsion_b'+str(i+1)].append(hawc2data[:, idents.index(single_key)])
                result_dict['torsion_b'+str(i+1)] = np.transpose(np.asarray(result_dict['torsion_b'+str(i+1)]))
            
            for i, key in enumerate(InPlane_all_key):
                result_dict['deflectionInPlane_b'+str(i+1)] = []
                for single_key in InPlane_all_key[key]:
                    result_dict['deflectionInPlane_b'+str(i+1)].append(hawc2data[:, idents.index(single_key)])
                result_dict['deflectionInPlane_b'+str(i+1)] = np.transpose(np.asarray(result_dict['deflectionInPlane_b'+str(i+1)]))
             
            for i, key in enumerate(OutOfPlane_all_key):
                result_dict['deflectionOutOfPlane_b'+str(i+1)] = []
                for single_key in OutOfPlane_all_key[key]:
                    result_dict['deflectionOutOfPlane_b'+str(i+1)].append(hawc2data[:, idents.index(single_key)])
                result_dict['deflectionOutOfPlane_b'+str(i+1)] = np.transpose(np.asarray(result_dict['deflectionOutOfPlane_b'+str(i+1)]))

            return result_dict

        except FileNotFoundError():
            logger.warning('No .sel file found in ' + self.iteration_run_directory)
            return None

    def parse_h2_st(self, h2_st_file):
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