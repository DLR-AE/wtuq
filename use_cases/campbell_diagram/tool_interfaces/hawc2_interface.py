import numpy as np
import os
from os import path
import glob
import shutil
from subprocess import call
import logging
from scipy.interpolate import interp1d

from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError

from threading import Timer


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
        
    def create_simulation(self, preprocessed_data):
        """
        Setup modified HAWC2 model

        Parameters
        ----------
        blade_data : dict
            Dictionary with structural blade data
        """
        # logger = logging.getLogger('quexus.uq.model.run.hawc2model.create_simulation')
        self.master_model_directory_path = self.config['master_model_directory_path']
        self.main_model_htc_file = os.path.join(self.iteration_run_directory, 'htc', self.config['main_model_htc_file'])
        self.shaft_st_file = os.path.join(self.iteration_run_directory, 'data', self.config['data_shaft_st_file'])
        self.tower_st_file = os.path.join(self.iteration_run_directory, 'data', self.config['data_tower_st_file'])
        self.blade_geom_file = os.path.join(self.iteration_run_directory, 'data', self.config['data_blade_geom_file'])
        self.blade_st_file = os.path.join(self.iteration_run_directory, 'data', self.config['data_blade_st_file'])
        self.pc_file = os.path.join(self.iteration_run_directory, 'data', self.config['pc_file'])

        ######### 1) Copy full model to iteration directory
        source_path = self.master_model_directory_path
        self.master_model_name = os.path.basename(os.path.normpath(self.master_model_directory_path))
        shutil.copytree(source_path, self.iteration_run_directory, dirs_exist_ok=True)

        ######### 1) Read in .htc file
        with open(self.main_model_htc_file, 'r') as htc_main_file:
            self.htc_content = htc_main_file.readlines()

        ######### 2) modify .htc file or data files
        if 'blade_chord_root' in preprocessed_data or 'blade_chord_tip' in preprocessed_data:
            self.modify_blade_geom(preprocessed_data)

        if any(param in preprocessed_data for param in ['blade_twist', 'blade_na_x', 'blade_na_y']):
            self.modify_blade_ref_axis(preprocessed_data)

        if any(param in preprocessed_data for param in ['blade_flap_stiffness',
                                                        'blade_edge_stiffness',
                                                        'blade_pa_orientation',
                                                        'blade_mass',
                                                        'blade_cog_x',
                                                        'blade_cog_y',
                                                        'blade_sc_x',
                                                        'blade_sc_y']):
            self.modify_blade_structure(preprocessed_data)

        if 'polar_clalpha' in preprocessed_data or 'polar_alpha_max' in preprocessed_data:
            # self.modify_polars(preprocessed_data)
            self.modify_polars_new(preprocessed_data)

        if 'polar_cd0' in preprocessed_data:
            self.modify_polars_cd(preprocessed_data)

        for param, value in preprocessed_data.items():
            if param == 'nacelle_mass':
                self.modify_nacelle_mass(value)

            elif param == 'nacelle_cog_vertical':
                self.modify_nacelle_cog_vertical(value)

            elif param == 'nacelle_cog_horizontal':
                self.modify_nacelle_cog_horizontal(value)

            elif param == 'nacelle_yaw_inertia':
                self.modify_nacelle_yaw_inertia(value)

            elif param == 'nacelle_nodding_inertia':
                self.modify_nacelle_nodding_inertia(value)

            elif param == 'nacelle_rolling_inertia':
                self.modify_nacelle_rolling_inertia(value)

            elif param == 'cone_angle':
                self.modify_cone_angle(value)

            elif param == 'LSS_stiffness':
                self.modify_LSS_stiffness(value)

            elif param == 'LSS_damping':
                self.modify_LSS_damping(value)

            elif param == 'tower_bending_stiffness':
                self.modify_tower_bending_stiffness(value)

            elif param == 'tower_torsional_stiffness':
                self.modify_tower_torsional_stiffness(value)

            elif param == 'tower_shear_stiffness':
                self.modify_tower_shear_stiffness(value)

            elif param == 'tower_mass':
                self.modify_tower_mass(value)

            elif param == 'ds_a1':
                self.modify_ds_a1(value)

            elif param == 'ds_a2':
                self.modify_ds_a2(value)

            elif param == 'ds_b1':
                self.modify_ds_b1(value)

            elif param == 'ds_b2':
                self.modify_ds_b2(value)

            elif param == 'ds_taupre':
                self.modify_ds_taupre(value)

            elif param == 'ds_taubly':
                self.modify_ds_taubly(value)

            elif 'blade_damping' in param:
                self.modify_blade_damping(value, int(param.split('_')[-1]))

            elif 'tower_damping' in param:
                self.modify_tower_damping(value, int(param.split('_')[-1]))

            else:
                continue

        with open(self.main_model_htc_file, 'w') as htc_main_file:
            htc_main_file.writelines(self.htc_content)

    def find_anchor(self, file_content, anchor):
        for idx, line in enumerate(file_content):
            if anchor in line:
                return idx, line

        print('Anchor {} not found in HAWC file'.format(anchor))
        return None, None

    def modify_line(self, anchor, split_idx, value, line_idx=None, line=None):
        if line_idx is None:
            line_idx, line = self.find_anchor(self.htc_content, anchor)
        line_split = line.split()
        orig_value = float(line_split[split_idx])
        line_split[split_idx] = str(orig_value * (1 + value))
        self.htc_content[line_idx] = ' '.join(line_split)+'\n'

    def modify_st_file(self, st_file, line_indices, column_idx, value):
        with open(st_file, 'r') as st_f:
            st_content = st_f.readlines()

        for line_idx in line_indices:
            line = st_content[line_idx]

            line_split = line.split()
            orig_value = float(line_split[column_idx])
            line_split[column_idx] = str(orig_value * (1 + value))
            st_content[line_idx] = ' '.join(line_split)+ '\n'

        with open(st_file, 'w') as st_f:
            st_f.writelines(st_content)

    def modify_nacelle_mass(self, value):
        """
        """
        anchor = '#nacelle_concentrated_mass'
        self.modify_line(anchor, 5, value)

    def modify_nacelle_cog_vertical(self, value):
        """
        """
        anchor = '#nacelle_concentrated_mass'
        self.modify_line(anchor, 4, value)

    def modify_nacelle_cog_horizontal(self, value):
        """
        """
        anchor = '#nacelle_concentrated_mass'
        self.modify_line(anchor, 3, value)

    def modify_nacelle_yaw_inertia(self, value):
        """
        """
        anchor = '#nacelle_concentrated_mass'
        self.modify_line(anchor, 8, value)

    def modify_nacelle_nodding_inertia(self, value):
        """
        """
        anchor = '#nacelle_concentrated_mass'
        self.modify_line(anchor, 6, value)

    def modify_nacelle_rolling_inertia(self, value):
        """
        """
        anchor = '#nacelle_concentrated_mass'
        self.modify_line(anchor, 7, value)

    def modify_ds_a1(self, value):
        """
        """
        anchor = '#a1'
        self.modify_line(anchor, 1, value)

    def modify_ds_a2(self, value):
        """
        """
        anchor = '#a2'
        self.modify_line(anchor, 1, value)

    def modify_ds_b1(self, value):
        """
        """
        anchor = '#b1'
        self.modify_line(anchor, 1, value)

    def modify_ds_b2(self, value):
        """
        """
        anchor = '#b2'
        self.modify_line(anchor, 1, value)

    def modify_ds_taupre(self, value):
        """
        """
        anchor = '#taupre'
        self.modify_line(anchor, 1, value)

    def modify_ds_taubly(self, value):
        """
        """
        anchor = '#taubly'
        self.modify_line(anchor, 1, value)

    def modify_cone_angle(self, value):
        """
        """
        anchor = '#cone1'
        self.modify_line(anchor, 1, value)

        anchor = '#cone2'
        self.modify_line(anchor, 1, value)

        anchor = '#cone3'
        self.modify_line(anchor, 1, value)

    def modify_LSS_stiffness(self, value):
        """
        """
        self.modify_st_file(self.shaft_st_file, [3, 4], 12, value)

    def modify_LSS_damping(self, value):
        """
        """
        anchor = '#shaft_damp'
        self.modify_line(anchor, 6, value)

    def modify_tower_bending_stiffness(self, value):
        """
        """
        self.modify_st_file(self.tower_st_file, np.arange(3, 23), 10, value)
        self.modify_st_file(self.tower_st_file, np.arange(3, 23), 11, value)

    def modify_tower_torsional_stiffness(self, value):
        """
        """
        self.modify_st_file(self.tower_st_file, np.arange(3, 23), 12, value)

    def modify_tower_shear_stiffness(self, value):
        """
        """
        # Modify shear modulus
        self.modify_st_file(self.tower_st_file, np.arange(3, 23), 9, value)
        # Do also an opposite modification of the polar moment of inertia (to not influence the torsional stiffness)
        self.modify_st_file(self.tower_st_file, np.arange(3, 23), 12, -value / (1 + value))

    def modify_tower_mass(self, value):
        """
        """
        self.modify_st_file(self.tower_st_file, np.arange(3, 23), 1, value)

    def modify_tower_damping(self, value, mode_ID):
        anchor = '#tower_crit_damp'
        self.modify_line(anchor, mode_ID, value)

    def modify_blade_damping(self, value, mode_ID):
        anchor = '#blade_crit_damp'
        self.modify_line(anchor, mode_ID, value)

    def read_geom(self):
        header_length_geom = 2
        with open(self.blade_geom_file, 'r') as f:
            lines_geom = f.readlines()

        blade_geometry = np.loadtxt(self.blade_geom_file, skiprows=header_length_geom)

        r_geom = blade_geometry[:, 0]
        chord_init = blade_geometry[:, 1]

        return blade_geometry, r_geom, chord_init, header_length_geom, lines_geom

    def modify_blade_geom(self, preprocessed_data):
        blade_geometry, r_geom, chord_init, header_length_geom, lines_geom = self.read_geom()
        # chord modification
        if 'blade_chord_root' in preprocessed_data or 'blade_chord_tip' in preprocessed_data:
            if 'blade_chord_root' in preprocessed_data:
                chord_root = preprocessed_data['blade_chord_root']
            else:
                chord_root = np.array(0)
            if 'blade_chord_tip' in preprocessed_data:
                chord_tip = preprocessed_data['blade_chord_tip']
            else:
                chord_tip = np.array(0)
            delta_interp = interp1d([r_geom[0], r_geom[-1]], [chord_root, chord_tip])
            chord_new = [chord_i * (1 + delta_interp(radpos)) for chord_i, radpos in zip(chord_init, r_geom)]
            blade_geometry[:, 1] = chord_new

        # write file
        with open(self.blade_geom_file, 'w') as f:
            f.writelines(lines_geom[:header_length_geom])
        with open(self.blade_geom_file, 'ab') as f:
            np.savetxt(f, blade_geometry, fmt='%1.12e')

    def modify_blade_ref_axis(self, preprocessed_data):
        line_idx_bra, line = self.find_anchor(self.htc_content, '#blade_ref_axis')
        nsec = int(line.split()[1])
        blade_ref_axis = np.loadtxt(self.main_model_htc_file, skiprows=line_idx_bra + 1, max_rows=nsec,
                                    usecols=(1, 2, 3, 4, 5))
        r_ref_axis = blade_ref_axis[:, 3]

        # twist modification
        if 'blade_twist' in preprocessed_data:
            twist_interp = interp1d([r_ref_axis[0], r_ref_axis[-1]], [0, preprocessed_data['blade_twist']])
            for ir, radpos in enumerate(r_ref_axis):
                self.modify_line(None, 5, twist_interp(radpos), line_idx=line_idx_bra + ir + 1,
                                 line=self.htc_content[line_idx_bra + ir + 1])

        # nax modification
        if 'blade_na_x' in preprocessed_data:
            na_x_interp = interp1d([r_ref_axis[0], r_ref_axis[-1]], [0, preprocessed_data['blade_na_x']])
            for ir, radpos in enumerate(r_ref_axis):
                self.modify_line(None, 3, na_x_interp(radpos), line_idx=line_idx_bra + ir + 1,
                                 line=self.htc_content[line_idx_bra + ir + 1])

        # nay modification
        if 'blade_na_y' in preprocessed_data:
            na_y_interp = interp1d([r_ref_axis[0], r_ref_axis[-1]], [0, preprocessed_data['blade_na_y']])
            for ir, radpos in enumerate(r_ref_axis):
                self.modify_line(None, 2, na_y_interp(radpos), line_idx=line_idx_bra + ir + 1,
                                 line=self.htc_content[line_idx_bra + ir + 1])

    def modify_blade_structure(self, preprocessed_data):

        blade_structure = np.loadtxt(self.blade_st_file, skiprows=5, max_rows=26)
        r_struc = blade_structure[:, 0]

        # flapwise stiffness modification
        if 'blade_flap_stiffness' in preprocessed_data:
            flap_stiffness_init = blade_structure[:, 10]
            flap_stiffness_new = flap_stiffness_init * (1 + preprocessed_data['blade_flap_stiffness'])
            blade_structure[:, 10] = flap_stiffness_new

        # edgewise stiffness modification
        if 'blade_edge_stiffness' in preprocessed_data:
            edge_stiffness_init = blade_structure[:, 11]
            edge_stiffness_new = edge_stiffness_init * (1 + preprocessed_data['blade_edge_stiffness'])
            blade_structure[:, 11] = edge_stiffness_new

        # mass modification
        if 'blade_mass' in preprocessed_data:
            mass_init = blade_structure[:, 1]
            mass_new = mass_init * (1 + preprocessed_data['blade_mass'])
            blade_structure[:, 1] = mass_new

        _, r_geom, chord_init, _, _ = self.read_geom()
        chord_interp = interp1d(r_geom, chord_init, fill_value='extrapolate')

        # cog x modification
        if 'blade_cog_x' in preprocessed_data:
            cog_x_init = blade_structure[:, 2]
            delta_interp = interp1d([r_struc[0], r_struc[-1]], [0, preprocessed_data['blade_cog_x']], fill_value='extrapolate')
            cog_x_new = [cog_x_i + chord_interp(radpos) * delta_interp(radpos) for cog_x_i, radpos in zip(cog_x_init, r_struc)]
            blade_structure[:, 2] = cog_x_new

        # cog y modification
        if 'blade_cog_y' in preprocessed_data:
            cog_y_init = blade_structure[:, 3]
            delta_interp = interp1d([r_struc[0], r_struc[-1]], [0, preprocessed_data['blade_cog_y']], fill_value='extrapolate')
            cog_y_new = [cog_y_i + chord_interp(radpos) * delta_interp(radpos) for cog_y_i, radpos in zip(cog_y_init, r_struc)]
            blade_structure[:, 3] = cog_y_new

        # shear center x modification
        if 'blade_sc_x' in preprocessed_data:
            sc_x_init = blade_structure[:, 6]
            delta_interp = interp1d([r_struc[0], r_struc[-1]], [0, preprocessed_data['blade_sc_x']], fill_value='extrapolate')
            sc_x_new = [sc_x_i + chord_interp(radpos) * delta_interp(radpos) for sc_x_i, radpos in zip(sc_x_init, r_struc)]
            blade_structure[:, 6] = sc_x_new

        # shear center y modification
        if 'blade_sc_y' in preprocessed_data:
            sc_y_init = blade_structure[:, 7]
            delta_interp = interp1d([r_struc[0], r_struc[-1]], [0, preprocessed_data['blade_sc_y']], fill_value='extrapolate')
            sc_y_new = [sc_y_i + chord_interp(radpos) * delta_interp(radpos) for sc_y_i, radpos in zip(sc_y_init, r_struc)]
            blade_structure[:, 7] = sc_y_new

        # principal axis orientation
        if 'blade_pa_orientation' in preprocessed_data:
            pa_orientation_init = blade_structure[:, 16]
            pa_orientation_new = pa_orientation_init * (1 + preprocessed_data['blade_pa_orientation'])
            blade_structure[:, 16] = pa_orientation_new

        with open(self.blade_st_file, 'r') as f:
            header = f.readlines()[:5]
        with open(self.blade_st_file, 'w') as f:
            f.writelines(header)
        with open(self.blade_st_file, 'ab') as f:
            np.savetxt(f, blade_structure, fmt='%1.12e')

    def get_polars(self):
        with open(self.pc_file, 'r') as f:
            lines = f.readlines()

        aero_data = []

        end_of_file = False
        header_length = 2
        line_idx = header_length
        intermediate_polar_headers = []

        while not end_of_file:
            intermediate_polar_headers.append(lines[line_idx])

            # set_nr = int(lines[line_idx].split()[0])
            nrows = int(lines[line_idx].split()[1])
            thickness = float(lines[line_idx].split()[2])

            aero_data.append(np.loadtxt(self.pc_file, skiprows=line_idx + 1, max_rows=nrows))

            line_idx = line_idx + nrows + 1
            if line_idx >= len(lines):
                end_of_file = True

        # transform to our standard format
        polars = []
        for polar in aero_data:
            polars.append({'alpha': polar[:, 0], 'cl': polar[:, 1], 'cd': polar[:, 2], 'cm': polar[:, 3]})

        return polars, aero_data, lines[:header_length], intermediate_polar_headers

    def set_polar(self, aero_data, header, intermediate_polar_headers):

        # write aero_data to pc_file
        with open(self.pc_file, 'w') as f:
            f.writelines(header)

        for ii, data in enumerate(aero_data):
            with open(self.pc_file, 'a') as f:
                f.write(intermediate_polar_headers[ii])
            with open(self.pc_file, 'ab') as f:
                np.savetxt(f, data, fmt='%1.12e')

    def modify_polars_new(self, preprocessed_data):

        polars, aero_data, header, intermediate_polar_headers = self.get_polars()

        print('CAREFUL! Reference polar data hardcoded here')
        ref_polar_parametrization = dict()

        # IEA 15 MW
        ref_polar_parametrization['alpha_lin'] = [-11.75, -11.5, -11.15, -10, -9.5, -9.5, -9, -9, -9.3, -9.3, -9.3,
                                                  -10, -9, -9, -9, -11.2, -11.2, -11.2, -11, -9, -9, -9, -9, -9.3,
                                                  -9.3, -7.5, -7.5, -8, -9, -9.3, -9.3, -9.5, -9.5, -15, -9.3, -9.3,
                                                  -9.3, 0, 0]
        ref_polar_parametrization['alpha_tes'] = [10, 9.5, 9.3, 9, 9, 9, 9, 9, 9.3, 9.3, 9.3, 9.3, 8.8, 8.8, 9, 10,
                                                  9.3, 10, 9.5, 9.5, 9.5, 9.3, 9.3, 9.3, 9.3, 9.5, 9, 9, 9.5, 9.3, 9.8,
                                                  9.3, 11, 11, 10, 9.5, 9.5, 0, 0]
        ref_polar_parametrization['alpha_max'] = [14.22, 14.5, 14.7, 14.8, 14.9, 14.8, 15.4, 15.4, 14.9, 14.8, 14.25,
                                                  14.25, 14.25, 14.25, 14.25, 14.8, 15.4, 15.4, 15.4, 16, 16, 16, 16,
                                                  16, 16, 16, 16, 16, 16, 17.3, 16.7, 17.3, 17.3, 17.8, 17.8, 19.7,
                                                  19.7, 0, 0]
        polars_to_modify_idx = np.arange(0, 37)  # Do not modify last two cylindrical polars

        cl_new_list = self.modify_polars_basis(preprocessed_data, polars, ref_polar_parametrization, polars_to_modify_idx)

        for ii, airfoilID in enumerate(polars_to_modify_idx):
            aero_data[airfoilID][:, 1] = cl_new_list[ii]

        self.set_polar(aero_data, header, intermediate_polar_headers)

    def modify_polars_cd(self, preprocessed_data):

        polars, aero_data, header, intermediate_polar_headers = self.get_polars()

        print('CAREFUL! Reference cd polar modification hardcoded for IEA 15 MW')
        polars_to_modify_idx = np.arange(0, 37)

        cd_new_list = self.modify_polars_cd_basis(preprocessed_data, polars, polars_to_modify_idx)

        for ii, airfoilID in enumerate(polars_to_modify_idx):
            aero_data[airfoilID][:, 2] = cd_new_list[ii]

        self.set_polar(aero_data, header, intermediate_polar_headers)

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

    def run_simulation(self, mp_lock):
        """
        Execute HAWC2 simulation
        """
        # In multiprocessing the HAWC2 call might fail due to a license problem. Likely because the different processes
        # want to access the license at the same time (once the license is verified, it is no problem to run multiple
        # processes in parallel)
        # Solution: acquire lock for 10 seconds, which is (hopefully) enough to do the license check
        if mp_lock is not None:
            mp_lock.acquire()
            Timer(10, mp_lock.release).start()

        call('/opt/HAWC2S/HAWC2S.exe ./htc/IEA_15MW_RWT_Onshore.htc', cwd=self.iteration_run_directory, shell=True)

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


if __name__ == '__main__':
    iteration_run_directory = r'C:\Users\verd_he\Desktop\remove'
    config = {'master_model_directory_path': r'Z:\projects\IEA-15-240-RWT\HAWC2\IEA-15-240-RWT-Onshore-master',
              'main_model_htc_file': r'IEA_15MW_RWT_Onshore.htc',
              'data_shaft_st_file': r'IEA_15MW_RWT_Shaft_st.dat',
              'data_tower_st_file': r'IEA_15MW_RWT_Tower_st.dat',
              'data_blade_geom_file': r'IEA_15MW_RWT_ae.dat',
              'data_blade_st_file': r'IEA_15MW_RWT_Blade_st_noFPM.st',
              'pc_file': r'IEA_15MW_RWT_pc_OpenFASTpolars_3dcorr.dat'}

    # nacelle_mass, nacelle_cog_vertical, nacelle_cog_horizontal, nacelle_yaw_inertia, nacelle_nodding_inertia, nacelle_rolling_inertia
    # cone_angle, LSS_stiffness, LSS_damping, tower_bending_stiffness, tower_torsional_stiffness, tower_shear_stiffness,
    # tower_mass, separation_time_constant, pressure_lag_time_constant, vortex_lift_time_constant
    # blade_damping_1, blade_damping_2, blade_damping_3, blade_damping_4, tower_damping_1, tower_damping_2
    # polar_clalpha, polar_alpha_max, polar_cd0, blade_chord_root, blade_chord_tip, blade_twist, blade_na_x, blade_na_y
    # blade_flap_stiffness, blade_edge_stiffness, blade_mass, blade_pa_orientation, blade_cog_x, blade_cog_y, blade_sc_x, blade_sc_y
    preprocessed_data = {'nacelle_mass': 0.1,
                         'nacelle_cog_vertical': 0.1,
                         'nacelle_cog_horizontal': 0.1,
                         'nacelle_yaw_inertia': 0.1,
                         'nacelle_nodding_inertia': 0.1,
                         'nacelle_rolling_inertia': 0.1}
    preprocessed_data = {'cone_angle': 0.1,
                         'LSS_stiffness': 0.1,
                         'LSS_damping': 0.1,
                         'tower_bending_stiffness': 0.1,
                         'tower_torsional_stiffness': 0.1,
                         'tower_mass': 0.1}
    preprocessed_data = {'blade_chord_root': 0.1,
                         'blade_chord_tip': 0.1,
                         'blade_twist': 0.1,
                         'blade_na_x': 0.1,
                         'blade_na_y': 0.1,
                         'blade_flap_stiffness': 0.1,
                         'blade_edge_stiffness': 0.1,
                         'blade_pa_orientation': 0.1,
                         'blade_cog_x': 0.01,
                         'blade_cog_y': 0.01,
                         'blade_sc_x': 0.01,
                         'blade_sc_y': 0.01}
    preprocessed_data = {'polar_clalpha': 0.0,
                         'polar_alpha_max': 0.,
                         'polar_cd0': 0.1}

    preprocessed_data = {'nacelle_mass': 0.1,
                         'nacelle_cog_vertical': 0.1,
                         'nacelle_cog_horizontal': 0.1,
                         'nacelle_yaw_inertia': 0.1,
                         'nacelle_nodding_inertia': 0.1,
                         'nacelle_rolling_inertia': 0.1,
                         'cone_angle': 0.1,
                         'LSS_stiffness': 0.1,
                         'LSS_damping': 0.1,
                         'tower_bending_stiffness': 0.1,
                         'tower_torsional_stiffness': 0.1,
                         'tower_shear_stiffness': 0.1,
                         'tower_mass': 0.1,
                         'blade_damping_1': 0.1,
                         'blade_damping_2': 0.1,
                         'blade_damping_3': 0.1,
                         'blade_damping_4': 0.1,
                         'tower_damping_1': 0.1,
                         'tower_damping_2': 0.1,
                         'blade_chord_root': 0.1,
                         'blade_chord_tip': 0.1,
                         'blade_twist': 0.1,
                         'blade_na_x': 0.1,
                         'blade_na_y': 0.1,
                         'blade_flap_stiffness': 0.1,
                         'blade_edge_stiffness': 0.1,
                         'blade_pa_orientation': 0.1,
                         'blade_mass': 0.1,
                         'blade_cog_x': 0.01,
                         'blade_cog_y': 0.01,
                         'blade_sc_x': 0.01,
                         'blade_sc_y': 0.01,
                         'polar_clalpha': 0.1,
                         'polar_alpha_max': 0.1,
                         'polar_cd0': 0.1,
                         'ds_a1': 0.1,
                         'ds_a2': 0.1,
                         'ds_b1': 0.1,
                         'ds_b2': 0.1,
                         'ds_taupre': 0.1,
                         'ds_taubly': 0.1}

    test = HAWC2Model(iteration_run_directory, config)
    test.create_simulation(preprocessed_data)

