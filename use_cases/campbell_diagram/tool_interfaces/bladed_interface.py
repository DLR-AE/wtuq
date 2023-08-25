"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 25.11.2020
"""
import copy

import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.misc import derivative
import matplotlib.pyplot as plt

import sys
sys.path.append('pyBladed')
sys.path.append(r'C:\DNV GL\Bladed 4.11')

from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError

from pyBladed.model import BladedModel as BladedAPIModel
from pyBladed.results import BladedResult


class BladedModel(SimulationModel):
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
    bl_model : BladedAPIModel
        Interactive Bladed model
    bl_st : dict
        Structural blade data in Bladed format
    run_directory : string
        Path to directory of this framework iteration
    geometry_definition : string
        Path to geometry input file
    result_prefix : string
        Prefix of output files
    """
    def __init__(self, run_directory, config):
        self.bl_model = BladedAPIModel(os.path.abspath(config['template_project']), run_directory)
        self.bl_model.suppress_logs()
        self.bl_st = dict()
        self.run_directory = run_directory

    def create_simulation(self, preprocessed_data):
        """
        Setup modified bladed model

        Parameters
        ----------
        preprocessed_data : dict
            Dictionary with the samples of the uncertain parameters which have to be set in the model
        """
        # h2_geom = self.parse_h2_becas_geom(self.geometry_definition)
        # self.transfer_h2_to_bladed(blade_data['h2_beam'], h2_geom)

        # self.bl_st = do_something(preprocessed_data)
        # self.bl_aero = do_something_else(preprocessed_data)
        # self.modify_prj()

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

            elif param == 'separation_time_constant':
                self.modify_separation_time_constant(value)

            elif param == 'pressure_lag_time_constant':
                self.modify_pressure_lag_time_constant(value)

            elif param == 'vortex_lift_time_constant':
                self.modify_vortex_lift_time_constant(value)

            elif 'blade_damping' in param:
                self.modify_blade_damping(value, int(param.split('_')[-1]))

            elif 'tower_damping' in param:
                self.modify_tower_damping(value, int(param.split('_')[-1]))

            else:
                continue

    def run_simulation(self, mp_lock=None):
        """
        Execute Bladed simulation
        """
        self.result_prefix = 'stab_analysis_run'
        self.bl_model.run_simulation(result_directory=self.run_directory, prefix=self.result_prefix, mp_lock=mp_lock)

    def modify_nacelle_mass(self, value):
        """
        """
        orig_value = self.bl_model.get_nacelle_mass()
        self.bl_model.set_nacelle_mass(orig_value * (1 + value))

    def modify_nacelle_cog_vertical(self, value):
        """
        """
        orig_value = self.bl_model.get_nacelle_cog_vertical()
        self.bl_model.set_nacelle_cog_vertical(orig_value * (1 + value))

    def modify_nacelle_cog_horizontal(self, value):
        """
        """
        orig_value = self.bl_model.get_nacelle_cog_horizontal()
        self.bl_model.set_nacelle_cog_horizontal(orig_value * (1 + value))

    def modify_nacelle_yaw_inertia(self, value):
        """
        """
        orig_value = self.bl_model.get_nacelle_yaw_inertia()
        self.bl_model.set_nacelle_yaw_inertia(orig_value * (1 + value))

    def modify_nacelle_nodding_inertia(self, value):
        """
        """
        orig_value = self.bl_model.get_nacelle_nodding_inertia()
        self.bl_model.set_nacelle_nodding_inertia(orig_value * (1 + value))

    def modify_nacelle_rolling_inertia(self, value):
        """
        """
        orig_value = self.bl_model.get_nacelle_rolling_inertia()
        self.bl_model.set_nacelle_rolling_inertia(orig_value * (1 + value))

    def modify_cone_angle(self, value):
        """
        """
        orig_value = self.bl_model.get_cone_angle()
        self.bl_model.set_cone_angle(orig_value * (1 + value))

    def modify_LSS_stiffness(self, value):
        """
        """
        orig_value = self.bl_model.get_LSS_stiffness()
        self.bl_model.set_LSS_stiffness(orig_value * (1 + value))

    def modify_LSS_damping(self, value):
        """
        """
        orig_value = self.bl_model.get_LSS_damping()
        self.bl_model.set_LSS_damping(orig_value * (1 + value))

    def modify_tower_bending_stiffness(self, value):
        """
        """
        orig_values = self.bl_model.get_tower_bending_stiffness()
        self.bl_model.set_tower_bending_stiffness(orig_values * (1 + value))

    def modify_tower_torsional_stiffness(self, value):
        """
        """
        orig_values = self.bl_model.get_tower_torsional_stiffness()
        self.bl_model.set_tower_torsional_stiffness(orig_values * (1 + value))

    def modify_tower_shear_stiffness(self, value):
        """
        """
        orig_values = self.bl_model.get_tower_shear_stiffness()
        self.bl_model.set_tower_shear_stiffness(orig_values * (1 + value))

    def modify_tower_mass(self, value):
        """
        """
        orig_values = self.bl_model.get_tower_mass()
        self.bl_model.set_tower_mass(orig_values * (1 + value))

    def modify_separation_time_constant(self, value):
        """
        """
        orig_value = self.bl_model.get_DS_separation_time_constant()
        self.bl_model.set_DS_separation_time_constant(orig_value * (1 + value))

    def modify_pressure_lag_time_constant(self, value):
        """
        """
        orig_value = self.bl_model.get_DS_pressure_lag_time_constant()
        self.bl_model.set_DS_pressure_lag_time_constant(orig_value * (1 + value))

    def modify_vortex_lift_time_constant(self, value):
        """
        """
        orig_value = self.bl_model.get_DS_vortex_lift_time_constant()
        self.bl_model.set_DS_vortex_lift_time_constant(orig_value * (1 + value))

    def modify_blade_damping(self, value, mode_ID):
        orig_value = self.bl_model.get_blade_damping(mode_ID-1)
        self.bl_model.set_blade_damping(orig_value * (1 + value), mode_ID-1)

    def modify_tower_damping(self, value, mode_ID):
        orig_value = self.bl_model.get_tower_damping(mode_ID-1)
        self.bl_model.set_tower_damping(orig_value * (1 + value), mode_ID-1)

    def modify_blade_structure(self, preprocessed_data):

        blade = self.bl_model.get_blade()
        blade_modified = copy.deepcopy(blade)

        r = blade['DistanceAlongPitchAxis']

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
            chord_init = blade['Chord']
            delta_interp = interp1d([r[0], r[-1]], [chord_root, chord_tip])
            chord_new = [chord_i * (1 + delta_interp(radpos)) for chord_i, radpos in zip(chord_init, r)]
            blade_modified['Chord'] = chord_new

        # twist modification
        if 'blade_twist' in preprocessed_data:
            twist_init = blade['TwistRadians']
            delta_interp = interp1d([r[0], r[-1]], [0, preprocessed_data['blade_twist']])
            twist_new = [twist_i * (1 + delta_interp(radpos)) for twist_i, radpos in zip(twist_init, r)]
            blade_modified['TwistRadians'] = twist_new

        # nax modification
        if 'blade_na_x' in preprocessed_data:
            na_x_init = blade['NeutralAxis.X']
            delta_interp = interp1d([r[0], r[-1]], [0, preprocessed_data['blade_na_x']])
            na_x_new = [na_x_i * (1 + delta_interp(radpos)) for na_x_i, radpos in zip(na_x_init, r)]
            blade_modified['NeutralAxis.X'] = na_x_new

        # nay modification
        if 'blade_na_y' in preprocessed_data:
            na_y_init = blade['NeutralAxis.Y']
            delta_interp = interp1d([r[0], r[-1]], [0, preprocessed_data['blade_na_y']])
            na_y_new = [na_y_i * (1 + delta_interp(radpos)) for na_y_i, radpos in zip(na_y_init, r)]
            blade_modified['NeutralAxis.Y'] = na_y_new

        # flapwise stiffness modification
        if 'blade_flap_stiffness' in preprocessed_data:
            flap_stiffness_init = blade['bendingStiffnessYp']
            flap_stiffness_new = flap_stiffness_init * (1 + preprocessed_data['blade_flap_stiffness'])
            blade_modified['bendingStiffnessYp'] = flap_stiffness_new

        # edgewise stiffness modification
        if 'blade_edge_stiffness' in preprocessed_data:
            edge_stiffness_init = blade['bendingStiffnessXp']
            edge_stiffness_new = edge_stiffness_init * (1 + preprocessed_data['blade_edge_stiffness'])
            blade_modified['bendingStiffnessXp'] = edge_stiffness_new

        # mass modification
        if 'blade_mass' in preprocessed_data:
            mass_init = blade['massPerUnitLength']
            mass_new = mass_init * (1 + preprocessed_data['blade_mass'])
            blade_modified['massPerUnitLength'] = mass_new

        # radial cog
        # preprocessed_data['blade_radial_cog']
        # ?

        # cog x modification
        if 'blade_cog_x' in preprocessed_data:
            cog_x_init = blade['CentreOfMass.X']
            delta_interp = interp1d([r[0], r[-1]], [0, preprocessed_data['blade_cog_x']])
            cog_x_new = [cog_x_i + delta_interp(radpos) * 100 for cog_x_i, radpos in zip(cog_x_init, r)]
            blade_modified['CentreOfMass.X'] = cog_x_new

        # vvv not required, cog and shear center positions are already normalized by chord length vvv
        # delta_cog_x = cog_x * blade['Chord']
        # cog_x_init_absolute = cog_x_init/100 * blade['Chord']
        # cog_x_new_absolute = cog_x_init_absolute + delta_cog_x
        # cog_x_new = (cog_x_new_absolute / blade['Chord']) * 100

        # cog y modification
        if 'blade_cog_y' in preprocessed_data:
            cog_y_init = blade['CentreOfMass.Y']
            delta_interp = interp1d([r[0], r[-1]], [0, preprocessed_data['blade_cog_y']])
            cog_y_new = [cog_y_i + delta_interp(radpos) * 100 for cog_y_i, radpos in zip(cog_y_init, r)]
            blade_modified['CentreOfMass.Y'] = cog_y_new

        # shear center x modification
        if 'blade_sc_x' in preprocessed_data:
            sc_x_init = blade['ShearCentre.X']
            delta_interp = interp1d([r[0], r[-1]], [0, preprocessed_data['blade_sc_x']])
            sc_x_new = [sc_x_i + delta_interp(radpos) * 100 for sc_x_i, radpos in zip(sc_x_init, r)]
            blade_modified['ShearCentre.X'] = sc_x_new

        # shear center y modification
        if 'blade_sc_y' in preprocessed_data:
            sc_y_init = blade['ShearCentre.Y']
            delta_interp = interp1d([r[0], r[-1]], [0, preprocessed_data['blade_sc_y']])
            sc_y_new = [sc_y_i + delta_interp(radpos) * 100 for sc_y_i, radpos in zip(sc_y_init, r)]
            blade_modified['ShearCentre.Y'] = sc_y_new

        # principal axis orientation
        if 'blade_pa_orientation' in preprocessed_data:
            pa_orientation_init = blade['PrincipalAxisOrientationRadians']
            pa_orientation_new = pa_orientation_init * (1 + preprocessed_data['blade_pa_orientation'])
            blade_modified['PrincipalAxisOrientationRadians'] = pa_orientation_new

        # self.plot_blade_modifications(blade, blade_modified)
        self.bl_model.modify_blade(blade_modified)

    def plot_blade_modifications(self, blade, blade_modified):

        r = blade['DistanceAlongPitchAxis']

        for param in ['Chord', 'TwistRadians', 'NeutralAxis.X', 'NeutralAxis.Y', 'bendingStiffnessYp',
                      'bendingStiffnessXp', 'massPerUnitLength', 'CentreOfMass.X', 'CentreOfMass.Y',
                      'ShearCentre.X', 'ShearCentre.Y', 'PrincipalAxisOrientationRadians']:
            plt.plot(r, blade[param], label='Original')
            plt.plot(r, blade_modified[param], label='Modified')

            plt.legend()
            plt.grid()
            plt.xlabel('Radius (distance along pitch axis) / m')
            plt.ylabel(param)
            plt.savefig(os.path.join(self.run_directory, param + '.png'), dpi=300)
            plt.show()

    def modify_polars(self, preprocessed_data):
        delta1 = preprocessed_data['polar_alpha_tes']
        delta2 = preprocessed_data['polar_alpha_max']
        delta3 = preprocessed_data['polar_alpha_sr']
        delta4 = preprocessed_data['polar_cl']

        ref_polar_parametrization = dict()
        # demo-a
        # ref_polar_parametrization['alpha_lin'] = [-8, -10, -11.3]
        # ref_polar_parametrization['alpha_tes'] = [5, 8, 10]
        # ref_polar_parametrization['alpha_max'] = [10, 13, 14]
        # ref_polar_parametrization['alpha_sr'] = [21.61, 21.61, 21.61]

        # IEA 15 MW
        ref_polar_parametrization['alpha_lin'] = [-8.5, -8, -10, -10, -10, -10, -10]
        ref_polar_parametrization['alpha_tes'] = [9.12, 8, 9, 10, 10, 10, 9]
        ref_polar_parametrization['alpha_max'] = [15.94, 14, 16, 15, 14, 16, 15]
        ref_polar_parametrization['alpha_sr'] = [24.26, 28, 32, 32, 28, 36.14, 36.14]

        polars = self.bl_model.get_polars()

        for airfoilID in range(len(ref_polar_parametrization['alpha_lin'])):
            # 1)
            alpha_tes_new = ref_polar_parametrization['alpha_tes'][airfoilID] * (1 + delta1)
            alpha_max_new = ref_polar_parametrization['alpha_max'][airfoilID] * (1 + delta1 + delta2)
            alpha_sr_new = ref_polar_parametrization['alpha_sr'][airfoilID] * (1 + delta1 + delta3)

            # 2)
            f_orig = interp1d(polars[airfoilID+2]['alpha'], polars[airfoilID+2]['cl'])

            cl_tes_orig = f_orig(ref_polar_parametrization['alpha_tes'][airfoilID])
            cl_max_orig = f_orig(ref_polar_parametrization['alpha_max'][airfoilID])
            cl_sr_orig = f_orig(ref_polar_parametrization['alpha_sr'][airfoilID])

            # 3)
            eps_tes = f_orig(alpha_tes_new) - cl_tes_orig
            eps_max = f_orig(alpha_max_new) - cl_max_orig
            eps_sr = f_orig(alpha_sr_new) - cl_sr_orig

            # 4)
            cl_diff_tes = delta4 * cl_tes_orig - eps_tes
            cl_diff_max = delta4 * cl_max_orig - eps_max
            cl_diff_sr = delta4 * cl_sr_orig - eps_sr

            alpha_coordinates = [ref_polar_parametrization['alpha_lin'][airfoilID],
                                 alpha_tes_new,
                                 alpha_max_new,
                                 alpha_sr_new,
                                 90]
            cl_diffs = [0,
                        cl_diff_tes,
                        cl_diff_max,
                        cl_diff_sr,
                        0]

            f_diff = interp1d(alpha_coordinates, cl_diffs, fill_value=0, bounds_error=False)

            cl_diff_interpolated = f_diff(polars[airfoilID+2]['alpha'])
            cl_new = polars[airfoilID+2]['cl'] + cl_diff_interpolated
            self.bl_model.set_polar_cl(sectionID=airfoilID+2, cl_new=cl_new)

            self.plot_polar_difference(airfoilID=airfoilID,
                                       alpha=polars[airfoilID+2]['alpha'],
                                       cl_orig=polars[airfoilID+2]['cl'],
                                       cl_new=cl_new,
                                       alpha_control_points_orig=[ref_polar_parametrization['alpha_lin'][airfoilID],
                                                                  ref_polar_parametrization['alpha_tes'][airfoilID],
                                                                  ref_polar_parametrization['alpha_max'][airfoilID],
                                                                  ref_polar_parametrization['alpha_sr'][airfoilID],
                                                                  90],
                                       cl_control_points_orig=f_orig([ref_polar_parametrization['alpha_lin'][airfoilID],
                                                                      ref_polar_parametrization['alpha_tes'][airfoilID],
                                                                      ref_polar_parametrization['alpha_max'][airfoilID],
                                                                      ref_polar_parametrization['alpha_sr'][airfoilID],
                                                                      90]),
                                       alpha_control_points_new=alpha_coordinates,
                                       cl_control_points_new=f_diff(alpha_coordinates) + f_orig(alpha_coordinates))

    def modify_polars_new(self, preprocessed_data):
        if 'polar_clalpha' in preprocessed_data:
            factor_clalpha = preprocessed_data['polar_clalpha']
        else:
            factor_clalpha = np.array(0)

        if 'polar_alpha_max' in preprocessed_data:
            factor_alpha_max = preprocessed_data['polar_alpha_max']
        else:
            factor_alpha_max = np.array(0)

        print('CAREFUL! Reference polar data hardcoded here')
        ref_polar_parametrization = dict()
        # demo-a
        # ref_polar_parametrization['alpha_lin'] = [-8, -10, -11.3]
        # ref_polar_parametrization['alpha_tes'] = [5, 8, 7]
        # ref_polar_parametrization['alpha_max'] = [10, 13, 14]

        # IEA 15 MW
        ref_polar_parametrization['alpha_lin'] = [-8.5, -8, -10, -10, -10, -10, -10]
        ref_polar_parametrization['alpha_tes'] = [9.12, 8, 9, 10, 10, 10, 9]
        ref_polar_parametrization['alpha_max'] = [15.94, 14, 16, 15, 14, 16, 15]

        polars = self.bl_model.get_polars()

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # AirfoilID = id of the reference polar data
        # the airfoils in bladed have 1 extra (dummy polar) at index 0 and for both the demo-a and IEA model the
        # first cylindrical airfoil is skipped. So the Bladed polar index is +2
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for airfoilID in range(len(ref_polar_parametrization['alpha_lin'])):
            # 1) Determine tilting point cl alpha
            alpha_lin = ref_polar_parametrization['alpha_lin'][airfoilID]
            alpha_tes = ref_polar_parametrization['alpha_tes'][airfoilID]

            alpha_tilt = (alpha_lin + alpha_tes) / 2

            f = interp1d(polars[airfoilID + 2]['alpha'][1:], polars[airfoilID + 2]['cl'][1:], fill_value='extrapolate')

            dcl_dalpha_orig = derivative(f, alpha_tilt, 0.5)
            dcl_dalpha = dcl_dalpha_orig * (1 + factor_clalpha)

            alpha_max = ref_polar_parametrization['alpha_max'][airfoilID]
            delta_alpha_max = alpha_max * factor_alpha_max
            alpha_max_new = alpha_max + delta_alpha_max
            alpha_tes_new = alpha_tes + delta_alpha_max

            f_delta_cl_pre_tes = interp1d([alpha_lin, alpha_tes_new],
                                          [f(alpha_tilt) + dcl_dalpha * (alpha_lin - alpha_tilt) - f(alpha_lin),
                                           f(alpha_tilt) + dcl_dalpha * (alpha_tes_new - alpha_tilt) - f(alpha_tes_new)],
                                          bounds_error=False,
                                          fill_value=(f(alpha_tilt) + dcl_dalpha * (alpha_lin - alpha_tilt) - f(alpha_lin),
                                                      f(alpha_tilt) + dcl_dalpha * (alpha_tes_new - alpha_tilt) - f(alpha_tes_new)))
            f_delta_cl_pre_tes_orig = interp1d([alpha_lin, alpha_tes_new],
                                               [f(alpha_tilt) + dcl_dalpha_orig * (alpha_lin - alpha_tilt) - f(alpha_lin),
                                                f(alpha_tilt) + dcl_dalpha_orig * (alpha_tes_new - alpha_tilt) - f(
                                                    alpha_tes_new)],
                                               bounds_error=False,
                                               fill_value=(
                                               f(alpha_tilt) + dcl_dalpha_orig * (alpha_lin - alpha_tilt) - f(alpha_lin),
                                               f(alpha_tilt) + dcl_dalpha_orig * (alpha_tes_new - alpha_tilt) - f(alpha_tes_new)))

            cl_new_pre_tes = f(polars[airfoilID + 2]['alpha']) + (f_delta_cl_pre_tes(polars[airfoilID + 2]['alpha']) -
                                                                  f_delta_cl_pre_tes_orig(polars[airfoilID + 2]['alpha']))

            cl_tes_orig = f(alpha_tes)
            cl_tes_new = f(alpha_tes_new) + (f_delta_cl_pre_tes(alpha_tes_new) - f_delta_cl_pre_tes_orig(alpha_tes_new))  # f(alpha_tilt) + dcl_dalpha * (alpha_tes_new - alpha_tilt)
            delta_cl_post_tes = cl_tes_new - cl_tes_orig

            f2 = interp1d(polars[airfoilID + 2]['alpha'] + delta_alpha_max, polars[airfoilID + 2]['cl'] + delta_cl_post_tes, fill_value='extrapolate')

            cl_new_post_tes = f2(polars[airfoilID + 2]['alpha'])

            cl_new = np.hstack((cl_new_pre_tes[np.where(polars[airfoilID + 2]['alpha'] <= alpha_tes_new)],
                                cl_new_post_tes[np.where(polars[airfoilID + 2]['alpha'] > alpha_tes_new)]))

            f_final = interp1d(polars[airfoilID + 2]['alpha'], cl_new)

            self.bl_model.set_polar_cl(sectionID=airfoilID+2, cl_new=cl_new)

            if False:
                self.plot_polar_difference(airfoilID=airfoilID,
                                           alpha=polars[airfoilID+2]['alpha'],
                                           cl_orig=polars[airfoilID+2]['cl'],
                                           cl_new=cl_new,
                                           alpha_control_points_orig=[alpha_lin,
                                                                      alpha_tes,
                                                                      alpha_max],
                                           cl_control_points_orig=f([alpha_lin,
                                                                     alpha_tes,
                                                                     alpha_max]),
                                           alpha_control_points_new=[ref_polar_parametrization['alpha_lin'][airfoilID],
                                                                     alpha_tes_new,
                                                                     alpha_max_new],
                                           cl_control_points_new=f_final([alpha_lin,
                                                                          alpha_tes_new,
                                                                          alpha_max_new]))

    def plot_polar_difference(self, airfoilID, alpha, cl_orig, cl_new,
                              alpha_control_points_orig, cl_control_points_orig,
                              alpha_control_points_new, cl_control_points_new):

        plt.plot(alpha, cl_orig, label='Cl orig')
        plt.plot(alpha, cl_new, label='Cl new')
        plt.plot(alpha_control_points_orig, cl_control_points_orig, linestyle='', marker='*', label='Control points orig')
        plt.plot(alpha_control_points_new, cl_control_points_new, linestyle='', marker='*', label='Control points new')
        plt.xlim([alpha_control_points_orig[0]-1, 90])
        plt.legend()
        plt.grid()
        plt.xlabel('alpha / deg.')
        plt.ylabel('Cl')
        plt.savefig(os.path.join(self.run_directory, str(airfoilID) + '.png'), dpi=300)
        plt.close()
        # plt.show()

    def modify_polars_cd(self, preprocessed_data):
        factor_cd0 = preprocessed_data['polar_cd0']
        if factor_cd0 < -1:
            print('CD values smaller than 0 not allowed -> hard limit at CD = 0')
            factor_cd0 = -1

        polars = self.bl_model.get_polars()

        for airfoilID in range(3):
            fcd = interp1d(polars[airfoilID + 2]['alpha'], polars[airfoilID + 2]['cd'])
            fcd_delta = interp1d([-180, 0, 180], [0, fcd(0) * factor_cd0, 0], fill_value='extrapolate')
            cd_new = fcd(polars[airfoilID + 2]['alpha']) + fcd_delta(polars[airfoilID + 2]['alpha'])
            self.bl_model.set_polar_cd(sectionID=airfoilID + 2, cd_new=cd_new)

            if False:
                self.plot_polar_difference_cd(polars[airfoilID + 2]['alpha'], polars[airfoilID + 2]['cd'], cd_new)

    def plot_polar_difference_cd(self, alpha, cd_orig, cd_new):

        plt.plot(alpha, cd_orig, label='Cd orig')
        plt.plot(alpha, cd_new, label='Cd new')
        plt.legend()
        plt.grid()
        plt.xlabel('alpha / deg.')
        plt.ylabel('Cd')
        plt.show()
