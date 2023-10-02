"""
Extra module to avoid cyclic imports
"""
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.misc import derivative
import matplotlib.pyplot as plt


class SimulationError(Exception):
    """
    Custom exception to indicate that a simulation failed.
    """
    pass


class SimulationModel:
    """
    Abstract base class for simulation tool interfaces (for clarity / documentation)
    """

    # abstract
    def __init__(self):
        """
        Initializes the object only, no work is done here

        @return: nothing
        """
        raise NotImplementedError('do not use this abstract method!')

    # abstract
    def create_simulation(self, blade_data):
        """
        Prepares a new model for a simulation run with the desired blade properties.

        @param blade_data: dict with blade data in different formats

        The necessary information from blade_data (which is redundant) is extracted here for each tool

        Example: Create a new project by
          - cloning the full model and modifying the blade model
          - creating a modified project file (Blade)
          - create specific sub-model and link the unchanged part (possible in Simpack)
        """
        raise NotImplementedError('do not use this abstract method!')

    # abstract
    def run_simulation(self):
        """
        Runs a simulation on the created model.

        Needs to raise an exception if simulation fails (either the simulator returns an error or result is not as
        expected.
        """
        raise NotImplementedError('do not use this abstract method!')

    # abstract
    def extract_results(self):
        """
        Reads and converts tool-specific output to python dictionary.

        keys: the nomenclature has to correspond to the var_name parameter in the [postprocessor] section of the config
        values: can be 1D or 2D arrays

        minimum required variables:
        - in case only 1D vars are given:
            time (1D array)
        - in case 2D vars are given (# of timesteps x # of radpos)
            time (1D array)
            radpos (1D array)

        example:
        {'time': <1D array>, 'deflectionOutOfPlane57': <1D array>, 'deflectionInPlane57': <1D array>}
        {'time': <1D array>, 'radpos': <1D array>, 'torsion_b1': <2D array>, 'deflectionOutOfPlane_b1': <2D array>}
        """
        raise NotImplementedError('do not use this abstract method!')

    def modify_polars_basis(self, preprocessed_data, polars, ref_polar_parametrization, polars_to_modify_idx):

        if 'polar_clalpha' in preprocessed_data:
            factor_clalpha = preprocessed_data['polar_clalpha']
        else:
            factor_clalpha = np.array(0)

        if 'polar_alpha_max' in preprocessed_data:
            factor_alpha_max = preprocessed_data['polar_alpha_max']
        else:
            factor_alpha_max = np.array(0)

        cl_new_list = []
        # AirfoilID = id of the reference polar data
        for airfoilID in polars_to_modify_idx:

            # 1) Determine tilting point cl alpha
            alpha_lin = ref_polar_parametrization['alpha_lin'][airfoilID]
            alpha_tes = ref_polar_parametrization['alpha_tes'][airfoilID]

            alpha_tilt = (alpha_lin + alpha_tes) / 2

            f = interp1d(polars[airfoilID]['alpha'][1:], polars[airfoilID]['cl'][1:], fill_value='extrapolate')

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

            cl_new_pre_tes = f(polars[airfoilID]['alpha']) + (f_delta_cl_pre_tes(polars[airfoilID]['alpha']) -
                                                                  f_delta_cl_pre_tes_orig(polars[airfoilID]['alpha']))

            cl_tes_orig = f(alpha_tes)
            cl_tes_new = f(alpha_tes_new) + (f_delta_cl_pre_tes(alpha_tes_new) - f_delta_cl_pre_tes_orig(alpha_tes_new))  # f(alpha_tilt) + dcl_dalpha * (alpha_tes_new - alpha_tilt)
            delta_cl_post_tes = cl_tes_new - cl_tes_orig

            f2 = interp1d(polars[airfoilID]['alpha'] + delta_alpha_max, polars[airfoilID]['cl'] + delta_cl_post_tes, fill_value='extrapolate')

            cl_new_post_tes = f2(polars[airfoilID]['alpha'])

            cl_new = np.hstack((cl_new_pre_tes[np.where(polars[airfoilID]['alpha'] <= alpha_tes_new)],
                                cl_new_post_tes[np.where(polars[airfoilID]['alpha'] > alpha_tes_new)]))
            cl_new_list.append(cl_new)

            f_final = interp1d(polars[airfoilID]['alpha'], cl_new)

            if False:
                self.plot_polar_difference(airfoilID=airfoilID,
                                           alpha=polars[airfoilID]['alpha'],
                                           cl_orig=polars[airfoilID]['cl'],
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
        return cl_new_list

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

    def modify_polars_cd_basis(self, preprocessed_data, polars, polars_to_modify_idx):

        factor_cd0 = preprocessed_data['polar_cd0']
        if factor_cd0 < -1:
            print('CD values smaller than 0 not allowed -> hard limit at CD = 0')
            factor_cd0 = -1

        cd_new_list = []
        for airfoilID in polars_to_modify_idx:
            fcd = interp1d(polars[airfoilID]['alpha'], polars[airfoilID]['cd'])
            fcd_delta = interp1d([-180, 0, 180], [0, fcd(0) * factor_cd0, 0], fill_value='extrapolate')
            cd_new = fcd(polars[airfoilID + 2]['alpha']) + fcd_delta(polars[airfoilID + 2]['alpha'])

            cd_new_list.append(cd_new)

            if False:
                self.plot_polar_difference_cd(polars[airfoilID]['alpha'], polars[airfoilID]['cd'], cd_new)

        return cd_new_list

    def plot_polar_difference_cd(self, alpha, cd_orig, cd_new):

        plt.plot(alpha, cd_orig, label='Cd orig')
        plt.plot(alpha, cd_new, label='Cd new')
        plt.legend()
        plt.grid()
        plt.xlabel('alpha / deg.')
        plt.ylabel('Cd')
        plt.show()