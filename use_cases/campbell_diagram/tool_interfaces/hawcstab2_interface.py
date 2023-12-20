import os

from tool_interfaces.HS2ASCIIReader import HS2Data
from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError
from tool_interfaces.hawc2_interface import HAWC2Model

import numpy as np
import sys
sys.path.append(r'/work/verd_he/tools/WiVis')
from hs2_mode_tracking import mode_tracking_check, compare_modes_with_reference, gather_modal_matrices, get_mac_matrices_first_op

class HAWCStab2Model(HAWC2Model):
    """
    HAWCStab2 model interface

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
             
    def extract_results(self):
        """
        Extract the useful, desired signals which can be used for damping determination.

        Returns
        -------
        result_dict : dict
            Collected simulation results. Key names have to match with var_names specified in the postprocessor
            settings in the config
        """
        cmb = os.path.join(self.iteration_run_directory, 'htc', self.config['cmb_filename'])
        amp = os.path.join(self.iteration_run_directory, 'htc', self.config['amp_filename'])
        opt = os.path.join(self.iteration_run_directory, 'htc', self.config['opt_filename'])

        HS2_results = HS2Data(cmb, amp, opt)
        HS2_results.read_data()

        # extracting damping of 'critical' mode
        # for a HS2 run with multiple operating points, the HS2_results.modes.damping matrix has the shape:
        # # of modes x # of wind speeds x 2

        result_dict = dict()
        result_dict['damping'] = HS2_results.modes.damping[:, :, 1]
        result_dict['frequency'] = HS2_results.modes.frequency[:, :, 1]
        result_dict['mode_names'] = HS2_results.modes.names

        return result_dict

    def verify_accurate_hs2_modetracking(self, mode_indices=None):

        path_to_cmb = os.path.join(self.iteration_run_directory, 'htc', self.config['cmb_filename'])
        path_to_bin = os.path.join(self.iteration_run_directory, 'htc', self.config['cmb_filename'][:-3] + 'bin')
        n_op_points = self.config['n_op_points']
        nmodes = self.config['nmodes']
        ndofs = self.config['ndofs']

        path_to_reference_bin = self.config['path_to_reference_bin']
        path_to_reference_cmb = self.config['path_to_reference_cmb']
        mode_indices_ref = [int(mode_idx) for mode_idx in self.config['mode_indices_ref']]

        full_modal_matrix_ref, freq_ref, damp_ref = gather_modal_matrices(path_to_reference_bin, path_to_reference_cmb,
                                                                          n_op_points, nmodes, ndofs)

        full_modal_matrix_analysis, freq_analysis, damp_analysis = gather_modal_matrices(path_to_bin, path_to_cmb,
                                                                                         n_op_points, nmodes, ndofs)

        mac_difference_to_ref, mac_hs2_difference_to_ref = compare_modes_with_reference(full_modal_matrix_ref, freq_ref, damp_ref,
                                                                                        full_modal_matrix_analysis, freq_analysis,
                                                                                        damp_analysis, mode_indices_ref, mode_indices)

        mac_difference, mac_hs2_difference = mode_tracking_check(path_to_bin, path_to_cmb, n_op_points, nmodes, ndofs)

        if mode_indices is not None:
            mac_difference_desired_modes = mac_hs2_difference[mode_indices, :]
        else:
            mac_difference_desired_modes = mac_hs2_difference

        return True, mac_difference_desired_modes, mac_difference_to_ref

    def pick_modes_based_on_reference(self, mode_indices=None):

        path_to_cmb = os.path.join(self.iteration_run_directory, 'htc', self.config['cmb_filename'])
        path_to_bin = os.path.join(self.iteration_run_directory, 'htc', self.config['cmb_filename'][:-3] + 'bin')
        n_op_points = self.config['n_op_points']
        nmodes = self.config['nmodes']
        ndofs = self.config['ndofs']

        path_to_reference_bin = self.config['path_to_reference_bin']
        path_to_reference_cmb = self.config['path_to_reference_cmb']

        full_modal_matrix_ref, freq_ref, damp_ref = gather_modal_matrices(path_to_reference_bin, path_to_reference_cmb,
                                                                          n_op_points, nmodes, ndofs)

        full_modal_matrix_analysis, freq_analysis, damp_analysis = gather_modal_matrices(path_to_bin, path_to_cmb,
                                                                                         n_op_points, nmodes, ndofs)

        full_mac_matrix, full_mac_hs2_matrix = get_mac_matrices_first_op(full_modal_matrix_ref, freq_ref, damp_ref,
                                                                         full_modal_matrix_analysis, freq_analysis,
                                                                         damp_analysis)

        picked_mode_indices = []
        picked_mode_indices_hs2 = []
        mode_indices_ref = [int(mode_idx) for mode_idx in self.config['mode_indices_ref']]
        picked_mode_indices = full_mac_matrix[mode_indices_ref].argmax(axis=1)
        picked_mode_indices_hs2 = full_mac_hs2_matrix[mode_indices_ref].argmax(axis=1)

        # checking result for each mode individually
        success = [True] * len(mode_indices_ref)
        for ii in range(len(mode_indices_ref)):

            # check 1: standard MAC and MAC XP (or MAC_hs2) give same result
            if not picked_mode_indices[ii] == picked_mode_indices_hs2[ii]:
                print('Picked modes are different based on MAC implementation')
                success[ii] = False

            # check 2: MAC value should be higher than 0.5
            if full_mac_matrix[mode_indices_ref[ii], picked_mode_indices[ii]] < self.config['minimum_MAC_mode_picking']:
                print('Minimum MAC value for the picked mode is too small (< minimum_MAC_mode_picking)')
                success[ii] = False

        return success, picked_mode_indices


if __name__ == '__main__':
    iteration_run_directory = r'/work/verd_he/projects/torque2024_hs2/wtuq/use_cases/campbell_diagram/results/IEA_15MW_pce_blade-only_NEW/f7fd7d0c-8f92-11ee-9d92-b88584bcc377'
    config = {'cmb_filename': 'IEA_15MW_RWT_Onshore_Blade.cmb', 'amp_filename': 'IEA_15MW_RWT_Onshore_Blade.amp', 'opt_filename': 'IEA_15MW_RWT_Onshore_12Points.opt'}
    test = HAWCStab2Model(iteration_run_directory, config)
    result_dict = test.extract_results()
