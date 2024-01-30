import os

from tool_interfaces.HS2ASCIIReader import HS2Data
from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError
from tool_interfaces.hawc2_interface import HAWC2Model

import numpy as np
import sys

from wivis.hs2_mode_tracking import mode_tracking_check, compare_modes_with_reference, gather_modal_matrices, get_mac_matrices_first_op, compare_modes_with_reference_last_op

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

        full_modal_matrix_ref, freq_ref, damp_ref, realpart_ref, imagpart_ref = gather_modal_matrices(path_to_reference_bin, path_to_reference_cmb,
                                                                                                      n_op_points, nmodes, ndofs)

        full_modal_matrix_analysis, freq_analysis, damp_analysis, realpart_analysis, imagpart_analysis = gather_modal_matrices(path_to_bin, path_to_cmb,
                                                                                                                               n_op_points, nmodes, ndofs)

        mac_difference_to_ref, mac_hs2_difference_to_ref = compare_modes_with_reference(full_modal_matrix_ref, freq_ref, damp_ref, realpart_ref, imagpart_ref,
                                                                                        full_modal_matrix_analysis, freq_analysis, realpart_analysis, imagpart_analysis,
                                                                                        damp_analysis, mode_indices_ref, mode_indices, with_damp=self.config['with_damp'])

        mac_difference_to_ref_last_op, mac_hs2_difference_to_ref_last_op = compare_modes_with_reference_last_op(full_modal_matrix_ref, freq_ref, damp_ref, realpart_ref, imagpart_ref,
                                                                                        full_modal_matrix_analysis, freq_analysis, realpart_analysis, imagpart_analysis,
                                                                                        damp_analysis, mode_indices_ref, mode_indices, with_damp=self.config['with_damp'])

        mac_difference, mac_hs2_difference = mode_tracking_check(path_to_bin, path_to_cmb, n_op_points, nmodes, ndofs, with_damp=self.config['with_damp'])

        if mode_indices is not None:
            mac_difference_desired_modes = mac_hs2_difference[mode_indices, :]
        else:
            mac_difference_desired_modes = mac_hs2_difference

        return mac_difference_desired_modes, mac_difference_to_ref, mac_difference_to_ref_last_op, mac_hs2_difference_to_ref_last_op

    def pick_modes_based_on_reference(self, mode_indices=None):

        path_to_cmb = os.path.join(self.iteration_run_directory, 'htc', self.config['cmb_filename'])
        path_to_bin = os.path.join(self.iteration_run_directory, 'htc', self.config['cmb_filename'][:-3] + 'bin')
        n_op_points = self.config['n_op_points']
        nmodes = self.config['nmodes']
        ndofs = self.config['ndofs']

        path_to_reference_bin = self.config['path_to_reference_bin']
        path_to_reference_cmb = self.config['path_to_reference_cmb']

        full_modal_matrix_ref, freq_ref, damp_ref, realpart_ref, imagpart_ref = gather_modal_matrices(path_to_reference_bin, path_to_reference_cmb,
                                                                                                      n_op_points, nmodes, ndofs)

        full_modal_matrix_analysis, freq_analysis, damp_analysis, realpart_analysis, imagpart_analysis = gather_modal_matrices(path_to_bin, path_to_cmb,
                                                                                                                               n_op_points, nmodes, ndofs)

        full_mac_matrix, full_mac_hs2_matrix = get_mac_matrices_first_op(full_modal_matrix_ref, freq_ref, damp_ref, realpart_ref, imagpart_ref,
                                                                         full_modal_matrix_analysis, freq_analysis, damp_analysis, realpart_analysis, imagpart_analysis,
                                                                         with_damp=self.config['with_damp'])

        picked_mode_indices = []
        picked_mode_indices_hs2 = []
        mode_indices_ref = [int(mode_idx) for mode_idx in self.config['mode_indices_ref']]
        picked_mode_indices = full_mac_matrix[mode_indices_ref].argmax(axis=1)
        picked_mode_indices_hs2 = full_mac_hs2_matrix[mode_indices_ref].argmax(axis=1)

        return picked_mode_indices, picked_mode_indices_hs2, full_mac_matrix, full_mac_hs2_matrix
