import os

from tool_interfaces.HS2ASCIIReader import HS2Data
from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError
from tool_interfaces.hawc2_interface import HAWC2Model


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

        except:
            print('no HS2 result data were found')