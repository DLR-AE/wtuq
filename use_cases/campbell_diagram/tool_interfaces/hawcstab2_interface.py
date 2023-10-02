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
        try:
            self.cmb = os.path.join(self.iteration_run_directory, "HAWC2.cmb")
            self.amp= os.path.join(self.iteration_run_directory, "HAWC2.amp")
            self.opt= os.path.join(self.master_file_path, "Control", "IWT_7.5-164_Rev-2.5.2_HS2_coarse_stability.opt")
            HS2_results = HS2Data(self.cmb, self.amp, self.opt)
            HS2_results.read_data()
    
            # extracting damping of 'critical' mode
            # for a HS2 run with multiple operating points, the HS2_results.modes.damping matrix has the shape:
            # # of modes x # of wind speeds x 2
    
            # find index for windspeed == 12 m/s
            desired_wind_speed = 12  # m/s
            idx_12ms = HS2_results.modes.damping[0, :, 0].tolist().index(desired_wind_speed)
    
            # find lowest damping for this wind speed
            # NOTE! there are some -100 entries, so only damping values up to -10 % are assumed to be feasible
            all_damping_values = HS2_results.modes.damping[:, idx_12ms, 1]
            accepted_damping_values = all_damping_values[(all_damping_values > -10)]
    
            result_dict = dict()
            result_dict['damping_criticalmode'] = min(accepted_damping_values)
            print(result_dict['damping_criticalmode'])
            return result_dict

        except:
            print('no HS2 result data were found')