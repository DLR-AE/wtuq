"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 25.11.2020
"""

import numpy as np
import os

from tool_interfaces.bladed_interface import BladedModel
from tool_interfaces.simulation_model_interface import SimulationError

from pyBladed.results import BladedResult


class BladedLinModel(BladedModel):
    """
    Bladed-lin model interface

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

    def extract_results(self):
        """
        Extract the useful, desired signals which can be used for damping determination.

        Returns
        -------
        result_dict : dict
            Collected simulation results. Key names have to match with var_names specified in the postprocessor
            settings in the config
        """
        bladed_result = BladedResult(self.run_directory, self.result_prefix)
        bladed_result.scan()
        result_dict = dict()
        result_dict['Frequency'], damping_metadata = bladed_result['Frequency (undamped)']
        result_dict['Damping'], damping_metadata = bladed_result['Damping']
        result_dict['Damping'] = 100 * result_dict['Damping']  # damping ratio in %
        mode_names = damping_metadata['AXITICK']

        result_dict['damping_criticalmode'] = np.min(result_dict['Damping'])
        result_dict['criticalmode'] = mode_names[
            np.where(result_dict['Damping'] == result_dict['damping_criticalmode'])[1][0]]
        result_dict['criticalfrequency'] = result_dict['Frequency'][
            0, np.where(result_dict['Damping'] == result_dict['damping_criticalmode'])[1][0]]

        f = open(os.path.join(self.run_directory, 'damping_criticalmode.txt'), "w")
        f.write("Critical mode: " + str(result_dict['criticalmode']))
        f.write("\nDamping ratio: " + str(result_dict['damping_criticalmode']))
        f.write("\nFrequency: " + str(result_dict['criticalfrequency'][0]))
        f.close()

        f = open(os.path.join(self.run_directory, 'fulloverview.txt'), "w")
        f.write("Frequency: " + str(result_dict['Frequency']))
        f.write("\nDamping ratio: " + str(result_dict['Damping']))
        f.write("\nMode names: " + str(mode_names))
        f.close()

        # save operational data
        wind_speed = bladed_result['Nominal wind speed at hub position']
        pitch = bladed_result['Nominal pitch angle']
        rpm = bladed_result['Rotor speed']
        Power = bladed_result['Electrical power']
        Thrust = np.zeros(wind_speed.shape)

        opt_data = np.zeros((wind_speed.size, 5))
        opt_data[:, 0] = wind_speed
        opt_data[:, 1] = pitch
        opt_data[:, 2] = rpm
        opt_data[:, 3] = Power
        opt_data[:, 4] = Thrust

        np.savetxt(os.path.join(self.run_directory, 'modified.opt'), opt_data,
                   header='wind speed, pitch, rpm, power, thrust')

        return result_dict
