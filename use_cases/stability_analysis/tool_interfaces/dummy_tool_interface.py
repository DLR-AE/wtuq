"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 10.12.2020
"""

from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError


class DummyToolModel(SimulationModel):
    """
    Dummy model interface
    """

    def __init__(self, iteration_run_directory, config):
        pass

    def create_simulation(self, blade_data):
        """
        """
        pass

    def run_simulation(self):
        """
        """
        pass

    def extract_results(self):
        """
        """
        return dict()