"""
Extra module to avoid cyclic imports
"""


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