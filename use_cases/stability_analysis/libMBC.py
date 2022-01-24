"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 23.08.2021

Multiblade Coordinate Transformation for 3-bladed turbines

More info, see here: https://www.nrel.gov/docs/fy08osti/42553.pdf

IMPORTANT ASSUMPTIONS:
- 3 bladed turbines
- all sensors are in the rotating reference frame
- sensors have the nomenclature: something_b1, something_b2, something_b3, -> blade order given as 'CW' or 'CCW'
- the sensors for the different blades have the same length and are positioned at the same radial positions
- time is given in a vector with name 'time'
- only first order states (positions)
"""
import numpy as np


class MBCTransformation(object):
    """
    Class for the Multiblade Coordinate Transformation of rotating systems

    Parameters
    ----------
    time : array
        1D time vector
    signal : array
        1D or 2D signal array (time dimension on axis=0)
    var_names : list
        list with names of the used sensors
    blade_order : string, optional
        'CW' or 'CCW', direction in which blades are ordered, default is 'CW'
    omega : float, optional
        rotational speed in rpm, default is 10
    azimuth : array, optional
        1D array with azimuth position blade 1, default is None

    Attributes
    ----------
    time : array
        1D time vector
    signal : array
        1D or 2D signal array (time dimension on axis=0)
    var_names : list
        list with names of the used sensors
    omega : float
        rotational speed in rpm, default is 10
    azimuth : array
        1D array with azimuth position blade 1, default is None
    x_r_array : array
        signals in rotating reference system
    x_nr_array: array
        signals in non-rotating reference system
    """
    def __init__(self, time, signal, var_names, blade_order='CW', omega=10, azimuth=None):
        self.time = time
        self.signal = signal
        self.var_names = var_names
        self.omega = omega
        self.azimuth = azimuth

        self.check_and_rearrange_input(blade_order)

    def check_and_rearrange_input(self, blade_order):
        """
        Check that provided input matches with script assumptions and restructure input signal array

        Structure of the input vector has to be: [sensor1_b1, sensor1_b2, sensor3_b2, sensor2_b1, ...], see eq.6 Bir

        Parameters
        ----------
        blade_order : string
            'CW' or 'CCW', direction in which blades are ordered

        Notes
        -----
        var_names is f.ex. torsion_b1, deflectionInPlane_b3, aoa_b1, cl_b3, etc.
        base_var_names is f.ex. torsion, deflectionInPlane, etc.
        sensors are torsion at XX m along blade
        """

        assert self.time.size == self.signal.shape[0], "size of time vector and corresponding signal array don't match"
        assert len(self.var_names) % 3 == 0, \
            "Number of var_names has to be divisible by three (assumed to be number of blades)"
        assert self.signal.shape[1] % len(self.var_names) == 0, \
            "size of signal array does not match with number of variables"
        nr_of_sensors_per_var_per_blade = int(self.signal.shape[1] / len(self.var_names))

        # procedure for restructuring
        # 1) list base_var_names
        # 2) loop over base_var_names -> based on var_names and nr_of_sensor_per_var_per_blade find where sensor data
        #    is in the signal array.
        base_var_names = set([])
        for var in self.var_names:
            base_var_names.add(var[:-3])
            assert var[-2:] == 'b1' or var[-2:] == 'b2' or var[-2:] == 'b3', "MBC requires signal var_names to end in _b1, _b2 or _b3"

        if blade_order == 'CW':
            blade_order_list = ['_b1', '_b2', '_b3']
        else:
            blade_order_list = ['_b3', '_b2', '_b1']
        structured_index_list = list()
        for base_var in base_var_names:
            for ii_sensor in range(nr_of_sensors_per_var_per_blade):
                for blade in blade_order_list:
                    structured_index_list.append(
                        self.var_names.index(base_var + blade) * nr_of_sensors_per_var_per_blade + ii_sensor)

        self.x_r_array = self.signal[:, structured_index_list]

    def do_MBC(self):
        """
        Execute multiblade coordinate transformation. For details see: 'Multiblade Coordinate Transformation and Its
        Application to Wind Turbine Analysis', G.Bir

        Returns
        -------
        x_nr_array : array
            signals in non-rotating reference system
        """
        # initialize output array (signals in non-rotating reference system)
        x_nr_array = np.zeros(self.signal.shape)

        # loop over time vector
        total_number_sensors = self.signal.shape[1]
        for iter, t in enumerate(self.time):
            psi1 = 0 + (self.omega * 2 * np.pi / 60) * t
            psi2 = 2*np.pi / 3 + (self.omega * 2 * np.pi / 60) * t
            psi3 = 4*np.pi / 3 + (self.omega * 2 * np.pi / 60) * t

            t_tilde = np.array([[1, np.cos(psi1), np.sin(psi1)],
                                [1, np.cos(psi2), np.sin(psi2)],
                                [1, np.cos(psi3), np.sin(psi3)]])

            T_1 = np.kron(np.eye(int(total_number_sensors/3)), t_tilde)

            x_nr = np.linalg.solve(T_1, self.x_r_array[iter, :])
            x_nr_array[iter, :] = x_nr

            """
            -> not necessary (used for 2nd order states (velocities))
            t_tilde_2 = np.array([[1, -np.sin(psi1), np.cos(psi1)],
                                  [1, -np.sin(psi2), np.cos(psi2)],
                                  [1, -np.sin(psi3), np.cos(psi3)]])
            T_2 =
            T_MBC = 
            """

        return x_nr_array


if __name__ == "__main__":
    from wtuq_framework.helperfunctions import load_dict_h5py, save_dict_h5py

    test_dict = load_dict_h5py('<path to result_dict .hdf5 file>')

    time = test_dict['time']
    signal = np.zeros((20001, 83*3))
    signal[:, 0:83] = test_dict['torsion_b1'].T
    signal[:, 83:83*2] = test_dict['torsion_b3'].T
    signal[:, 83*2:] = test_dict['torsion_b2'].T

    var_names = ['torsion_b1', 'torsion_b2', 'torsion_b3']
    nr_sensors_per_var_per_blade = 83  # -> number of interpolated radial sections in framework

    example = MBCTransformation(time, signal, var_names, blade_order='CCW', omega=10, azimuth=None)
    mbc_array = example.do_MBC()
    output_dict = dict()
    output_dict['time'] = time
    output_dict['signal1_fixed_ref_sys_collective'] = mbc_array[:, 240]
    output_dict['signal2_fixed_ref_sys_cosine_cyclic'] = mbc_array[:, 241]
    output_dict['signal3_fixed_ref_sys_sine_cyclic'] = mbc_array[:, 242]
    save_dict_h5py('<path to output .hdf5 file>', output_dict)