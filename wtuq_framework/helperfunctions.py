"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 11.02.2020

Some general purpose subroutines used in the framework
"""

import numpy as np
import json
import h5py


def equal_dicts(dict1, dict2, precision=1e-15):
    """
    Check if the two dictionaries with numerical values are equal

    Parameters
    ----------
    dict1 : dict
        Dictionary with numerical values
    dict2 : dict
        Dictionary with numerical values
    precision : {float, int}, optional
        Precision level to which numerical values in the dictionaries are identical
        Default value is 1e-15

    Returns
    -------
    match : boolean
        flag if dictionaries are identical

    Notes
    -----
    comparing could also be done by dict1 == dict2, however this might
    give false negatives for rounding errors

    """
    # check if the two dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        return False

    # check if the two dictionaries have the same values
    for key in dict1:
        if abs(dict1[key]-dict2[key]) < abs(precision):
            continue
        else:
            return False

    return True


def save_dict_json(file, data):
    """
    Save dictionary with json package

    Parameters
    ----------
    file : string
        path to output file
    data : dict
        dictionary to be saved
    """
    with open(file, 'w') as outfile:
        json.dump(data, outfile)


def load_dict_json(file):
    """
    Load dictionary from json file

    Parameters
    ----------
    file : string
        path to json file

    Returns
    -------
    data : dict
        parsed data
    """
    with open(file) as json_file:
        return json.load(json_file)


def save_dict_h5py(filename, dic):
    """
    Save data dictionary with h5py package. Dictionary can be nested.
    Allowed dictionary value types: np.ndarray, np.int64, np.float64, float, str, bytes

    Parameters
    ----------
    filename : string
        path to output file
    dic : dict
        dictionary to be saved

    See Also
    --------
    recursively_save_dict_contents_to_group
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Save dictionary to open h5py file

    Parameters
    ----------
    h5file : h5py file
        opened h5py file
    path : string
        path (inside h5py file) to write dic
    dic : dict
        dictionary to be written to path within h5file

    Raises
    ------
    ValueError
        If the item saved in the can not be saved in a h5py file

    See Also
    --------
    save_dict_h5py
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, int, np.float64, float, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, list):
            # assumed that all elements in array are the same type
            if isinstance(item[0], (np.ndarray, np.int64, int, np.float64, float)):
                h5file[path + key] = np.array(item)
            elif isinstance(item[0], str):
                h5file[path + key] = [n.encode("ascii", "ignore") for n in item]
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


def load_dict_h5py(filename):
    """
    Load data dictionary with h5py package. Dictionary can be nested.

    Parameters
    ----------
    filename : string
        Path to h5py file

    Returns
    -------
    loaded_data : dict
        Dictionary with data loaded from h5py file

    See Also
    --------
    recursively_load_dict_contents_from_group
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    """
    read from opened h5py file

    Parameters
    ----------
    h5file : h5py file
        opened h5py file
    path : string
        path (inside h5py file) to read

    Returns
    -------
    ans : dict
        Loaded dictionary sub-dataset
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if isinstance(item[()], bytes):
                ans[key] = item[()].decode("ascii", "ignore")
            elif isinstance(item[()], (np.int64, np.int32, int, np.float64, np.float64, float, str)):
                ans[key] = item[()]
            elif isinstance(item[()], np.ndarray):
                if isinstance(item[()][0], bytes):
                    ans[key] = [some_string.decode("ascii", "ignore") for some_string in item[()]]
                else:
                    ans[key] = item[()]

        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def check_min_req_data(available_data, required_data):
    """
    Checks if the available data contains all required data variables

    Parameters
    ----------
    available_data : {dict, list, array}
        Iterable object representing all available data variables
    required_data : {dict, list, array}
        Iterable object representing all required data variables

    Returns
    -------
    missing_fields : list
        List of all variables in the required_data object which are not in the the available_data object
    """
    if isinstance(available_data, dict):
        available_data = available_data.keys()
    if isinstance(required_data, dict):
        required_data = required_data.keys()

    missing_fields = np.setdiff1d(list(required_data), list(available_data))
    return list(missing_fields)


def find_corresponding_axis(arr, size, max_dim):
    """
    Finds dimension of array (arr) which corresponds to the size integer

    Parameters
    ----------
    arr : nd.array
        array to be investigated
    size : int
        dimension size which has to be found in array
    max_dim : int
        maximum allowed dimension of the array

    Returns
    -------
    dim : int
        dimension of array which matches with size
    """

    if len(arr.shape)>max_dim:
        raise ValueError('Only ' + str(max_dim) + '-dimensional arrays are allowed')

    if arr.shape.count(size) == 0:
        print('WARNING: Dimensions of arrays to be correlated do not match')
        return -1
    elif arr.shape.count(size) > 1:
        print('WARNING: requested dimension occurs multiple times. First dimension is used.')
    else:
        dim = arr.shape.index(size)
        return dim


def moving_average(arr, window):
    """
    Applies moving average filter of <window> number of samples on the arr

    Parameters
    ----------
    arr : nd.array
        input array
    window : int
        number of samples over which is averaged

    Returns
    -------
    arr : nd.array
        filtered signal

    Notes
    -----
    If the window is even, the applied window will be effectively window + 1
    """

    arr = np.squeeze(arr) # only 1D arrays
    filtered_arr = np.zeros(arr.shape)
    for ii in range(filtered_arr.size):
        idx1 = ii - window//2
        idx2 = ii + window//2
        if idx1 < 0:
            idx1 = 0
        if idx2 >= filtered_arr.size:
            idx2 = filtered_arr.size

        filtered_arr[ii] = np.mean(arr[idx1:idx2+1])

    return filtered_arr


def get_CPU(i, run_type):
    """
    Parses user input for n_CPU

    Parameters
    ----------
    i : int
        number of CPUs requested by user in config
    reference_run : bool
        flag for reference_run

    Returns
    -------
    n_CPU : {int, None}
        input for uncertainpy
    """
    if i == 1:
        return None
    elif run_type=='test' or run_type=='reference':
        return None
    else:
        return i