"""
@author: Sarah Mueller <smueller2@nordex-online.com>
@date: 17.03.2021
"""

# global lib
import sys
import os
import xml.dom.minidom
import shutil
import numpy as np
import subprocess
from copy import deepcopy
from scipy import interpolate

# local lib
sys.path.append(r"D:\12_Projekte\QuexUS\03_UQ\coding")
from compressed_int_loader import CompressedIntLoader
from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError


class AlaskaModel(SimulationModel):
    """
    alaska/Wind model interface

    Parameters
    ----------
    iteration_run_directory : string
        Path to directory of this framework iteration
    config : dict
        User-specified settings for this tool

    Attributes
    ----------
    GEBT_template
    fef_template
    alaska_exe
    run_directory
    alaska_model
    alaska_temp
    """
    def __init__(self, iteration_run_directory, config):
        # super(AlaskaModel, self).__init__() # super class constructor is called
        self.GEBT_template = config['path_GEBT_template']
        self.fef_template = config['path_fef_template']
        self.alaska_exe = config['path_alaska_exe']
        self.run_directory = iteration_run_directory
        self.alaska_model = config['path_alaska_model']
        self.alaska_temp = config['path_alaska_templates']

    def create_simulation(self, blade_data):
        """
        Setup modified alaska model

        Parameters
        ----------
        blade_data : dict
            Dictionary with structural blade data

        Notes
        -----
        blade data: alaska specific stiffness and mass matrices
        blade_data['alaska_6x6']['stiffness_matrix'][blade section id][row][column]
        blade_data['alaska_6x6']['mass_matrix'][blade section id][row][column]
        """

        # copy alaska model
        src_ala = os.path.join(self.alaska_temp, 'alaska_model_UQ_temp')
        dest_ala = os.path.join(self.run_directory, 'alaska_model')
        shutil.copytree(src_ala, dest_ala)
        shutil.copyfile(os.path.join(self.alaska_temp, 'Includes.dat'), os.path.join(self.run_directory, 'Includes.dat'))

        # write parameters
        src_param = os.path.join(self.alaska_temp, 'Parameters_UQ_temp.dat')
        dest_param = os.path.join(self.run_directory, 'alaska_model', 'Parameters', 'Parameters.dat')
        shutil.copyfile(src_param, dest_param)
        with open(dest_param,'r') as f:
            f_param = f.read()
        with open(dest_param,'w') as f:
            f.write(f_param % os.path.join(self.run_directory, 'alaska_model'))

        # write load cases
        src_LC = os.path.join(self.alaska_temp, 'LC_UQ_temp.xml')
        dest_LC = os.path.join(self.run_directory, 'alaska_model', 'LoadCases', 'LC_UQ.xml')
        shutil.copyfile(src_LC, dest_LC)
        with open(dest_LC, 'r') as f:
            f_LC = f.read()
        with open(dest_LC, 'w') as f:
            f.write(f_LC % os.path.join(self.run_directory, 'alaska_model'))

        # write GEBT xml
        #self.write_GEBT_xml(self.GEBT_template, blade_data)
        # write fef
        self.write_fef(self.fef_template, blade_data)

    def write_GEBT_xml(self, file, blade_data):
        """
        function that writes GEBT.xml file
        """
        # parsing the xml file
        xml_doc = xml.dom.minidom.parse(file)
        # get the xml data tree
        xml_tree = xml_doc.documentElement

        # change stiffness and mass matrix
        for section_id, cross_section in enumerate(xml_tree.getElementsByTagName("CROSS_SECTION")):
            # write stiffness data into the xml
            k_txt = np.array2string(blade_data['alaska_6x6_element_ref']['stiffness_matrix'][section_id])
            k_txt = k_txt.replace("[", "")
            k_txt = k_txt.replace("]]", "]")
            k_txt = k_txt.replace("]", "\n")
            cross_section.getElementsByTagName("FLEXIBILITY")[0].firstChild.replaceWholeText("\n " + k_txt)
            # write mass data into the xml
            m_txt = np.array2string(blade_data['alaska_6x6_element_ref']['mass_matrix'][section_id])
            m_txt = m_txt.replace("[", "")
            m_txt = m_txt.replace("]]", "]")
            m_txt = m_txt.replace("]", "\n")
            cross_section.getElementsByTagName("MASS")[0].firstChild.replaceWholeText("\n " + m_txt)

        # write xml
        with open(file.replace(".xml", "_UQ_temp.xml"), "w")as f:
            xml_string = xml_tree.toprettyxml()
            f.write(self.remove_blank_lines_xml(xml_string))

    def write_fef(self, file, blade_data):
        """
        function that writes .fef file
        """
        # parsing the xml file
        xml_doc = xml.dom.minidom.parse(file)
        # get the xml data tree
        xml_tree = xml_doc.documentElement

        # define permutation vector
        perm_vec = np.array([0, 4, 5, 3, 1, 2])
        stiffmat_perm = np.empty([6, 6])
        massmat_perm = np.empty([6, 6])

        # get indices for the upper-triangle
        i_upper = np.triu_indices(6, k=0)
        # get number of blade sections
        i_blade_section = len(blade_data['alaska_6x6_element_ref']['stiffness_matrix'])
        # initialize string
        matrices_str = ""
        # calculate averaged matrices (alaska setting: BladeDescriptionType = mean)
        for i in range(0, i_blade_section - 1):
            stiffmat_ave = (blade_data['alaska_6x6_element_ref']['stiffness_matrix'][i] + blade_data['alaska_6x6_element_ref']['stiffness_matrix'][i + 1]) / 2
            massmat_ave = (blade_data['alaska_6x6_element_ref']['mass_matrix'][i] + blade_data['alaska_6x6_element_ref']['mass_matrix'][i + 1]) / 2
            # apply permutation (not needed for mass matrix)
            for j in range(0, 6):
                for k in range(0, 6):
                    stiffmat_perm[j, k] = deepcopy(stiffmat_ave[perm_vec[j], perm_vec[k]])  # use a deepcopy
                    massmat_perm[j, k] = deepcopy(massmat_ave[perm_vec[j], perm_vec[k]])  # use a deepcopy
            # get upper triangle and convert to string
            stiffmat_fef = np.array2string(stiffmat_perm[i_upper])  # use permutated matrix
            massmat_fef = np.array2string(massmat_ave[i_upper])  # use averaged matrix
            # format to one line
            matrices_str = (matrices_str + "\n" + "M" + np.str(i+1) + " " + massmat_fef + "\n" + "C" + np.str(i+1) + " " + stiffmat_fef)

        # remove bracket
        matrices_str = matrices_str.replace("[", "")
        matrices_str = matrices_str.replace("]", "")
        # change matrices
        xml_tree.getElementsByTagName("TABLE")[6].firstChild.data = ("\n " + matrices_str + "\n")

        # write fef
        head, tail = os.path.split(file)
        with open(file.replace(".fef", "_UQ_temp.fef"), "w")as f:
            xml_string = xml_tree.toprettyxml()
            f.write(self.remove_blank_lines_xml(xml_string))

        src = os.path.join(head, 'FEBlade_UQ_temp.fef')
        dest = os.path.join(self.run_directory, 'alaska_model', 'SubModels', 'Blade_%i', 'FEBlade.fef')
        for blade_id in range(1, 4):
            shutil.copyfile(src, dest % blade_id)
        # check if fef file is not empty
        for blade_id in range(1, 4):
            if os.stat(dest % blade_id).st_size == 0:
                print('error-fef: empty fef in blade ')
                print(blade_id)
                print(self.run_directory)
                shutil.copyfile(src, dest % blade_id)

    def remove_blank_lines_xml(self, xml_str):
        """ Remove blank lines from xml string """
        new_str = ""
        for line in xml_str.split("\n"):
            if not line.strip() == "\t" and not line.strip() == "":
                new_str += line + "\n"
        return new_str

    def run_simulation(self):
        """
        Execute alaska simulation
        """
        # define command line
        command = self.alaska_exe+' '+os.path.join(self.run_directory, 'alaska_model', 'LoadCases', 'LC_UQ.xml')
        # execute command
        subprocess.run(command, cwd=self.run_directory)

    def extract_results(self):
        """
        Extract the useful, desired signals which can be used for damping determination.

        Returns
        -------
        result_dict : dict
            Collected simulation results. Key names have to match with var_names specified in the postprocessor
            settings in the config
        """
        # get time series data
        #c = CompressedIntLoader(self.alaska_int)
        c = CompressedIntLoader(os.path.join(self.run_directory, 'alaska_model', 'Results', 'IWT-7.5-164_UQ_ref_12ms_ref_HS2.int'))
        # get sensor names
        names = [x.name for x in c.sensors()]
        # hard coded radial sensor positions (geometry information only in blade description, not in int file)
        radpos = np.array([2, 3.14290000000000, 4.20070000000000, 5.00120000000000, 6.10070000000000, 7.10080000000000,
                           8.20070000000000, 8.90060000000000, 10.1002000000000, 10.9000000000000, 12.2000000000000, 13,
                           13.9990000000000, 14.8990000000000, 15.9990000000000, 17.0520000000000, 17.9990000000000,
                           18.9460000000000, 20.1040000000000, 21.1570000000000, 21.9990000000000, 23.1040000000000,
                           23.9469000000000, 24.9989000000000, 25.8989000000000, 26.9989000000000, 27.9468000000000,
                           29.0997000000000, 29.9996000000000, 30.9985000000000, 31.8984000000000, 32.9993000000000,
                           33.9990000000000, 34.9988000000000, 36.0985000000000, 36.9983000000000, 37.9979000000000,
                           38.9615000000000, 40.0320000000000, 40.9965000000000, 41.9958000000000, 42.9591000000000,
                           44.0292000000000, 44.9934000000000, 45.9394000000000, 46.9912000000000, 47.9369000000000,
                           48.9885000000000, 49.9339000000000, 50.9851000000000, 51.9831000000000, 52.9810000000000,
                           53.9785000000000, 54.9761000000000, 55.9203000000000, 56.9701000000000, 57.9397000000000,
                           58.9093000000000, 59.9858000000000, 61.0543000000000, 61.9499000000000, 62.9450000000000,
                           63.9035000000000, 64.9694000000000, 65.9271000000000, 66.9205000000000, 67.9130000000000,
                           68.8528000000000, 69.8967000000000, 70.9406000000000, 71.8782000000000, 72.8159000000000,
                           73.9640000000000, 75.0327000000000, 75.8871000000000, 76.8474000000000, 77.7005000000000,
                           78.7401000000000, 79.5692000000000, 80.5638000000000, 80.7605000000000, 81.2520000000000,
                           81.7235900000000])

        result_dict = dict()
        result_dict['time'] = self.get_time(c)
        result_dict['radpos'] = radpos
        result_dict['deflectionInPlane_b1'] = -self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['UYR1' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['deflectionInPlane_b2'] = -self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['UYR2' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['deflectionInPlane_b3'] = -self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['UYR3' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['deflectionOutOfPlane_b1'] = self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['UZR1' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['deflectionOutOfPlane_b2'] = self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['UZR2' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['deflectionOutOfPlane_b3'] = self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['UZR3' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['torsion_b1'] = self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['RXV1' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['torsion_b2'] = self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['RXV2' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['torsion_b3'] = self.get_sensor_data_over_blade_position(c, [names.index(i) for i in
                                                                                 ['RXV3' + str(i) for i in
                                                                                  range(1, len(radpos))]], radpos)
        result_dict['towertop_fa'] = self.get_sensor_data(c, names.index('GZ'))
        result_dict['towertop_ss'] = self.get_sensor_data(c, names.index('GY'))

        return result_dict

    def get_time(self, c):
        """ get time vector """
        time = np.arange(c.timeSerie(0).tstart, len(c.timeSerie(0).data)*c.timeSerie(0).tstep, c.timeSerie(0).tstep)
        return time

    def get_sensor_data(self, c, idx):
        sensordata = c.timeSerie(idx).data * c.timeSerie(idx).scale
        return sensordata

    def get_sensor_data_over_blade_position(self, c, idx, radpos):
        # get sensordata
        sensordata = np.array([c.timeSerie(i).data for i in idx])
        sensordata = sensordata.astype('float64')  # change dtype from int32 to float64
        scale = np.array([c.timeSerie(i).scale for i in idx])
        # scale sensordata
        for i in range(len(scale)):
            sensordata[i] = sensordata[i] * scale[i]
        # extrapolate sensordata to last radial position
        sensordata_extrap = np.zeros((len(sensordata) + 1, len(sensordata[0])))
        sensordata_extrap[0:-1, :] = sensordata
        for i in range(len(sensordata[0])):
            f = interpolate.interp1d(radpos[0:-1], sensordata[:, i], fill_value='extrapolate')
            sensordata_extrap[-1, i] = f(radpos[-1])  # use interpolation function returned by 'interp1d'
        return sensordata_extrap


# debug config
#  1) set run_type to "reference" (unchanged blade data)
#  2) set run_type to "test" (with blade data changes)
#
# execute:
# use "python run_analysis.py -c config/alaska/<datei>"

if __name__ == "__main__":

    # raise NotImplementedError('No stand-alone running possible yet')
    alaska_example = AlaskaModel()
    alaska_example.create_simulation('')
    alaska_example.run_simulation()
    alaska_example.extract_results()