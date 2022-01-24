"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 25.11.2020
"""

import numpy as np
import os

from tool_interfaces.simulation_model_interface import SimulationModel, SimulationError

from pyBladed.model import BladedModel as BladedAPIModel
from pyBladed.results import BladedResult


class BladedModel(SimulationModel):
    """
    Bladed model interface

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
    def __init__(self, run_directory, config):
        self.bl_model = BladedAPIModel(os.path.abspath(config['template_project']))
        self.bl_model.suppress_logs()
        self.bl_st = dict()
        self.run_directory = run_directory
        self.geometry_definition = config['becas_geometry']

    def create_simulation(self, blade_data):
        """
        Setup modified bladed model

        Parameters
        ----------
        blade_data : dict
            Dictionary with structural blade data
        """
        h2_geom = self.parse_h2_becas_geom(self.geometry_definition)

        self.transfer_h2_to_bladed(blade_data['h2_beam'], h2_geom)
        self.modify_prj()

    def run_simulation(self):
        """
        Execute Bladed simulation
        """
        self.result_prefix = 'stab_analysis_run'
        self.bl_model.run_simulation(result_directory=self.run_directory, prefix=self.result_prefix)

    def extract_results(self):
        """
        Extract the useful, desired signals which can be used for damping determination.

        Returns
        -------
        result_dict : dict
            Collected simulation results. Key names have to match with var_names specified in the postprocessor
            settings in the config
        """
        bladed_result = BladedResult(self.run_directory,
                                     self.result_prefix)  # results.BladedResult(self.run_directory, self.result_prefix)
        bladed_result.scan()
        result_dict = dict()
        result_dict['time'] = np.copy(bladed_result['Time from start of simulation'])
        result_dict['torsion_b1'], torsion_b1_metadata = np.copy(bladed_result['Blade 1 rotation about z (plane)'])
        result_dict['torsion_b2'], torsion_b2_metadata = np.copy(bladed_result['Blade 2 rotation about z (plane)'])
        result_dict['torsion_b3'], torsion_b3_metadata = np.copy(bladed_result['Blade 3 rotation about z (plane)'])
        result_dict['deflectionInPlane_b1'], deflectionInPlane_b1_metadata = np.copy(
            bladed_result['Blade 1 y-deflection (in rotor plane)'])
        result_dict['deflectionInPlane_b2'], deflectionInPlane_b2_metadata = np.copy(
            bladed_result['Blade 2 y-deflection (in rotor plane)'])
        result_dict['deflectionInPlane_b3'], deflectionInPlane_b3_metadata = np.copy(
            bladed_result['Blade 3 y-deflection (in rotor plane)'])
        result_dict['deflectionOutOfPlane_b1'], deflectionOutOfPlane_b1_metadata = np.copy(
            bladed_result['Blade 1 x-deflection (perpendicular to rotor plane)'])
        result_dict['deflectionOutOfPlane_b2'], deflectionOutOfPlane_b2_metadata = np.copy(
            bladed_result['Blade 2 x-deflection (perpendicular to rotor plane)'])
        result_dict['deflectionOutOfPlane_b3'], deflectionOutOfPlane_b3_metadata = np.copy(
            bladed_result['Blade 3 x-deflection (perpendicular to rotor plane)'])
        result_dict['towertop_fa'] = np.copy(bladed_result['Nacelle fore-aft displacement'])
        result_dict['towertop_ss'] = np.copy(bladed_result['Nacelle side-side displacement'])
        result_dict['radpos'] = np.copy(np.array(torsion_b1_metadata['AXIVAL']))
        return result_dict

    def parse_h2_becas_geom(self, h2_becas_geom_luh):
        """
        Read BECAS related geometric data (extract from LUH report, 83 stations)
        83 stations
        - Radial position (along blade)
        - chord
        - x/c
        - twist

        Parameters
        ----------
        h2_becas_geom_luh : string
            Path to geometry file

        Returns
        -------
        h2_geom : dict
            Dictionary with radius, chord and twist data of the blade
        """
        table_geo_h2 = np.loadtxt(h2_becas_geom_luh, skiprows=2)

        r83_h2 = table_geo_h2[:, 0]
        chord_h2 = table_geo_h2[:, 1]
        twist83_h2 = table_geo_h2[:, 3]

        h2_geom = dict()
        h2_geom['r83_h2'] = r83_h2
        h2_geom['chord_h2'] = chord_h2
        h2_geom['twist83_h2'] = twist83_h2

        return h2_geom

    def parse_h2_st(self, h2_st_file):
        """
        Read a HAWC2 .st file

        Parameters
        ----------
        h2_st_file : string
            Path to HAWC2 structural blade data file

        Returns
        -------
        h2_st : dict
            HAWC2 blade data

        Notes
        -----
        HAWC2 structural parameters:
            arc
            m_pm
            x_cg
            y_cg
            ri_x
            ri_y
            x_sc
            y_sc
            E
            G
            I_x
            I_y
            I_T
            k_x
            k_y
            A
            theta_s
            x_ec
            y_ec
        """
        h2_st = dict()
        table_struct_h2 = np.loadtxt(h2_st_file, skiprows=8)
        h2_st['arc'] = table_struct_h2[:, 0]
        h2_st['m_pm'] = table_struct_h2[:, 1]
        h2_st['x_cg'] = table_struct_h2[:, 2]
        h2_st['y_cg'] = table_struct_h2[:, 3]
        h2_st['ri_x'] = table_struct_h2[:, 4]
        h2_st['ri_y'] = table_struct_h2[:, 5]
        h2_st['x_sc'] = table_struct_h2[:, 6]
        h2_st['y_sc'] = table_struct_h2[:, 7]
        h2_st['E'] = table_struct_h2[:, 8]
        h2_st['G'] = table_struct_h2[:, 9]
        h2_st['I_x'] = table_struct_h2[:, 10]
        h2_st['I_y'] = table_struct_h2[:, 11]
        h2_st['I_T'] = table_struct_h2[:, 12]
        h2_st['k_x'] = table_struct_h2[:, 13]
        h2_st['k_y'] = table_struct_h2[:, 14]
        h2_st['A'] = table_struct_h2[:, 15]
        h2_st['theta_s'] = table_struct_h2[:, 16]
        h2_st['x_ec'] = table_struct_h2[:, 17]
        h2_st['y_ec'] = table_struct_h2[:, 18]

        return h2_st

    def transfer_h2_to_bladed(self, h2_st, h2_geom):
        """
        Transfer structural blade data from HAWC2 to Bladed format

        Parameters
        ----------
        h2_st : dict
            Dictionary with HAWC2 structural blade data
        h2_geom : dict
            Dictionary with HAWC2 geometry data

        Notes
        -----
        HAWC2 structural parameters:
            arc
            m_pm
            x_cg
            y_cg
            ri_x
            ri_y
            x_sc
            y_sc
            E
            G
            I_x
            I_y
            I_T
            k_x
            k_y
            A
            theta_s
            x_ec
            y_ec

        Bladed structural parameters:
            CentreOfMass.X
            CentreOfMass.Y
            massPerUnitLength
            IntertiaPerUnitLength
            RadiiOfGyrationRatio
            MassAxisOrientationRadians
            bendingStiffnessXp
            bendingStiffnessYp
            PrincipalAxisOrientationRadians
            torsionalStiffness
            AxialStiffness
            ShearCentre.X
            ShearCentre.Y
            shearStiffnessXp
            shearStiffnessYp

        """
        bl_st = dict()

        ## STRUCTURE

        # CoM (h2: relative to c/2, b: relative to LE)
        xcm_h2 = h2_st['x_cg']
        ycm_h2 = h2_st['y_cg']
        bl_st['CentreOfMass.X'] = ycm_h2 / h2_geom['chord_h2'] * 100.
        bl_st['CentreOfMass.Y'] = (1 / 2 - xcm_h2 / h2_geom['chord_h2']) * 100.

        # mass axis orientation
        # (defaults in Bladed to twist)
        mao_bl = np.zeros((83))
        bl_st['massAxisOrientationRadians'] = mao_bl

        # mass per unit length
        m_l_bl = h2_st['m_pm']
        bl_st['massPerUnitLength'] = m_l_bl

        # polar mass moment of inertia per unit length
        # H2: separate radii of gyration for bending around principle axes
        #     relative to elastic center
        # B: Polar moment of inertia around local z-axis in CoM
        rix_h2 = h2_st['ri_x']
        riy_h2 = h2_st['ri_y']
        mass_inertia_bl = m_l_bl * (rix_h2 ** 2 + riy_h2 ** 2)  # order and sign not significant
        # add Steiner for translation from EC o CoM
        # use H2 data for the distance since it is given in [m]
        xea_h2 = h2_st['x_ec']
        yea_h2 = h2_st['y_ec']
        d_h2 = np.sqrt((xcm_h2 - xea_h2) ** 2 + (ycm_h2 - yea_h2) ** 2)
        mass_inertia_bl -= m_l_bl * d_h2 ** 2
        bl_st['IntertiaPerUnitLength'] = mass_inertia_bl

        # ratio of radii of gyration (defaults to thickness/chord ratio)
        # ratio y/x in Bladed -> rix/riy
        # rad_gyr_ratio_bl = np.zeros((83))
        rad_gyr_ratio_bl = rix_h2 / riy_h2
        bl_st['RadiiOfGyrationRatio'] = rad_gyr_ratio_bl

        # principle axis orientation
        # (B: defaults to twist)
        # principle_axis_rot_bl = np.zeros((83))
        # H2: rotation from twisted reference axis
        # B: rotation from blade reference axis
        th_s83_h2 = h2_st['theta_s']
        principle_axis_rot_bl = -(h2_geom['twist83_h2'] + th_s83_h2)
        # note angle has to given in Radians to the Bladed API, but will be
        # listed in degrees in the GUI
        bl_st['PrincipalAxisOrientationRadians'] = np.pi / 180 * principle_axis_rot_bl

        # shear center
        xsc_h2 = h2_st['x_sc']
        ysc_h2 = h2_st['y_sc']
        xsc_loc_bl = ysc_h2 / h2_geom['chord_h2'] * 100.
        ysc_loc_bl = (1 / 2 - xsc_h2 / h2_geom['chord_h2']) * 100.
        bl_st['ShearCentre.X'] = xsc_loc_bl
        bl_st['ShearCentre.Y'] = ysc_loc_bl

        # bending stiffness x/y
        # H2: with respect to elastic center
        # Bladed: idem
        E = h2_st['E']
        Ix_h2 = h2_st['I_x']
        Iy_h2 = h2_st['I_y']
        EIx_bl = E * Iy_h2
        EIy_bl = E * Ix_h2
        bl_st['bendingStiffnessXp'] = EIx_bl
        bl_st['bendingStiffnessYp'] = EIy_bl

        # torsion stiffness
        G = h2_st['G']
        K = h2_st['I_T']
        # GIt_bl = G * (Ix_h2 + Iy_h2)
        GIt_bl = G * K
        bl_st['torsionalStiffness'] = GIt_bl

        # shear stiffness
        # shear factors
        kx_h2 = h2_st['k_x']
        ky_h2 = h2_st['k_y']
        A = h2_st['A']
        shear_stiff_x_bl = G * A * ky_h2
        shear_stiff_y_bl = G * A * kx_h2
        bl_st['shearStiffnessXp'] = shear_stiff_x_bl
        bl_st['shearStiffnessYp'] = shear_stiff_y_bl

        # axial stiffness E*A
        axial_stiff_bl = E * A
        bl_st['AxialStiffness'] = axial_stiff_bl

        self.bl_st = bl_st

    def modify_prj(self):
        """
        Modify Bladed .prj description with new Bladed structural description
        """
        bl_st = self.bl_st
        # in bladed we have a section at 0m and at 0.1m -> so the first value
        # of the bladed data has to be doubled
        for key in bl_st:
            bl_st[key] = np.hstack((bl_st[key][0], bl_st[key]))

        self.bl_model.modify_blade(bl_st)
