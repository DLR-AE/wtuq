"""
@author: Johannes Rieke <jrieke@nordex-online.com>, Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 18.03.2022

pre-processor of structural blade data for uncertainty quantification framework

Some of the methods in this module (especially those in CrossSection) are based on the BECAS user manual:
https://backend.orbit.dtu.dk/ws/portalfiles/portal/7711204/ris_r_1785.pdf and on the work of Hodges.

Blasques, J.P.A.A. (2012). User's Manual for BECAS: A cross section analysis tool for anisotropic and inhomogeneous beam
sections of arbitrary geometry. Risoe DTU - National Laboratory for Sustainable Energy. Denmark. Forskningceter Risoe.
Risoe-R No. 1785(EN)

Hodges, D.H. (2006). Nonlinear Composite Beam Theory (F.K. Lu, Ed.). AIAA.
"""

# global libs
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import deepcopy
import logging

# local libs
from libCrossSection import CrossSectionData


class PreProcessor:
    """
    Establish structural blade description and allow modification of its properties by means of spline objects along
    the arclength of the blade for the uncertain structural parameters.

    Parameters
    ----------
    preprocessor_config : dict
        dict with preprocessor specific information (paths to ref data)
            -> BECAS_directory                      directory of reference BECAS data
            -> h2_becas_geometry_path               file_name of HAWC2/BECAS geometry data
            -> focus_geometry_path                  file_name of FOCUS geometry data
            -> template_h2_beam_properties_path     file_name of template HAWC2 beam properties data
            -> modification_method                  dictionary defining how each physical parameter has to be modified
                                                    (multiplicative or additive)

    Attributes
    ----------
    reference_HAWC_data : dict
        reference h2 beam properties
    h2_becas_geom : dict
        blade geometry description
    arc : array
        vector with arclength positions of the blade where cross-secional data is defined
    focus_geom : dict
        geometry information
    modification_method : dict
        description how each uncertain parameter has to be varied -> 'additive' or 'multiplicative'
    blade : BladeCrossSections
        structural blade representation with multiple cross-sections in BECAS reference system
    """
    def __init__(self, preprocessor_config):
        self.reference_HAWC_data = self.read_template_data(preprocessor_config['template_h2_beam_properties_path'])
        self.h2_becas_geom = self.parse_h2_becas_geom(preprocessor_config['h2_becas_geometry_path'])
        self.arc = self.reference_HAWC_data['arc']
        self.focus_geom = self.parse_and_interp_focus_geom(preprocessor_config['focus_geometry_path'], self.arc)
        self.modification_method = preprocessor_config['modification_method']

        # build reference IWT blade
        self.blade = BladeCrossSections(BECAS_directory=preprocessor_config['BECAS_directory'],
                                        arc=self.arc, nr_of_sections=len(self.arc))

        # apply modifications for QuexUS reference blade (30% GJ, 30% EIflap, 70% EIedge)
        for sect in range(len(self.blade.css)):
            self.blade.css[sect].modify_torsion_s6x6(0.3, method='multiplicative')
            self.blade.css[sect].modify_EIPA_flap_s6x6(0.3, method='multiplicative')
            self.blade.css[sect].modify_EIPA_edge_s6x6(0.7, method='multiplicative')

    def apply_structural_modifications(self, interpolation_objects):
        """
        Apply modifications on BladeCrossSections instance

        Parameters
        ----------
        interpolation_objects : dict
            Dictionary with a radial interpolation object for each uncertain parameter
        """
        for uncertain_parameter in interpolation_objects:
            self.modify_structural_model(uncertain_parameter,
                                         interpolation_objects[uncertain_parameter],
                                         self.modification_method[uncertain_parameter])

    @staticmethod
    def read_template_data(reference_h2_st):
        """
        Read h2 .st structural blade description

        Parameters
        ----------
        reference_h2_st : string
            Path to HAWC2 .st file
        """
        # read full file
        try:
            with open(reference_h2_st, 'r') as f:
                full_file = f.readlines()
        except OSError:
            print(reference_h2_st, 'could not be found')
            raise

        # find end of header -> $1
        anchor_header = '$1'
        end_header = [full_file.index(s) for s in full_file if anchor_header in s]

        # load data array
        reference_h2_data = np.loadtxt(full_file[end_header[0]+1:])
        blade_data = dict()
        blade_data['arc'] = reference_h2_data[:, 0]
        blade_data['m_pm'] = reference_h2_data[:, 1]
        blade_data['x_cg'] = reference_h2_data[:, 2]
        blade_data['y_cg'] = reference_h2_data[:, 3]
        blade_data['ri_x'] = reference_h2_data[:, 4]
        blade_data['ri_y'] = reference_h2_data[:, 5]
        blade_data['x_sc'] = reference_h2_data[:, 6]
        blade_data['y_sc'] = reference_h2_data[:, 7]
        blade_data['E'] = reference_h2_data[:, 8]
        blade_data['G'] = reference_h2_data[:, 9]
        blade_data['I_x'] = reference_h2_data[:, 10]
        blade_data['I_y'] = reference_h2_data[:, 11]
        blade_data['I_T'] = reference_h2_data[:, 12]
        blade_data['k_x'] = reference_h2_data[:, 13]
        blade_data['k_y'] = reference_h2_data[:, 14]
        blade_data['A'] = reference_h2_data[:, 15]
        blade_data['theta_s'] = reference_h2_data[:, 16]
        blade_data['x_ec'] = reference_h2_data[:, 17]
        blade_data['y_ec'] = reference_h2_data[:, 18]
        return blade_data

    @staticmethod
    def parse_h2_becas_geom(h2_becas_geom_luh):
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
        h2_becas_geom = dict()
        h2_becas_geom['radius'] = table_geo_h2[:, 0]
        h2_becas_geom['chord'] = table_geo_h2[:, 1]
        h2_becas_geom['twist'] = table_geo_h2[:, 3]
        return h2_becas_geom

    @staticmethod
    def parse_and_interp_focus_geom(focus_geom_csv, arc):
        """
        Read focus geometry file and linearly interpolate to 'arc' positions

        Parameters
        ----------
        focus_geom_csv : string
            Path to FOCUS geometry file
        arc : array
            Array with blade positions along the curved blade

        Returns
        -------
        focus_geom_83s : array
            Geometry data for the 83 radial sections defined in the FOCUS geometry file
        """
        focus_geom = np.loadtxt(focus_geom_csv, delimiter=',', skiprows=1)
        f = interp1d(focus_geom[:, 1], focus_geom, axis=0, fill_value='extrapolate')
        focus_geom_83s = f(arc)
        return focus_geom_83s

    def apply_spline(self, spline, arc_pos):
        """
        Get spline coefficient for a given radial (arc) position

        Parameters
        ----------
        spline : {LinearInterp or QuexusSpline}
            Object describing spline/interpolation along the blade
        arc_pos : float
            Radial position where spline value should be given

        Returns
        -------
        spline.as_array([eval_pnt]) : float
            Spline value at (arc) position
        """
        # map arc to [0, 1]
        radial_pnt = arc_pos / self.arc[-1]
        eval_pnt = radial_pnt
        return spline.as_array([eval_pnt])

    def get_local_chord(self, arc_pos):
        """
        Get local chord by linear interpolation for a given radial (arc) position

        Parameters
        ----------
        arc_pos : float
            Radial position where spline value should be given

        Returns
        -------
        f(arc_pos) : float
            Chord value determined from h2_becas_geom at (arc) position
        """
        f = interp1d(self.h2_becas_geom['radius'], self.h2_becas_geom['chord'], fill_value='extrapolate')
        return f(arc_pos)

    def modify_structural_model(self, uncertain_parameter, spline_object, method):
        """
        Gets uncertain parameter and corresponding spline object, modifies structural cross section matrices for each
        blade section.

        Parameters
        ----------
        uncertain_parameter : string
            Physical parameter that has to be modified (f.ex. mass_distribution)
        spline_object : {LinearInterp or QuexusSpline}
            Object describing spline/interpolation of uncertain parameter along the blade
        method : string
            Description how each uncertain parameter has to be varied -> 'additive' or 'multiplicative'
        """
        logger = logging.getLogger('quexus.uq.preprocessor.modify_structural_model')
        for sect in range(len(self.blade.css)):
            css = self.blade.css[sect]
            if uncertain_parameter == 'mass_distribution':
                css.modify_mass_distribution_m6x6(self.apply_spline(spline_object, css.arclength), method=method)

            elif uncertain_parameter == 'flap_stiffness':
                css.modify_EIPA_flap_s6x6(self.apply_spline(spline_object, css.arclength), method=method)

            elif uncertain_parameter == 'edge_stiffness':
                css.modify_EIPA_edge_s6x6(self.apply_spline(spline_object, css.arclength), method=method)

            elif uncertain_parameter == 'torsion_stiffness':
                css.modify_torsion_s6x6(self.apply_spline(spline_object, css.arclength), method=method)

            elif uncertain_parameter == 'shear_center_x':
                if method == 'multiplicative':
                    # This spline application is chosen very unlucky... Now the input changes sign, so when the
                    # spline is >1 the delta actually moves in negative direction...
                    delta_sc_x = (1-self.apply_spline(spline_object, css.arclength)[0]) * \
                                 self.get_local_chord(css.arclength)
                elif method == 'additive':
                    delta_sc_x = self.apply_spline(spline_object, css.arclength)[0]
                delta_sc = [delta_sc_x, 0]
                css.modify_sc(delta_sc)

            elif uncertain_parameter == 'shear_center_y':
                if method == 'multiplicative':
                    # This spline application is chosen very unlucky... Now the input changes sign, so when the
                    # spline is >1 the delta actually moves in negative direction...
                    delta_sc_y = (1-self.apply_spline(spline_object, css.arclength)[0]) * \
                                 self.get_local_chord(css.arclength)
                elif method == 'additive':
                    delta_sc_y = self.apply_spline(spline_object, css.arclength)[0]
                delta_sc = [0, delta_sc_y]
                css.modify_sc(delta_sc)

            elif uncertain_parameter == 'cog_x':
                if method == 'multiplicative':
                    # This spline application is chosen very unlucky... Now the input changes sign, so when the
                    # spline is >1 the delta actually moves in negative direction...
                    delta_cog_x = (1-self.apply_spline(spline_object, css.arclength)[0]) * \
                                  self.get_local_chord(css.arclength)
                elif method == 'additive':
                    delta_cog_x = self.apply_spline(spline_object, css.arclength)[0]
                delta_cog = [delta_cog_x, 0]
                css.modify_cog(delta_cog)

            elif uncertain_parameter == 'cog_y':
                if method == 'multiplicative':
                    # This spline application is chosen very unlucky... Now the input changes sign, so when the
                    # spline is >1 the delta actually moves in negative direction...
                    delta_cog_y = (1-self.apply_spline(spline_object, css.arclength)[0]) * \
                                  self.get_local_chord(css.arclength)
                elif method == 'additive':
                    delta_cog_y = self.apply_spline(spline_object, css.arclength)[0]
                delta_cog = [0, delta_cog_y]
                css.modify_cog(delta_cog)

            elif uncertain_parameter == 'principle_axis_orientation':
                css.modify_principle_axis(self.apply_spline(spline_object, css.arclength)[0], method=method)

            elif uncertain_parameter == 'elastic_center':
                pass

    def export_data(self):
        """
        Wrapping method to export blade data as a dictionary with transformed blade data in different formats

        Returns
        -------
        blade_data : dict
            Dictionary with structural blade data in h2_beam, h2_FPM, alaska_6x6, beamdyn_6x6,
            and alaska_6x6_element_ref format
        """
        blade_data = dict()
        blade_data['h2_beam'] = self.blade.export2hawc2beam(self.reference_HAWC_data)
        blade_data['h2_FPM'] = self.blade.export2hawc2FPM()
        blade_data['alaska_6x6'] = self.blade.export2alaska6x6(self.focus_geom)
        blade_data['beamdyn_6x6'] = self.blade.export2beamdyn(self.focus_geom)
        blade_data['alaska_6x6_element_ref'] = self.blade.export2alaska6x6elemref(self.focus_geom)
        return blade_data

    def plot_distribution(self, run_directory, plot_bool):
        """
        Plot all engineering beam properties along the blade, save plots to run_directory

        Parameters
        ----------
        run_directory : string
            Path to directory where figure will be saved
        plot_bool : bool
            Plotting flag
        """
        if plot_bool is True:
            blade_data = self.export_data()
            for param in self.reference_HAWC_data:
                fig = plt.figure(run_directory+param)
                if not fig.axes:
                    # only add the axes one time (MatplotlibDepreciationWarning -> in the future axes will be
                    # overwritten if a new axis is opened while an older one exists with the same name
                    ax1 = fig.add_subplot(311)
                    ax2 = fig.add_subplot(312)
                    ax3 = fig.add_subplot(313)
                else:
                    ax1 = fig.axes[0]
                    ax2 = fig.axes[1]
                    ax3 = fig.axes[2]

                ax1.plot(self.reference_HAWC_data['arc'],
                         self.reference_HAWC_data[param],
                         label='reference hawc2 input file')
                ax1.plot(blade_data['h2_beam']['arc'],
                         blade_data['h2_beam'][param],
                         '--', label='BECAS2HAWC2 conversion modified model')
                ax1.set_title(param)
                #ax1.legend()
                ax1.grid()

                abs_diff = self.reference_HAWC_data[param] - blade_data['h2_beam'][param]
                ax2.plot(self.reference_HAWC_data['arc'], abs_diff)
                ax2.set_xlabel('arclength [m]')
                ax2.set_ylabel('Absolute difference')
                ax2.grid()

                # relative difference of the positions are measure in % chord
                if param in ['x_sc', 'y_sc', 'x_ec', 'y_ec', 'x_cg', 'y_cg']:
                    rel_diff = (self.reference_HAWC_data[param] - blade_data['h2_beam'][param]) / \
                               self.get_local_chord(self.reference_HAWC_data['arc']) * 100
                    ax3.plot(self.reference_HAWC_data['arc'], rel_diff)
                    ax3.set_xlabel('arclength [m]')
                    ax3.set_ylabel('Absolute difference [% local chord]')
                    ax3.grid()
                    plt.savefig(os.path.join(run_directory, 'uq_results', str(param) + '.pdf'))
                # all other values are relative to themselves
                else:
                    rel_diff = (self.reference_HAWC_data[param] - blade_data['h2_beam'][param]) / \
                               self.reference_HAWC_data[param] * 100
                    ax3.plot(self.reference_HAWC_data['arc'], rel_diff)
                    ax3.set_xlabel('arclength [m]')
                    ax3.set_ylabel('Relative difference [%]')
                    ax3.grid()
                    plt.savefig(os.path.join(run_directory, 'uq_results', str(param) + '.pdf'))


class BladeCrossSections:
    """
    Structural blade description consisting of multiple cross sections along the span.

    Parameters
    ----------
    BECAS_directory : string
        path to directory with reference BECAS data
    arc : list
        list with nodal positions along the blade arc
    nr_of_sections : int
        number of structural blade sections

    Attributes
    ----------
    BECAS_directory : string
        path to directory with reference BECAS data
    arc : list
        list with nodal positions along the blade arc
    nr_of_sections : int
        number of structural blade sections
    css : list
        list filled with CrossSection objects
    """
    def __init__(self, BECAS_directory, arc, nr_of_sections):
        self.BECAS_directory = BECAS_directory
        self.arc = arc
        self.nr_of_sections = nr_of_sections
        self.css = []
        self._build_blade()

    def _build_blade(self):
        """ Read input BECAS information and set up BladeCrossSections object """
        for sect in range(self.nr_of_sections):
            id = '{:03d}'.format(sect+1)
            filename = os.path.join(self.BECAS_directory, 'BL_'+id+'_BCS', 'BECAS_2D.out')
            structural_cs = CrossSection(id=id, arclength=self.arc[sect])
            structural_cs.get_input_matrices(filename)

            self.css.append(structural_cs)

    def export2hawc2beam(self, reference_template_data=None):
        """
        Transform BECAS properties to hawc2 engineering beam properties

        Parameters
        ----------
        reference_template_data : string, optional
            path to template hawc2 beam data

        Returns
        -------
        blade_data : dict
            Structural blade description in HAWC2 beam properties format
        """
        if reference_template_data == None:
            blade_data = dict()
            blade_data['arc'] = self.arc
            blade_data['m_pm'] = np.zeros(len(self.arc))
            blade_data['x_cg'] = np.zeros(len(self.arc))
            blade_data['y_cg'] = np.zeros(len(self.arc))
            blade_data['ri_x'] = np.zeros(len(self.arc))
            blade_data['ri_y'] = np.zeros(len(self.arc))
            blade_data['x_sc'] = np.zeros(len(self.arc))
            blade_data['y_sc'] = np.zeros(len(self.arc))
            blade_data['E'] = np.zeros(len(self.arc))
            blade_data['G'] = np.zeros(len(self.arc))
            blade_data['I_x'] = np.zeros(len(self.arc))
            blade_data['I_y'] = np.zeros(len(self.arc))
            blade_data['I_T'] = np.zeros(len(self.arc))
            blade_data['k_x'] = np.zeros(len(self.arc))
            blade_data['k_y'] = np.zeros(len(self.arc))
            blade_data['A'] = np.zeros(len(self.arc))
            blade_data['theta_s'] = np.zeros(len(self.arc))
            blade_data['x_ec'] = np.zeros(len(self.arc))
            blade_data['y_ec'] = np.zeros(len(self.arc))
        else:
            blade_data = deepcopy(reference_template_data)

        for sect in range(len(self.css)):
            
            stiffmat = self.css[sect].stiffnessMatrix
            massmat = self.css[sect].massMatrix
            cog = self.css[sect].cog()
            sc = self.css[sect].sc()
            ec = self.css[sect].ec()

            # stiffmat/massmat in ec and sc
            stiffmat_ec = self.css[sect].translateMatrix(stiffmat, ec[0], ec[1])
            stiffmat_sc = self.css[sect].translateMatrix(stiffmat, sc[0], sc[1])
            massmat_ec = self.css[sect].translateMatrix(massmat, ec[0], ec[1])
            
            # orientation of principle axis with respect to elastic center
            pa_ec = self.css[sect].calculateOrientationPrincipleAxes(stiffmat_ec)
            
            # stiffmat/massmat in ec rotated wrt pa_ec
            stiffmat_ec_pa = self.css[sect].rotateMatrix(stiffmat_ec, pa_ec)  
            massmat_ec_pa = self.css[sect].rotateMatrix(massmat_ec, pa_ec)
            
            # cross-sectional area           
            # use reference data -> can not be uniquely determined from BECAS matrices
        
            # mass per meter
            blade_data['m_pm'][sect] = massmat[0, 0]

            # cog wrt c/2
            blade_data['x_cg'][sect] = cog[0]
            blade_data['y_cg'][sect] = cog[1]

            # shear center wrt c/2
            blade_data['x_sc'][sect] = sc[0]
            blade_data['y_sc'][sect] = sc[1] 

            # elastic center wrt c/2
            blade_data['x_ec'][sect] = ec[0]
            blade_data['y_ec'][sect] = ec[1]

            # radius of inertia wrt elastic center rotated wrt principle axis         
            blade_data['ri_x'][sect] = np.sqrt(massmat_ec_pa[3, 3]/massmat[0, 0])
            blade_data['ri_y'][sect] = np.sqrt(massmat_ec_pa[4, 4]/massmat[0, 0])
                        
            # Young's modulus
            # -> could also be read from reference
            blade_data['E'][sect] = stiffmat_ec_pa[2, 2]/blade_data['A'][sect]
            
            # Shear modulus
            # -> material and geometric property which can not be derived
            # with only the cross-section matrices -> use reference value

            # Area moment of inertia wrt pa
            blade_data['I_x'][sect] = stiffmat_ec_pa[3, 3]/blade_data['E'][sect]
            blade_data['I_y'][sect] = stiffmat_ec_pa[4, 4]/blade_data['E'][sect]
            
            # Torsional stiffness            
            blade_data['I_T'][sect] = stiffmat_sc[5, 5]/blade_data['G'][sect]
            
            # Shear factor
            blade_data['k_x'][sect] = stiffmat_sc[0, 0] / (blade_data['G'][sect] * blade_data['A'][sect])
            blade_data['k_y'][sect] = stiffmat_sc[1, 1] / (blade_data['G'][sect] * blade_data['A'][sect])

            # Principle axis
            blade_data['theta_s'][sect] = pa_ec 

        return blade_data

    def export2hawc2FPM(self):
        """
        Transform BECAS cross-section data in HAWC2 FPM input data.

        Returns
        -------
        blade_data : dict
            Structural blade description in HAWC2 FPM format

        Notes
        -----
        Description of HAWC2 FPM input data given in table 2, p.26 of How2HAWC2 manual.
        Note: the cross sectional stiffness matrix is given at the elastic center
        rotated along the principle bending axes.
        """
        blade_data = dict()
        blade_data['arc'] = self.arc
        blade_data['m_pm'] = np.zeros(len(self.arc))
        blade_data['x_cg'] = np.zeros(len(self.arc))
        blade_data['y_cg'] = np.zeros(len(self.arc))
        blade_data['ri_x'] = np.zeros(len(self.arc))
        blade_data['ri_y'] = np.zeros(len(self.arc))
        blade_data['theta_s'] = np.zeros(len(self.arc))
        blade_data['x_ec'] = np.zeros(len(self.arc))
        blade_data['y_ec'] = np.zeros(len(self.arc))
        blade_data['K_11'] = np.zeros(len(self.arc))
        blade_data['K_12'] = np.zeros(len(self.arc))
        blade_data['K_13'] = np.zeros(len(self.arc))
        blade_data['K_14'] = np.zeros(len(self.arc))
        blade_data['K_15'] = np.zeros(len(self.arc))
        blade_data['K_16'] = np.zeros(len(self.arc))
        blade_data['K_22'] = np.zeros(len(self.arc))
        blade_data['K_23'] = np.zeros(len(self.arc))
        blade_data['K_24'] = np.zeros(len(self.arc))
        blade_data['K_25'] = np.zeros(len(self.arc))
        blade_data['K_26'] = np.zeros(len(self.arc))
        blade_data['K_33'] = np.zeros(len(self.arc))
        blade_data['K_34'] = np.zeros(len(self.arc))
        blade_data['K_35'] = np.zeros(len(self.arc))
        blade_data['K_36'] = np.zeros(len(self.arc))
        blade_data['K_44'] = np.zeros(len(self.arc))
        blade_data['K_45'] = np.zeros(len(self.arc))
        blade_data['K_46'] = np.zeros(len(self.arc))
        blade_data['K_55'] = np.zeros(len(self.arc))
        blade_data['K_56'] = np.zeros(len(self.arc))
        blade_data['K_66'] = np.zeros(len(self.arc))

        for sect in range(len(self.css)):
            stiffmat = self.css[sect].stiffnessMatrix
            massmat = self.css[sect].massMatrix
            cog = self.css[sect].cog()
            ec = self.css[sect].ec()

            # stiffmat/massmat in ec and sc
            stiffmat_ec = self.css[sect].translateMatrix(stiffmat, ec[0], ec[1])
            massmat_ec = self.css[sect].translateMatrix(massmat, ec[0], ec[1])

            # orientation of principle axis with respect to elastic center
            pa_ec = self.css[sect].calculateOrientationPrincipleAxes(stiffmat_ec)

            # stiffmat/massmat in ec rotated wrt pa_ec
            stiffmat_ec_pa = self.css[sect].rotateMatrix(stiffmat_ec, pa_ec)
            massmat_ec_pa = self.css[sect].rotateMatrix(massmat_ec, pa_ec)

            # mass per meter
            blade_data['m_pm'][sect] = massmat[0, 0]

            # cog wrt c/2
            blade_data['x_cg'][sect] = cog[0]
            blade_data['y_cg'][sect] = cog[1]

            # elastic center wrt c/2
            blade_data['x_ec'][sect] = ec[0]
            blade_data['y_ec'][sect] = ec[1]

            # radius of inertia wrt elastic center rotated wrt principle axis
            blade_data['ri_x'][sect] = np.sqrt(massmat_ec_pa[3, 3] / massmat[0, 0])
            blade_data['ri_y'][sect] = np.sqrt(massmat_ec_pa[4, 4] / massmat[0, 0])

            # Principle axis
            blade_data['theta_s'][sect] = pa_ec

            # stiffness matrix in ec rotated along pa
            blade_data['K_11'][sect] = stiffmat_ec_pa[0, 0]
            blade_data['K_12'][sect] = stiffmat_ec_pa[0, 1]
            blade_data['K_13'][sect] = stiffmat_ec_pa[0, 2]
            blade_data['K_14'][sect] = stiffmat_ec_pa[0, 3]
            blade_data['K_15'][sect] = stiffmat_ec_pa[0, 4]
            blade_data['K_16'][sect] = stiffmat_ec_pa[0, 5]
            blade_data['K_22'][sect] = stiffmat_ec_pa[1, 1]
            blade_data['K_23'][sect] = stiffmat_ec_pa[1, 2]
            blade_data['K_24'][sect] = stiffmat_ec_pa[1, 3]
            blade_data['K_25'][sect] = stiffmat_ec_pa[1, 4]
            blade_data['K_26'][sect] = stiffmat_ec_pa[1, 5]
            blade_data['K_33'][sect] = stiffmat_ec_pa[2, 2]
            blade_data['K_34'][sect] = stiffmat_ec_pa[2, 3]
            blade_data['K_35'][sect] = stiffmat_ec_pa[2, 4]
            blade_data['K_36'][sect] = stiffmat_ec_pa[2, 5]
            blade_data['K_44'][sect] = stiffmat_ec_pa[3, 3]
            blade_data['K_45'][sect] = stiffmat_ec_pa[3, 4]
            blade_data['K_46'][sect] = stiffmat_ec_pa[3, 5]
            blade_data['K_55'][sect] = stiffmat_ec_pa[4, 4]
            blade_data['K_56'][sect] = stiffmat_ec_pa[4, 5]
            blade_data['K_66'][sect] = stiffmat_ec_pa[5, 5]

        return blade_data

    def export2alaska6x6(self, focus_geom_83s):
        """
        Transform BECAS cross-section data in alaska 6x6 input data.

        Parameters
        ----------
        focus_geom_83s : array
            Geometry description at 83 (FOCUS) radial sections

        Returns
        -------
        blade_data : dict
            Structural blade description in alaska 6x6 format

        Notes
        -----
        Steps:
            1) move BECAS cross-section matrices to tp
            2) permutate cross-section matrices into GEBT order
        """
        blade_data = dict()
        blade_data['stiffness_matrix'] = np.empty([len(self.css), 6, 6])
        blade_data['mass_matrix'] = np.empty([len(self.css), 6, 6])

        twist_focus_83s = focus_geom_83s[:, 3]
        c2_to_TA_focus_83s = focus_geom_83s[:, 6:8] - focus_geom_83s[:, 12:14]

        for sect in range(len(self.css)):

            # 1) Move cross section matrix to tp
            angle_pitch_ref_to_chord_parallel_ref = (twist_focus_83s[sect] + 90) * np.pi / 180
            rotmat = np.array(
                [[np.cos(angle_pitch_ref_to_chord_parallel_ref), -np.sin(angle_pitch_ref_to_chord_parallel_ref)],
                 [np.sin(angle_pitch_ref_to_chord_parallel_ref), np.cos(angle_pitch_ref_to_chord_parallel_ref)]])
            c2_to_TA_chord_parallel = np.dot(rotmat, (c2_to_TA_focus_83s[sect, :].T))

            # stiffness and mass matrix with c2 as reference point
            stiffmat_c2 = self.css[sect].stiffnessMatrix
            massmat_c2 = self.css[sect].massMatrix

            # stiffness and mass matrix with tp as reference point
            stiffmat_tp = self.css[sect].translateMatrix(stiffmat_c2,
                                                         c2_to_TA_chord_parallel[0], c2_to_TA_chord_parallel[1])
            massmat_tp = self.css[sect].translateMatrix(massmat_c2,
                                                        c2_to_TA_chord_parallel[0], c2_to_TA_chord_parallel[1])

            # 2) apply permutations
            perm_vec = np.array([2, 0, 1, 5, 3, 4])
            stiffmat_export = np.empty([6, 6])
            massmat_export = np.empty([6, 6])
            for i in range(0, 6):
                for j in range(0, 6):
                    stiffmat_export[i, j] = deepcopy(stiffmat_tp[perm_vec[i], perm_vec[j]])  # use a deepcopy!
                    massmat_export[i, j] = deepcopy(massmat_tp[perm_vec[i], perm_vec[j]])  # use a deepcopy!

            # 3) add to blade_data dict
            blade_data['stiffness_matrix'][sect, :, :] = stiffmat_export
            blade_data['mass_matrix'][sect, :, :] = massmat_export

        return blade_data

    def export2alaska6x6elemref(self, focus_geom_83s):
        """
        Transform BECAS cross-section data in alaska 6x6 (rotated to element ref) input data.

        Parameters
        ----------
        focus_geom_83s : array
            Geometry description at 83 (FOCUS) radial sections

        Returns
        -------
        blade_data : dict
            Structural blade description in alaska 6x6 element reference format

        Notes
        -----
        Steps:
            1) move BECAS cross-section matrices to tp
            2) rotate matrices with twist angle
            3) permutate cross-section matrices into GEBT order
        """
        blade_data = dict()
        blade_data['stiffness_matrix'] = np.empty([len(self.css), 6, 6])
        blade_data['mass_matrix'] = np.empty([len(self.css), 6, 6])

        twist_focus_83s = focus_geom_83s[:, 3]
        c2_to_TA_focus_83s = focus_geom_83s[:, 6:8] - focus_geom_83s[:, 12:14]

        for sect in range(len(self.css)):

            # 1) Move cross section matrix to tp
            angle_pitch_ref_to_chord_parallel_ref = (twist_focus_83s[sect] + 90) * np.pi / 180
            rotmat = np.array(
                [[np.cos(angle_pitch_ref_to_chord_parallel_ref), -np.sin(angle_pitch_ref_to_chord_parallel_ref)],
                 [np.sin(angle_pitch_ref_to_chord_parallel_ref), np.cos(angle_pitch_ref_to_chord_parallel_ref)]])
            c2_to_TA_chord_parallel = np.dot(rotmat, (c2_to_TA_focus_83s[sect, :].T))

            # stiffness and mass matrix with c2 as reference point
            stiffmat_c2 = self.css[sect].stiffnessMatrix
            massmat_c2 = self.css[sect].massMatrix

            # stiffness and mass matrix with tp as reference point
            stiffmat_tp = self.css[sect].translateMatrix(stiffmat_c2,
                                                         c2_to_TA_chord_parallel[0], c2_to_TA_chord_parallel[1])
            massmat_tp = self.css[sect].translateMatrix(massmat_c2,
                                                        c2_to_TA_chord_parallel[0], c2_to_TA_chord_parallel[1])

            # 2) rotate matrices with twist angle
            stiffmat_tp_elem_ref = self.css[sect].rotateMatrix(stiffmat_tp, twist_focus_83s[sect])
            massmat_tp_elem_ref = self.css[sect].rotateMatrix(massmat_tp, twist_focus_83s[sect])

            # 3) apply permutations
            perm_vec = np.array([2, 0, 1, 5, 3, 4])
            stiffmat_export = np.empty([6, 6])
            massmat_export = np.empty([6, 6])
            for i in range(0, 6):
                for j in range(0, 6):
                    stiffmat_export[i, j] = deepcopy(stiffmat_tp_elem_ref[perm_vec[i], perm_vec[j]])  # use a deepcopy!
                    massmat_export[i, j] = deepcopy(massmat_tp_elem_ref[perm_vec[i], perm_vec[j]])  # use a deepcopy!

            # 3) add to blade_data dict
            blade_data['stiffness_matrix'][sect, :, :] = stiffmat_export
            blade_data['mass_matrix'][sect, :, :] = massmat_export

        return blade_data

    def export2beamdyn(self, focus_geom_83s):
        """
        Transform BECAS cross-section data to beamdyn 6x6 input data.

        Parameters
        ----------
        focus_geom_83s : array
            Geometry description at 83 (FOCUS) radial sections

        Returns
        -------
        blade_data : dict
            Structural blade description in BeamDyn format

        Notes
        -----
        Steps:
            1) move BECAS cross-section matrices to tp (still parallel to chord)
            2) rotate cross-section matrices around 90 deg
        """
        blade_data = dict()
        blade_data['arc'] = self.arc
        blade_data['stiffness_matrix'] = np.empty([len(self.css), 6, 6])
        blade_data['mass_matrix'] = np.empty([len(self.css), 6, 6])

        twist_focus_83s = focus_geom_83s[:, 3]
        c2_to_TA_focus_83s = focus_geom_83s[:, 6:8] - focus_geom_83s[:, 12:14]

        for sect in range(len(self.css)):

            # 1) Move cross section matrix to tp
            angle_pitch_ref_to_chord_parallel_ref = (twist_focus_83s[sect] + 90) * np.pi / 180
            rotmat = np.array(
                [[np.cos(angle_pitch_ref_to_chord_parallel_ref), -np.sin(angle_pitch_ref_to_chord_parallel_ref)],
                 [np.sin(angle_pitch_ref_to_chord_parallel_ref), np.cos(angle_pitch_ref_to_chord_parallel_ref)]])
            c2_to_TA_chord_parallel = np.dot(rotmat, (c2_to_TA_focus_83s[sect, :].T))

            # stiffness and mass matrix with c2 as reference point
            stiffmat_c2 = self.css[sect].stiffnessMatrix
            massmat_c2 = self.css[sect].massMatrix

            # stiffness and mass matrix with tp as reference point
            stiffmat_tp = self.css[sect].translateMatrix(stiffmat_c2,
                                                         c2_to_TA_chord_parallel[0], c2_to_TA_chord_parallel[1])
            massmat_tp = self.css[sect].translateMatrix(massmat_c2,
                                                        c2_to_TA_chord_parallel[0], c2_to_TA_chord_parallel[1])

            # 2) rotate cross section matrices
            stiffmat_tp_90deg = self.css[sect].rotateMatrix(stiffmat_tp, 90)
            massmat_tp_90deg = self.css[sect].rotateMatrix(massmat_tp, 90)

            # 3) add to blade_data dict
            blade_data['stiffness_matrix'][sect, :, :] = stiffmat_tp_90deg
            blade_data['mass_matrix'][sect, :, :] = massmat_tp_90deg

        return blade_data


class CrossSection(CrossSectionData):
    """
    Extended functionalities for CrossSectionData class which allows modification of the structural cross-section data
    according to specific routines
    """
    def modify_mass_distribution_m6x6(self, mass_modification, method):
        """
        Modify radial mass distribution

        Parameters
        ----------
        mass_modification : float
            Modification factor for mass
        method : string
            Description how each uncertain parameter has to be varied -> 'additive' or 'multiplicative'

        Notes
        -----
        Procedure:
            1) translate mass matrix to cog
            2) rotate into mass pa
            2) modify elements 0,0 1,1 and 2,2 of mass matrix
            3) translate back to c/2 system
        """
        # unmodified cog position
        cog_original = self.cog()

        # translate mass matrix into cog
        mass_cog = self.translateMassMatrix(cog_original)
        self.massMatrix = mass_cog

        # rotate mass matrix into pa
        pa_cog = self.calculateOrientationPrincipleAxes(mass_cog)
        mass_cog_pa = self.rotateMassMatrix(pa_cog)

        # apply mass modifications
        if method == 'multiplicative':
            mass_cog_pa[0, 0] = mass_cog_pa[0, 0] * mass_modification
            mass_cog_pa[1, 1] = mass_cog_pa[1, 1] * mass_modification
            mass_cog_pa[2, 2] = mass_cog_pa[2, 2] * mass_modification
        elif method == 'additive':
            mass_cog_pa[0, 0] = mass_cog_pa[0, 0] + mass_modification
            mass_cog_pa[1, 1] = mass_cog_pa[1, 1] + mass_modification
            mass_cog_pa[2, 2] = mass_cog_pa[2, 2] + mass_modification

        self.massMatrix = mass_cog_pa
        mass_cog = self.rotateMassMatrix(-pa_cog)
        self.massMatrix = mass_cog

        # move mass matrix back to reference point
        mass_new = self.translateMassMatrix(-cog_original)
        self.massMatrix = mass_new

    def modify_torsion_s6x6(self, GIT_modification, method):
        """
        Modify torsional stiffness.

        Parameters
        ----------
        GIT_modification : float
            Modification factor for torsional stiffness
        method : string
            Description how each uncertain parameter has to be varied -> 'additive' or 'multiplicative'

        Notes
        -----
        Procedure:
            1) translate stiffness matrix to shear center
            2) apply torsional stiffness modification (on last element diagonal)
            3) translate back to c/2 system
            4) check that ec and sc positions are unmodified
        """
        # ec and sc of unmodified stiffness matrix
        ec_original = self.ec()
        sc_original = self.sc()

        # translate stiffness matrix into shear center
        stiffness_sc = self.translateStiffnessMatrix(sc_original)

        # modify torsion stiffness value (6,6) -> python [5, 5]
        if method == 'multiplicative':
            stiffness_sc[5, 5] = stiffness_sc[5, 5] * GIT_modification
        elif method == 'additive':
            stiffness_sc[5, 5] = stiffness_sc[5, 5] + GIT_modification

        # translate back
        self.stiffnessMatrix = stiffness_sc
        self.stiffnessMatrix = self.translateStiffnessMatrix(-sc_original)

        # check results
        if not np.allclose(ec_original, self.ec(), rtol=1e-05):
            print('original elastic center:', ec_original)
            print('elastic center after torsional stiffness modification:', self.ec())
            raise ValueError('unwanted modification of elastic center by torsional stiffness modification')
        if not np.allclose(sc_original, self.sc(), rtol=1e-05):
            print('original shear center:', sc_original)
            print('shear center after torsional stiffness modification:', self.sc())
            raise ValueError('unwanted modification of shear center by torsional stiffness modification')

    def modify_EIPA_flap_s6x6(self, EI_I_modification, method):
        """
        Modify flap stiffness.

        Parameters
        ----------
        EI_I_modification : float
            Modification factor for flapwise stiffness
        method : string
            Description how each uncertain parameter has to be varied -> 'additive' or 'multiplicative'

        Notes
        -----
        Procedure:
            1) translate stiffness matrix to elastic center
            2) rotate stiffness matrix to principle axes
            3) apply flap stiffness modification
            4) rotate back to chord-parallel system, translate back to c/2 system
            5) check that ec and sc positions are unmodified
        """
        # ec and sc of unmodified stiffness matrix
        ec_original = self.ec()
        sc_original = self.sc()

        # translate stiffness matrix into elastic center
        stiffness_ec = self.translateStiffnessMatrix(ec_original)
        self.stiffnessMatrix = stiffness_ec

        # rotate to principle axes
        pa_ec = self.calculateOrientationPrincipleAxes(stiffness_ec)
        stiffness_ec_pa = self.rotateStiffnessMatrix(pa_ec)

        # modify flap stiffness value (4,4) -> python [3, 3]
        if method == 'multiplicative':
            stiffness_ec_pa[3, 3] = stiffness_ec_pa[3, 3] * EI_I_modification
        elif method == 'additive':
            stiffness_ec_pa[3, 3] = stiffness_ec_pa[3, 3] + EI_I_modification
        self.stiffnessMatrix = stiffness_ec_pa

        # rotate and translate back
        self.stiffnessMatrix = self.rotateStiffnessMatrix(-pa_ec)
        self.stiffnessMatrix = self.translateStiffnessMatrix(-ec_original)

        # check results
        if not np.allclose(ec_original, self.ec(), rtol=1e-05):
            print('original elastic center:', ec_original)
            print('elastic center after EIPA_flap modification:', self.ec())
            raise ValueError('unwanted modification of elastic center by flap stiffness modification')
        if not np.allclose(sc_original, self.sc(), rtol=1e-05):
            print('original shear center:', sc_original)
            print('shear center after EIPA_flap modification:', self.sc())
            raise ValueError('unwanted modification of shear center by flap stiffness modification')

    def modify_EIPA_edge_s6x6(self, EI_II_modification, method):
        """
        Modify edge stiffness.

        Parameters
        ----------
        EI_II_modification : float
            Modification factor for edgewise stiffness
        method : string
            Description how each uncertain parameter has to be varied -> 'additive' or 'multiplicative'

        Notes
        -----
        Procedure:
            1) translate stiffness matrix to elastic center
            2) rotate stiffness matrix to principle axes
            3) apply edge stiffness modification
            4) rotate back to chord-parallel system, translate back to c/2 system
            5) check that ec and sc positions are unmodified
        """
        # ec and sc of unmodified stiffness matrix
        ec_original = self.ec()
        sc_original = self.sc()

        # translate stiffness matrix into elastic center
        stiffness_ec = self.translateStiffnessMatrix(ec_original)
        self.stiffnessMatrix = stiffness_ec

        # rotate to principle axes
        pa_ec = self.calculateOrientationPrincipleAxes(stiffness_ec)
        stiffness_ec_pa = self.rotateStiffnessMatrix(pa_ec)

        # modify flap stiffness value (5,5) -> python [4, 4]
        if method == 'multiplicative':
            stiffness_ec_pa[4, 4] = stiffness_ec_pa[4, 4] * EI_II_modification
        elif method == 'additive':
            stiffness_ec_pa[4, 4] = stiffness_ec_pa[4, 4] + EI_II_modification

        # rotate and translate back
        self.stiffnessMatrix = stiffness_ec_pa
        self.stiffnessMatrix = self.rotateStiffnessMatrix(-pa_ec)
        self.stiffnessMatrix = self.translateStiffnessMatrix(-ec_original)

        # check results
        if not np.allclose(ec_original, self.ec(), rtol=1e-05):
            print('original elastic center:', ec_original)
            print('elastic center after EIPA_edge modification:', self.ec())
            raise ValueError('unwanted modification of elastic center by edge stiffness modification')
        if not np.allclose(sc_original, self.sc(), rtol=1e-05):
            print('original shear center:', sc_original)
            print('shear center after EIPA_edge modification:', self.sc())
            raise ValueError('unwanted modification of shear center by edge stiffness modification')

    def modify_principle_axis(self, delta_pa, method):
        """
        Modify principle axis orientation.

        Parameters
        ----------
        delta_pa : float
            Modification factor for principle axis position
        method : string
            Description how each uncertain parameter has to be varied -> 'additive' or 'multiplicative'

        Notes
        -----
        Procedure:
            1) translate stiffness matrix to elastic center
            2) rotate stiffness matrix to principle axes and rotate stiffness matrix to new principle axes
            3) use elements [3:5, 3:5] of new stiffness matrix
            4) rotate back to chord-parallel system, translate back to c/2 system
            5) check that ec and sc positions are unmodified
        """
        # ec and sc of unmodified stiffness matrix
        ec_original = self.ec()
        sc_original = self.sc()

        # translate stiffness matrix into elastic center
        stiffness_ec = self.translateStiffnessMatrix(ec_original)
        self.stiffnessMatrix = stiffness_ec

        # rotate to principle axes
        pa_ec = self.calculateOrientationPrincipleAxes(stiffness_ec)

        if method == 'multiplicative':
            # pa_ec_new = pa_ec * (2 - delta_pa)
            pa_ec_new = pa_ec * delta_pa
        elif method == 'additive':
            pa_ec_new = pa_ec - delta_pa

        stiffness_ec_pa = self.rotateStiffnessMatrix(pa_ec)
        stiffness_ec_pa_new = self.rotateStiffnessMatrix(pa_ec_new)

        stiffness_ec_pa[3:5, 3:5] = stiffness_ec_pa_new[3:5, 3:5]

        # rotate back
        self.stiffnessMatrix = stiffness_ec_pa
        self.stiffnessMatrix = self.rotateStiffnessMatrix(-pa_ec)

        # translate back
        self.stiffnessMatrix = self.translateStiffnessMatrix(-ec_original)

        # check results
        if not np.allclose(ec_original, self.ec(), rtol=1e-05):
            print('original elastic center:', ec_original)
            print('elastic center after principle axis modification:', self.ec())
            raise ValueError('unwanted modification of elastic center by principle axis modification')
        if not np.allclose(sc_original, self.sc(), rtol=1e-05):
            print('original shear center:', sc_original)
            print('shear center after principle modification:', self.sc())
            raise ValueError('unwanted modification of shear center by principle axis modification')

    def modify_cog(self, delta_cog):
        """
        Modify cog position in cross-section

        Parameters
        ----------
        delta_cog : float
            Modification delta for cog position

        Notes
        -----
        Procedure:
            1) move mass matrix to cog
            2) move mass matrix back to cross-section reference point (but with new cog displacement)
        """
        # unmodified cog position
        cog_original = self.cog()
        cog_new = cog_original + delta_cog

        # translate mass matrix into cog
        mass_cog = self.translateMassMatrix(cog_original)
        self.massMatrix = mass_cog

        # move mass matrix back to reference point
        mass_ref_new = self.translateMassMatrix(-cog_new)
        self.massMatrix = mass_ref_new

        # check results
        if not np.allclose(cog_new, self.cog(), rtol=1e-05):
            print('requested new cog:', cog_new)
            print('new cog determined from mass matrix:', self.cog())
            raise ValueError('mass matrix modification for cog displacement unsuccessful')

    def modify_sc(self, delta_sc):
        """
        Modify sc position in cross-section

        Parameters
        ----------
        delta_sc : float
            Modification delta for sc position

        Notes
        -----
        Procedure:
            1) move stiffness matrix to sc
            2) invert stiffness matrix (complience matrix)
            3) modify elements [0, 5] and [1, 5]
            4) invert back to stiffness matrix
            5) translate back to reference
        """
        sc_original = self.sc()
        stiffness_sc_original = self.translateStiffnessMatrix(sc_original)
        complienceMatrix = np.linalg.inv(stiffness_sc_original)
        complienceMatrix[1, 5] = -delta_sc[0] * complienceMatrix[5, 5]
        complienceMatrix[0, 5] = delta_sc[1] * complienceMatrix[5, 5]
        self.stiffnessMatrix = np.linalg.inv(complienceMatrix)
        self.stiffnessMatrix = self.translateStiffnessMatrix(-sc_original)

        # check results
        if not np.allclose(sc_original + delta_sc, self.sc(), rtol=1e-05):
            print('requested new sc:', sc_original + delta_sc)
            print('new sc determined from stiffness matrix:', self.sc())
            raise ValueError('stiffness matrix modification for sc displacement unsuccessful')
