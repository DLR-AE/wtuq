"""
@author: Johannes Rieke <jrieke@nordex-online.com>, Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 11.02.2020

A library module, which allows interaction with structural cross-section data. The data follows the BECAS convention,
but an extension to VABS or ANBA2 output parsing could easily be made. Information on the BECAS format and on beam
cross-sectional analysis can be found in the following:

BECAS user manual:
https://backend.orbit.dtu.dk/ws/portalfiles/portal/7711204/ris_r_1785.pdf

Blasques, J.P.A.A. (2012). User's Manual for BECAS: A cross section analysis tool for anisotropic and inhomogeneous beam
sections of arbitrary geometry. Risoe DTU - National Laboratory for Sustainable Energy. Denmark. Forskningceter Risoe.
Risoe-R No. 1785(EN)

Hodges, D.H. (2006). Nonlinear Composite Beam Theory (F.K. Lu, Ed.). AIAA.
"""

import numpy as np


class CrossSectionData(object):
    """
    Data class for structural cross-section data (in BECAS reference system).

    This class reads and parses BECAS result data (stiffness and mass matrix) and allows
    to derive:
    - center of gravity, 
    - principle axis angle,
    - elastic center,
    - shear center.
    Further rotation and translation operations are implemented.\n

    BECAS uses its own coordinate system, which is given below::
    
        BECAS notation:
        - stiffness matrix: force-strain relation
            |F_x|    |S_11   ...      S_16| | gamma_x |             y (flap,SS) ^   7 z (longitudinal, tip)
            |F_y|    |                    | | gamma_y |                         |  /
            |F_z| =  | :      S_33      : | | eps_z   |                         | /
            |M_x|    |                    | | kappa_x |                         |/
            |M_y|    |                    | | kappa_y |     x (edge,LE)  <------o  
            |M_z|    |S_61   ...      S_66| | psi_z   |   
            
        - mass matrix: momentum-velocity relation
            |P_x|    |m_11   0    0    0    0   m_16| | v_x     | 
            |P_y|    | 0   m_22   0    0    0   m_26| | v_y     | 
            |P_z| =  | 0    0   m_33 m_34  m_35   0 | | v_z     | 
            |H_x|    | 0    0   m_43 m_44  m_45   0 | | omega_x | 
            |H_y|    | 0    0   m_53 m_54  m_55   0 | | omega_y | 
            |H_z|    |m_61 m_62  0    0     0   m_66| | omega_z | 

    Parameters
    ----------
    id : int
        number of section
    arclength : float
        arclength position of section in m from blade flange along pitch axis
    stiffnessMatrix : ndarray
        the 6x6 stiffness matrix in BECAS notation (6,6)
    massMatrix : ndarray
        the 6x6 mass matrix in BECAS notation (6,6)

    Attributes
    ----------
    id : int
        number of section
    arclength : float
        arclength position of section in m from blade flange along pitch axis
    stiffnessMatrix : ndarray
        the 6x6 stiffness matrix in BECAS notation (6,6)
    massMatrix : ndarray
        the 6x6 mass matrix in BECAS notation (6,6)
    """

    def __init__(self, id=None, arclength=None, stiffnessMatrix=None, massMatrix=None):
        super(CrossSectionData, self).__init__()
        if id is None:
            self.id = 0
        else:
            self.id = id
        if arclength is None:
            self.arclength = 0.0
        else:
            self.arclength = arclength
        # define matrices
        if stiffnessMatrix is None:
            self.stiffnessMatrix = np.empty([6, 6])
        else:
            self.stiffnessMatrix = stiffnessMatrix
        if massMatrix is None:
            self.massMatrix = np.empty([6, 6])
        else:
            self.massMatrix = massMatrix

    def rotateStiffnessMatrix(self, angle):
        """
        Performs a rotation of a stiffness matrix by given angle
        in degree in mathematical positive direction in BECAS coordinate system.

        Parameters
        ----------
        angle : float
             angle for rotation in degree, mathematical sign convention.

        Returns
        -------
        rotMatrix : ndarray
            rotated 6x6 stiffness matrix (6,6)
        """
        matrix = self.stiffnessMatrix
        rotMatrix = self.rotateMatrix(matrix, angle)
        return rotMatrix

    def rotateMassMatrix(self, angle):
        """ 
        Performs a rotation of a mass matrix by given angle 
        in degree in mathematical positive direction in BECAS coordinate system.

        Parameters
        ----------
        angle : float
             angle for rotation in degree, mathematical sign convention.

        Returns
        -------
        rotMatrix : ndarray
            rotated 6x6 mass matrix (6,6)
        """
        matrix = self.massMatrix
        rotMatrix = self.rotateMatrix(matrix, angle)
        return rotMatrix

    def rotateMatrix(self, matrix, angle):
        """
        rotates 6x6 stiffness/mass matrix in BECAS notation by angle in degree

        Parameters
        ----------
        angle : float
             angle for rotation in degree, mathematical sign convention.
        matrix : ndarray
            the 6x6 (stiffness/mass) matrix in BECAS notation (6,6)

        Returns
        -------
        rotMatrix : ndarray
            rotated matrix (6,6)
        """
        # rotation matrix
        sa = np.sin(np.deg2rad(angle))
        ca = np.cos(np.deg2rad(angle))
        R = np.eye(6, dtype=float)
        R[0, 0] = ca
        R[0, 1] = sa
        R[1, 0] = -sa
        R[1, 1] = ca
        R[3, 3] = ca
        R[3, 4] = sa
        R[4, 3] = -sa
        R[4, 4] = ca

        # rotate
        rotMatrix = np.matmul(R, np.matmul(matrix, np.transpose(R)))
        return rotMatrix

    def rotateVector(self, vector, angle):
        """
        rotates a vector of length 2 by angle in degree

        Parameters
        ----------
        vector : ndarray
             2D vector (2,)
        angle : float
             angle for rotation in degree, mathematical sign convention.

        Returns
        -------
        rotVector : ndarray
            rotated vector (2,)
        """
        # rotation matrix
        sa = np.sin(np.deg2rad(angle))
        ca = np.cos(np.deg2rad(angle))
        R = np.eye(2, dtype=float)
        R[0, 0] = ca
        R[0, 1] = sa
        R[1, 0] = -sa
        R[1, 1] = ca

        # rotate
        rotVector = np.matmul(R, vector)
        return rotVector

    def rotateCrossSection(self, angle):
        """
        rotates stiffness and mass matrix as well as COA

        Parameters
        ----------
        angle : float
             angle for rotation in degree, mathematical sign convention.

        Returns
        -------
        StiffnessMatrix : ndarray
            rotated stiffness matrix (6,6)
        MassMatrix : ndarray
            rotated mass matrix (6,6)
        COA : ndarray
            rotated COA vector (2,)
        """
        StiffnessMatrix = self.rotateStiffnessMatrix(angle)
        MassMatrix = self.rotateMassMatrix(angle)
        COA = self.rotateVector(self.COA, angle)

        return [StiffnessMatrix, MassMatrix, COA]

    def translateStiffnessMatrix(self, pxy):
        """
        Stiffness matrix wrapper for translateMatrix
        
        Parameters
        ----------
        pxy : ndarray
             translation vector (2,)

        Returns
        -------
        transMatrix : ndarray
            translated stiffness matrix (6,6)
        """
        matrix = self.stiffnessMatrix
        px = pxy[0]
        py = pxy[1]
        transMatrix = self.translateMatrix(matrix, px, py)
        return transMatrix

    def translateMassMatrix(self, pxy):
        """
        Mass matrix wrapper for translateMatrix

        Parameters
        ----------
        pxy : ndarray
             translation vector (2,)

        Returns
        -------
        transMatrix : ndarray
            translated mass matrix (6,6)
        """
        matrix = self.massMatrix
        px = pxy[0]
        py = pxy[1]
        transMatrix = self.translateMatrix(matrix, px, py)
        return transMatrix

    def translateMatrix(self, matrix, px, py):
        """
        translates 6x6 stiffness/mass matrix in BECAS notation by :math:`\mathbf{p}=(p_x,p_y)^T`

        Parameters
        ----------
        matrix : ndarray
             6x6 matrix (6,6)
        px : float
             translation in x direction in m
        py : float
             translation in y direction in m

        Returns
        -------
        transMatrix : ndarray
            6x6 matrix translated to input offset (6,6)
        """

        # translation matrix
        T = np.eye(6, dtype=float)
        T[3, 2] = -py
        T[4, 2] = px
        T[5, 0] = py
        T[5, 1] = -px

        # translate
        transMatrix = np.matmul(T, np.matmul(matrix, np.transpose(T)))
        return transMatrix

    def pa(self):
        """
        Wrapper for principle axis calculation of stiffness matrix

        Returns
        -------
        pa : float
            principle axis angle stiffness matrix in degree
        """
        pa = self.calculateOrientationPrincipleAxes(self.stiffnessMatrix)
        return pa

    def pa_mass(self):
        """
        Wrapper for principle axis calculation of mass matrix

        Returns
        -------
        pa : float
            principle axis angle mass matrix in degree
        """
        pa_mass = self.calculateOrientationPrincipleAxes(self.massMatrix)
        return pa_mass

    def calculateOrientationPrincipleAxes(self, matrix):
        """
        Calculates principle axis angle in degree from stiffness/mass matrix in BECAS notation.

        .. math::

            \\alpha_{pa} = 0.5 * atan\\left( \\frac{2.0 * S_{45}}{S_{55} - S_{44}} \\right) \\textrm{| }{^\\circ} \n

        Parameters
        ----------
        matrix : ndarray
             6x6 matrix (6,6)

        Returns
        -------
        pa : float
            principle axis angle in degree

        """
        pa = -np.rad2deg(0.5 * np.arctan2((2.0 * matrix[3, 4]), (matrix[4, 4] - matrix[3, 3])))
        return pa

    def sc(self):
        """
        Calculates section shear center from stiffness/complience matrix in BECAS notation. \n
            - first index is chordwise               --> sc[0]\n
            - second index is perpendicular to chord --> sc[1].\n
            Consider the orientation of the matrix carefully!

        .. math::
        
            sc_x &= -C_{26}/C_{66}   &\\textrm{| m}\n
            sc_y &=  C_{16}/C_{66}   &\\textrm{| m}\n 

        Returns
        -------
        sc : ndarray
            shear center position in m (2,)
        """
        complienceMatrix = np.linalg.inv(self.stiffnessMatrix)
        sc = np.empty([2])
        sc[0] = -complienceMatrix[1, 5] / complienceMatrix[5, 5]
        sc[1] = complienceMatrix[0, 5] / complienceMatrix[5, 5]
        return sc

    def ec(self):
        """
        Calculates section elastic center from stiffness matrix in BECAS notation. \n
            - first index is chordwise               --> ec[0]\n
            - second index is perpendicular to chord --> ec[1].\n
            Consider the orientation of the matrix carefully!
        
        .. math::
            ec_x &= -S_{35}/S_{33}  &\\textrm{| m} \n
            ec_y &=  S_{34}/S_{33}  &\\textrm{| m} \n

        Returns
        -------
        ec : ndarray
            elastic center position in m (2,)
        """
        ec = np.empty([2])
        ec[0] = -self.stiffnessMatrix[2, 4] / self.stiffnessMatrix[2, 2]
        ec[1] = self.stiffnessMatrix[2, 3] / self.stiffnessMatrix[2, 2]
        return ec

    def cog(self):
        """
        Calculates section COG from mass matrix in BECAS notation. \n
            - first index is chordwise               --> cog[0]\n
            - second index is perpendicular to chord --> cog[1].\n
            Consider the orientation of the matrix carefully! 
        
        .. math::
        
            cog_x &=  m_{26}/m_{11}   &\\rightarrow  m_{26}= m_{11} \\cdot cog_x \\qquad &\\textrm{| kg/m*m} \n
            cog_y &= -m_{16}/m_{11}   &\\rightarrow  m_{16}=-m_{11} \\cdot cog_y \\qquad &\\textrm{| kg/m*m} \n

        Returns
        -------
        cog : ndarray
            cog position in m (2,)
        """
        cog = np.empty([2])
        cog[0] = self.massMatrix[1, 5] / self.massMatrix[0, 0]
        cog[1] = -self.massMatrix[0, 5] / self.massMatrix[0, 0]
        return cog

    def calc_J1_J2(self):
        """
        calculates the principle inertia from the mass matrix

        Returns
        -------
        [J1, J2] : list
            Principle inertia components (2,)
        """
        pa = self.pa_mass()
        rot_mass_matrix = self.rotateMassMatrix(pa)
        J1 = rot_mass_matrix[3, 3]
        J2 = rot_mass_matrix[4, 4]
        return [J1, J2]

    def get_input_matrices(self, fileName):
        """
        Wrapper for reading and parsing

        Parameters
        ----------
        fileName : string
            path to file with structural input matrices input
        """
        fileString = self.read_input_matrices(fileName)
        self.parse_input_matrices(fileString)

    def read_input_matrices(self, fileName):
        """
        read the structural input matrices file (BECAS notation)

        Parameters
        ----------
        fileName : string
            path to file with structural input matrices input

        Returns
        -------
        filestring : string
            fileName content

        Raises
        ------
        OSError
            If fileName could not be found
        """
        try:
            with open(fileName, 'r') as f:
                filestring = f.read()
        except OSError:
            print('File %s not found! Abort!' % fileName)
            raise
        return filestring

    def set_id(self, id):
        """ set the id of the section """
        self.id = id

    def set_pos(self, pos):
        """ set the length position of the section """
        self.pos = pos

    def parse_input_matrices(self, fileString):
        """
        Parse the structural input matrices string (BECAS notation) and give stiffness and mass matrices back

        input : content of input file in BECAS notation as string
        output: stiffness and mass matrices

        Parameters
        ----------
        fileString : string
            String, which contains the BECAS output data from standard BECAS output file.

        Notes
        -----
            *** Example of BECAS output ***
        
            Stiffness matrix w.r.t. the cross section reference point 
            K=
             1.003835216768e+10   2.157518904150e+07   8.420520895218e+02   1.872404669550e+04  -7.179458823479e+02   1.903553635467e+07
             2.157518904147e+07   9.548085563946e+09  -1.659069928154e+04  -9.704891408412e+02   2.659581057192e+04   1.591398803145e+08
             8.420521020008e+02  -1.659069925856e+04   6.021275849433e+10  -1.489874131414e+08  -6.047993574828e+07  -7.300769247882e+04
             1.872404669318e+04  -9.704891549646e+02  -1.489874131757e+08   5.921209827945e+10   5.105508411959e+05  -1.788956593429e+03
            -7.179458707102e+02   2.659581055564e+04  -6.047993579234e+07   5.105508496455e+05   5.918534377190e+10   4.070681449326e+04
             1.903553635453e+07   1.591398803149e+08  -7.300769235899e+04  -1.788956651381e+03   4.070681445447e+04   3.842788754567e+10
            
            Mass matrix w.r.t. the cross section reference point 
            M=
             2.492731715693e+03   0.000000000000e+00   0.000000000000e+00   0.000000000000e+00   0.000000000000e+00   6.107702986998e+00
             0.000000000000e+00   2.492731715693e+03   0.000000000000e+00   0.000000000000e+00   0.000000000000e+00   5.411045742066e+00
             0.000000000000e+00   0.000000000000e+00   2.492731715693e+03  -6.107702986998e+00  -5.411045742066e+00   0.000000000000e+00
            -0.000000000000e+00  -0.000000000000e+00  -6.107702986998e+00   2.444488350490e+03  -1.003694742597e+00   0.000000000000e+00
            -0.000000000000e+00  -0.000000000000e+00  -5.411045742066e+00  -1.003694742597e+00   2.464259746179e+03   0.000000000000e+00
             6.107702986998e+00   5.411045742066e+00  -0.000000000000e+00   0.000000000000e+00   0.000000000000e+00   4.908748096668e+03
        """
        # define matrices
        self.stiffnessMatrix = np.empty([6, 6])
        self.massMatrix = np.empty([6, 6])
        self.areaMoments = np.empty([3])  # in m^4
        self.COA = np.empty([2])  # in m
        self.area = 0.0  # in m^2
        self.nsm = 0.0  # in kg/m  additional non-structural masses

        # split at "K="
        tempString = fileString.split('K=')[1].splitlines()
        for i in range(0, 6):
            self.stiffnessMatrix[i, :] = np.asarray(tempString[i + 1].split(), dtype=np.float64)

        # split at "M="
        tempString = fileString.split('M=')[1].splitlines()
        for i in range(0, 6):
            self.massMatrix[i, :] = np.asarray(tempString[i + 1].split(), dtype=np.float64)

        # Get area related data, because these cannot be derivated from stiffness or mass matrix.        
        tempString = fileString.split('AreaX=')[1].splitlines()[0]
        self.COA[0] = np.float64(tempString)
        tempString = fileString.split('AreaY=')[1].splitlines()[0]
        self.COA[1] = np.float64(tempString)
        tempString = fileString.split('Area=')[1].splitlines()[0]
        self.area = np.float64(tempString)
        keyString = ['Ixx=', 'Iyy=', 'Ixy=']
        for index, key in enumerate(keyString):
            tempString = fileString.split(key)[1].splitlines()[0]
            self.areaMoments[index] = np.float64(tempString)
