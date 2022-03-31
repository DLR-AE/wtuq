import numpy as np


class HS2DataContainer(object):
    """ data container for HS2 linearization result data """
    
    def __init__(self):
        super(HS2DataContainer, self).__init__()
        
        self.names      = []
        self.frequency  = 0.0  # np array: windspeed , frequency
        self.damping    = 0.0  # np array: windspeed , damping ratio
        self.realpart   = 0.0  # np array: windspeed , real part
    

class HS2Data(object):
    """
    This is a class for operating with HAWCStab2 linearization result data

    Parameters
    ----------
    filenamecmb : string
        Path to .cmb HAWCStab2 output file
    filenameamp : string
        Path to .amp HAWCStab2 output file
    filenameop : string
        Path to .op/.opt HAWCStab2 output file
    """
    def __init__(self, filenamecmb, filenameamp, filenameop):
        super(HS2Data, self).__init__()
        
        self.filenamecmb = filenamecmb
        self.filenameamp = filenameamp
        self.filenameop  = filenameop
        self.modes       = HS2DataContainer()

    def read_cmb_data(self):
        """
        Reads and parses the HS2 result cmb data

        Raises
        ------
        OSError
            If .cmb file could not be found

        Notes
        -----
        There are three blocks for freq./damping/real part.
        """
        
        # read file
        try:
            hs2cmd = np.loadtxt(self.filenamecmb,skiprows=1,dtype='float')

            myshape=np.shape(hs2cmd)
            if np.shape(myshape)==(1,):
                num_windspeeds = 1
                num_modes      = int((myshape[0]-1)/3)
            else:
                num_windspeeds = int(myshape[0])
                num_modes      = int((myshape[1]-1)/3)
            self.modes.frequency  = np.zeros([num_modes,num_windspeeds,2])
            self.modes.damping    = np.zeros([num_modes,num_windspeeds,2])
            self.modes.realpart   = np.zeros([num_modes,num_windspeeds,2])
                
            if np.shape(myshape)==(1,):   
                for i_mode in range(0,num_modes):
                    self.modes.frequency[i_mode,:,0] = hs2cmd[0         ]
                    self.modes.frequency[i_mode,:,1] = hs2cmd[i_mode+  1]
                    self.modes.damping  [i_mode,:,0] = hs2cmd[0         ]
                    self.modes.damping  [i_mode,:,1] = hs2cmd[i_mode+ 61]
                    self.modes.realpart [i_mode,:,0] = hs2cmd[0         ]
                    self.modes.realpart [i_mode,:,1] = hs2cmd[i_mode+121]
            else:    
                for i_mode in range(0,num_modes):
                    self.modes.frequency[i_mode,:,0] = hs2cmd[:,0         ]
                    self.modes.frequency[i_mode,:,1] = hs2cmd[:,i_mode+  1]
                    self.modes.damping  [i_mode,:,0] = hs2cmd[:,0         ]
                    self.modes.damping  [i_mode,:,1] = hs2cmd[:,i_mode+ 61]
                    self.modes.realpart [i_mode,:,0] = hs2cmd[:,0         ]
                    self.modes.realpart [i_mode,:,1] = hs2cmd[:,i_mode+121]

            print('INFO: HS2 campbell data loaded successfully:')
            print('      - %4i modes'%num_modes)
            print('      - %4i wind speeds'%num_windspeeds)
            
        except OSError:
            print('ERROR: HAWCStab2 cmb file %s not found! Abort!'%self.filenamecmb)
            raise
        
    def read_amp_data(self):
        """
        Reads and parses the HS2 result amp data

        Raises
        ------
        OSError
            If .amp file could not be found

        Notes
        -----
        1st column is wind speed, each mode has 30 column

        # Mode number:             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1             1
        # Column num.:             2             3             4             5             6             7             8             9            10            11            12            13            14            15            16            17            18            19            20            21            22            23            24            25            26            27            28            29            30            31
        # Wind speed       TWR x [m]   phase [deg]     TWR y [m]   phase [deg] TWR yaw [rad]   phase [deg]     SFT x [m]   phase [deg]     SFT y [m]   phase [deg] SFT tor [rad]   phase [deg]  Sym edge [m]   phase [deg]   BW edge [m]   phase [deg]   FW edge [m]   phase [deg]  Sym flap [m]   phase [deg]   BW flap [m]   phase [deg]   FW flap [m]   phase [deg]Sym tors [rad]   phase [deg] BW tors [rad]   phase [deg] FW tors [rad]   phase [deg]
        """
        sensor_list = ['TWR SS','TWR FA','TWR yaw','SFT x','SFT y','SFT tor','Sym edge','BW edge','FW edge','Sym flap','BW flap','FW flap','Sym tors','BW tors','FW tors']
        num_sensors = len(sensor_list)
        
        # read file
        try:
            hs2amp = np.loadtxt(self.filenameamp,skiprows=5,dtype='float')

            myshape=np.shape(hs2amp)
            if np.shape(myshape)==(1,):
                num_windspeeds = 1
                num_modes      = int((myshape[0]-1)/num_sensors/2)
            else:
                num_windspeeds = int(myshape[0])
                num_modes      = int((myshape[1]-1)/num_sensors/2)
            amp_data = np.zeros([num_windspeeds,num_sensors,num_modes])
    
            if np.shape(myshape)==(1,):
                for i in range(0,num_modes):
                    for j in range(0,num_sensors):
                        amp_index   = i*num_sensors*2+1+2*j
                        amp_data[:,j,i] = hs2amp[amp_index]
            else:
                for i in range(0,num_modes):
                    for j in range(0,num_sensors):
                        amp_index   = i*num_sensors*2+1+2*j
                        amp_data[:,j,i] = hs2amp[:,amp_index]
    
            # determine dominant DOF per mode
            for i in range(0,num_modes):
                mean_DOF = np.mean(amp_data[:,:,i],axis=0)
                self.modes.names.append(sensor_list[np.argmax(mean_DOF)])
            
            # override first tower mode
            if self.modes.names[2] == sensor_list[0]:
                self.modes.names[1] = sensor_list[1]
                    
            print('INFO: HS2 amplitude data loaded successfully:')
            print('      - %4i modes'%num_modes)
            print('      - %4i wind speeds'%num_windspeeds)
        except OSError:
           print('ERROR: HAWCStab2 amp file %s not found! Abort!'%self.filenameamp)
           raise
        
    def read_op_data(self):
        """
        Reads the operational data from HS2

        Raises
        ------
        OSError
            If .op/.opt file could not be found
        """
        try:
            self.HS2_opdata = np.loadtxt(self.filenameop,skiprows=1,dtype='float')
        except OSError:
            print('ERROR: HAWCStab2 op file %s not found! Abort!'%self.filenameamp)
            raise
    
    def read_data(self):
        self.read_cmb_data()
        self.read_amp_data()
        self.read_op_data()