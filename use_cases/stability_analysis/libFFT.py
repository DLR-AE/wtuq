"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 02.03.2021

Frequency domain analysis of time domain signals

class: FrequencyDomainAnalysis -> core functionalities (make FFT's etc.)
class: InteractivePlot -> all interactive plotting functionalities
func: FFT_user_interface -> handles user input (config, datasets, plotting_boolean)
func: _make_data_dict -> makes structured data_dict which contains all data for the frequency domain analysis
func: _make_time_dom_array -> composes the required time domain signals into an array
func: make_FFT
func: make_cpsd
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib import rcParams
import scipy.fftpack
from scipy.interpolate import interp1d

from wtuq_framework.helperfunctions import check_min_req_data, find_corresponding_axis


class FrequencyDomainAnalysis:
    """"""

    def __init__(self, data_dict, settings):
        self.data_dict = data_dict
        self.settings = settings

    def generate_freq_dom_data(self):

        for comparison_iter, comparison_set in enumerate(self.data_dict.keys()):
            time = self.data_dict[comparison_set]['time']
            spectrum_sets = self.data_dict[comparison_set]['spectrum_sets']
            time_dom_results_set = self.data_dict[comparison_set]['time_dom_results']
            legend_set = self.data_dict[comparison_set]['legend']

            #############################
            ####---- PREPARE FFT ----####
            #############################
            fs = (time[-1] - time[0]) / (time.size - 1)
            print('timestep =', fs)
            N = int((self.settings['window_length']) / fs)
            window_centres = spectrum_sets + self.settings['window_length'] / 2  # x-axis for contour plot
            self.settings['N'] = N
            self.settings['fs'] = fs
            self.settings['window_centres'] = window_centres

            self.data_dict[comparison_set]['freq_dom_results'] = dict()
            for ii, (time_dom_result, lgnd) in enumerate(zip(time_dom_results_set.T, legend_set)):
                self.data_dict[comparison_set]['freq_dom_results'][lgnd] = dict()

                if self.settings['mode'] == 'FFT':
                    self.data_dict[comparison_set]['freq_dom_results'][lgnd]['fft_y_array'] = np.zeros(
                        (int(N / 2) - 1, window_centres.size))
                    self.data_dict[comparison_set]['freq_dom_results'][lgnd]['fft_x_array'] = np.zeros((int(N / 2) - 1,
                                                                                                        window_centres.size))  # probably redundant since all fft_x vectors will be the same
                    self.data_dict[comparison_set]['freq_dom_results'][lgnd]['phase_array'] = np.zeros(
                        (int(N / 2) - 1, window_centres.size))
                elif self.settings['mode'] == 'cpsd':
                    self.data_dict[comparison_set]['freq_dom_results'][lgnd]['cpsd_y_array'] = np.zeros(
                        (int(N / 2) - 1, window_centres.size))
                    self.data_dict[comparison_set]['freq_dom_results'][lgnd]['cpsd_x_array'] = np.zeros((int(N / 2) - 1,
                                                                                                         window_centres.size))  # probably redundant since all fft_x vectors will be the same
                    self.data_dict[comparison_set]['freq_dom_results'][lgnd]['cpsd_phase_array'] = np.zeros(
                        (int(N / 2) - 1, window_centres.size))

                for iter, start_time in enumerate(spectrum_sets):
                    print('\nFFT starting point:', start_time, 's')

                    # assumed that all arrays have the same length
                    start_idx = (np.abs(time - start_time)).argmin()

                    if self.settings['mode'] == 'FFT':
                        FFT_x, FFT_y, phase_y = make_FFT(time_dom_result, time, self.settings['window'], start_idx, N,
                                                         fs)
                        if self.settings['db_scale'] is True:
                            self.data_dict[comparison_set]['freq_dom_results'][lgnd]['fft_y_array'][:, iter] = np.log(
                                2.0 / N * np.abs(FFT_y))
                        else:
                            self.data_dict[comparison_set]['freq_dom_results'][lgnd]['fft_y_array'][:, iter] = np.abs(
                                FFT_y)
                        self.data_dict[comparison_set]['freq_dom_results'][lgnd]['fft_x_array'][:, iter] = FFT_x[1:]
                        self.data_dict[comparison_set]['freq_dom_results'][lgnd]['phase_array'][:, iter] = phase_y
                    elif self.settings['mode'] == 'cpsd':
                        for time_dom_result_2 in time_dom_results_set[:, ii + 1:].T:
                            cpsd_x, cpsd_y, cpsd_phase_y = make_cpsd(time_dom_result, time_dom_result_2, time,
                                                                     self.settings['window'], start_idx, N, fs)
                            if self.settings['db_scale'] is True:
                                self.data_dict[comparison_set]['freq_dom_results'][lgnd]['cpsd_y_array'][:,
                                iter] = np.log(2.0 / N * np.abs(cpsd_y))
                            else:
                                self.data_dict[comparison_set]['freq_dom_results'][lgnd]['cpsd_y_array'][:,
                                iter] = cpsd_y
                            self.data_dict[comparison_set]['freq_dom_results'][lgnd]['cpsd_x_array'][:, iter] = cpsd_x[
                                                                                                                1:]
                            self.data_dict[comparison_set]['freq_dom_results'][lgnd]['cpsd_phase_array'][:,
                            iter] = cpsd_phase_y


class InteractivePlot(FrequencyDomainAnalysis):
    """
    #################################
    ####---- DATA STRUCTURES ----####
    #################################

    # |- graphics
    # |   |- fig_comb
    # |   |- comparison_iter_1
    # |   |   |- time_dom_lines
    # |   |   |   |- start_time_1
    # |   |   |   |   |- figure artist 1
    # |   |   |   |   |- figure artist 2
    # |   |   |   |   |- ...
    # |   |   |   |- start_time_2
    # |   |   |   |   |- figure artist 1
    # |   |   |   |   |- figure artist 2
    # |   |   |   |   |- ...
    # |   |   |   |- ...
    # |   |   |- spectrum_lines
    # |   |   |   |- start_time_1
    # |   |   |   |   |- figure artist 1
    # |   |   |   |   |- figure artist 2
    # |   |   |   |   |- ...
    # |   |   |   |- start_time_2
    # |   |   |   |   |- figure artist 1
    # |   |   |   |   |- figure artist 2
    # |   |   |   |   |- ...
    # |   |   |   |- ...
    # |   |   |- visible_idx
    # |   |   |- visible_set
    # |   |   |- related_axes
    # |   |   |- spectrum_sets
    # |   |- comparison_iter_2
    # ...
    """

    def __init__(self, data_dict, settings):
        super(InteractivePlot, self).__init__(data_dict, settings)
        self.modify_default_plt_settings()

    def modify_default_plt_settings(self):
        # change default figure style matplotlib
        rcParams['lines.linewidth'] = 0.8
        rcParams['lines.markersize'] = 0.8

        # change default function of arrow keys in plot
        # arrow keys are by default used to toggle between views (zoom in/out)
        try:
            rcParams['keymap.back'].remove('left')
            rcParams['keymap.forward'].remove('right')
        except ValueError:
            print("keymap default parameters unchanged")

    def init_interactive_figure(self):
        self.graphics = dict()  # containing all required figure information

        self.graphics['fig_comb'] = plt.figure('FFT results',
                                               figsize=(80, 60))  # larger than full screen -> resizes to full screen
        self.graphics['fig_comb'].canvas.mpl_connect('button_press_event', self._onpick1)
        self.graphics['fig_comb'].canvas.mpl_connect('key_press_event', self._on_key)
        plt.show(block=False)

    def complete_graphics_data(self):

        nr_comparisons = len(self.data_dict)
        for comparison_iter, comparison_set in enumerate(self.data_dict.keys()):
            ###################################################################
            ####---- OPEN UP THE AXES BELONGING TO THIS COMPARISON_SET ----####
            ###################################################################
            # unpack data_dict
            time = self.data_dict[comparison_set]['time']
            spectrum_sets = self.data_dict[comparison_set]['spectrum_sets']
            rotspeed = self.data_dict[comparison_set]['rotspeed']
            title = self.data_dict[comparison_set]['title']
            lgnd = self.data_dict[comparison_set]['legend']
            time_dom_results_set = self.data_dict[comparison_set]['time_dom_results']
            freq_dom_results_set = self.data_dict[comparison_set]['freq_dom_results']

            # setup graphics dict
            self.graphics[comparison_set] = dict()
            self.graphics[comparison_set]['time_dom_lines'] = dict()
            self.graphics[comparison_set]['spectrum_lines'] = dict()
            self.graphics[comparison_set]['phase_lines'] = dict()  # only used in cpsd
            self.graphics[comparison_set]['visible_idx'] = 0
            self.graphics[comparison_set]['visible_set'] = self.data_dict[comparison_set]['spectrum_sets'][0]
            self.graphics[comparison_set]['spectrum_sets'] = self.data_dict[comparison_set]['spectrum_sets']

            # Make all axes for this dataset/signal combination
            if self.settings['show_phase'] is True:
                ax_time_dom = self.graphics['fig_comb'].add_subplot(nr_comparisons * 3, 1, comparison_iter * 3 + 1)
                ax_freq_spectrum_magn = self.graphics['fig_comb'].add_subplot(nr_comparisons * 3, 1,
                                                                              comparison_iter * 3 + 2)
                ax_freq_spectrum_phase = self.graphics['fig_comb'].add_subplot(nr_comparisons * 3, 1,
                                                                               comparison_iter * 3 + 3)
            else:
                ax_time_dom = self.graphics['fig_comb'].add_subplot(nr_comparisons * 2, 1, comparison_iter * 2 + 1)
                ax_freq_spectrum_magn = self.graphics['fig_comb'].add_subplot(nr_comparisons * 2, 1,
                                                                              comparison_iter * 2 + 2)

            ax_time_dom.set_xlabel('Time [s]')
            ax_time_dom.set_ylabel('Signal amplitude')
            ax_time_dom.set_title(title, loc='right')
            ax_time_dom.grid(True)

            for time_dom_result in time_dom_results_set.T:
                ax_time_dom.plot(time, time_dom_result)
            ax_time_dom.legend(lgnd)

            if self.settings['contourplot'] is True:
                ax_freq_spectrum_magn.set_ylim([0, 5])
                ax_freq_spectrum_magn.set_xlabel('Time [s]')
                ax_freq_spectrum_magn.set_ylabel('Frequency [Hz]')
            else:
                ax_freq_spectrum_magn.set_xlim([0, 10])
                ax_freq_spectrum_magn.set_xlabel('Frequency [Hz]')
                if self.settings['db_scale']:
                    ax_freq_spectrum_magn.set_ylabel('Amplitude [dB]')
                else:
                    ax_freq_spectrum_magn.set_ylabel('Amplitude')
                ax_freq_spectrum_magn.grid(True)

            if self.settings['show_phase'] is True:
                ax_freq_spectrum_phase.set_xlim([0, 10])
                ax_freq_spectrum_phase.set_xlabel('Frequency [Hz]')
                ax_freq_spectrum_phase.set_ylabel('Phase [deg]')
                ax_freq_spectrum_phase.grid(True)

            ###########################################################################
            ####---- PLOT DATA IN OPENED AXES BELONGING TO THIS COMPARISON SET ----####
            ###########################################################################

            for iter, start_time in enumerate(spectrum_sets):

                # assumed that all arrays have the same length
                start_idx = (np.abs(time - start_time)).argmin()

                # plot lines indicating which time data is used
                ax_time_dom.axvline(x=start_time, c='r')
                ax_time_dom.axvline(x=start_time + self.settings['window_length'], c='r')

                # plot FFT or cpsd curves
                for ii, (key, value) in enumerate(freq_dom_results_set.items()):
                    if self.settings['mode'] == 'FFT':
                        FFT_x = value['fft_x_array'][:, iter]
                        FFT_y = value['fft_y_array'][:, iter]
                        phase_y = value['phase_array'][:, iter]
                        if self.settings['contourplot'] is False:
                            ax_freq_spectrum_magn.plot(FFT_x, FFT_y, '.-', color='C' + str(ii))
                            if self.settings['show_phase']:
                                ax_freq_spectrum_phase.plot(FFT_x, phase_y, '.-', color='C' + str(ii))

                    elif self.settings['mode'] == 'cpsd':
                        cpsd_x = value['cpsd_x_array'][:, iter]
                        cpsd_y = value['cpsd_y_array'][:, iter]
                        cpsd_phase_y = value['cpsd_phase_array'][:, iter]
                        if self.settings['contourplot'] is False:
                            ax_freq_spectrum_magn.plot(cpsd_x, cpsd_y, '.-', color='C' + str(ii))
                            if self.settings['show_phase']:
                                ax_freq_spectrum_phase.plot(cpsd_x, cpsd_phase_y, '.-', color='C' + str(ii))

                # plot rotor multiples
                if self.settings['contourplot'] is False:
                    for i in range(1, self.settings['nr_rotor_multiples']):
                        ax_freq_spectrum_magn.axvline(
                            x=i * np.min(rotspeed[start_idx:start_idx + self.settings['N']]) / 60, ymax=0.1, c='g',
                            ls='--')
                        ax_freq_spectrum_magn.axvline(
                            x=i * np.mean(rotspeed[start_idx:start_idx + self.settings['N']]) / 60, ymax=0.1,
                            c='b')
                        ax_freq_spectrum_magn.axvline(
                            x=i * np.max(rotspeed[start_idx:start_idx + self.settings['N']]) / 60, ymax=0.1, c='r',
                            ls='--')

                    # add all lines to the graphics dictionary and set them invisible
                    self.graphics[comparison_set]['spectrum_lines'][start_time] = []
                    for artist in ax_freq_spectrum_magn.lines:
                        if artist.get_visible():  # all visible lines are added to the spectrum dictionary and are turned invisible
                            self.graphics[comparison_set]['spectrum_lines'][start_time].append(artist)
                            artist.set_visible(False)

                    if self.settings['show_phase'] is True:
                        for artist in ax_freq_spectrum_phase.lines:
                            if artist.get_visible():  # all visible lines are added to the spectrum dictionary and are turned invisible
                                self.graphics[comparison_set]['spectrum_lines'][start_time].append(artist)
                                artist.set_visible(False)
                else:
                    self.graphics[comparison_set]['spectrum_lines'][start_time] = []
                # the time domain signal itself should remain on the plot.
                # it should therefore belong to either all or none of the time_dom_lines
                self.graphics[comparison_set]['time_dom_lines'][start_time] = []
                for line_nr, artist in enumerate(ax_time_dom.lines):
                    if artist.get_visible() and line_nr > time_dom_results_set.shape[1] - 1:  # the first visible lines (time signals) are not added to the time_dom_lines
                        self.graphics[comparison_set]['time_dom_lines'][start_time].append(artist)
                        artist.set_visible(False)

            if self.settings['contourplot'] is True:
                ax_freq_spectrum_magn.set_xlim(ax_time_dom.get_xlim())
                contourplot_x = self.settings['window_centres']
                contourplot_y = freq_dom_results_set[lgnd[0]]['fft_x_array'][:, 0]
                contourplot_z = freq_dom_results_set[lgnd[0]]['fft_y_array']
                # multiple options to make the contour plot
                # 1) use the matplotlib .specgram method (this makes the FFT's itself). I don't fully understand all
                # settings yet...
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html#matplotlib.pyplot.specgram
                # ax_freq_spectrum_magn.specgram(time_dom_results_set[:, 0], Fs=self.settings['fs'])
                # 2) full contour plot
                ax_freq_spectrum_magn.contourf(contourplot_x, contourplot_y, contourplot_z, cmap='jet')
                # 3) pcolormesh
                # ax_freq_spectrum_magn.pcolormesh(contourplot_x, contourplot_y, contourplot_z)

            # save axes corresponding to this fname/signal combination
            if self.settings['show_phase']:
                self.graphics[comparison_set]['related_axes'] = [ax_time_dom, ax_freq_spectrum_magn,
                                                                 ax_freq_spectrum_phase]
            else:
                self.graphics[comparison_set]['related_axes'] = [ax_time_dom, ax_freq_spectrum_magn]

        if comparison_iter == -1:
            print('WARNING: no results for FFT analysis')
            plt.close(self.graphics['fig_comb'])
            return

        # set tight layout
        # TODO: tight_layout does not seem to be working automatically. Padding added manually
        self.graphics['fig_comb'].tight_layout(pad=15, h_pad=15)

        # set all axes to first FFT instance
        # set initial visible figure
        for comparison_set in self.graphics.keys():
            if comparison_set == 'fig_comb':
                continue
            self.active_plot = comparison_set  # dataset/signal combination (comparison_iter) which can be modified with arrow keys
            self._update_figure(self.graphics[self.active_plot]['visible_set'])

        self._draw_active_plot()  # highlight axes which are active

        plt.show()  # hold and allow interactivity until figure is closed

    #################################################
    ####---- Interactive picking active plot ----####
    #################################################
    def _onpick1(self, event):
        """
        Sets active plot based on mouse input.

        INPUT:
            event (matplotlib.MouseEvent): mouse click on axes
            self.graphics (dict): dictionary containing all required figure info
            self.active_plot (str): indicator which plot (fname/signal combination) is active

        OUTPUT:
            updates active_plot attribute
        """
        for comparison_set in self.graphics.keys():
            if comparison_set == 'fig_comb':
                continue
            if event.inaxes in self.graphics[comparison_set]['related_axes']:
                print('active_plot:', comparison_set)
                if self.active_plot != comparison_set:  # only update if the active plot really changed
                    # set axes view back to default
                    for active_axes in self.graphics[self.active_plot]['related_axes']:
                        for child in active_axes.get_children():
                            if isinstance(child, Spine):
                                child.set_linewidth(0.5)
                    # update active_plot
                    self.active_plot = comparison_set
                    self._draw_active_plot()
                break
            else:
                print('click on axes to change active plot')

    ###########################################
    ####---- Interactive set selection ----####
    ###########################################

    def _on_key(self, event):
        """
        Steps through different FFT starting points based on arrow keys input

        INPUT:
            event (matplotlib.KeyEvent): Pressed key (only arrow key inputs are used)
            self.graphics (dict): dictionary containing all required figure info
            self.active_plot (str): indicator which plot (fname/signal combination) is active

        OUTPUT:
            updates figure to new timestep if the right of left arrow key is pressed
        """
        print('you pressed', event.key)

        spectrum_sets = self.graphics[self.active_plot]['spectrum_sets']

        if event.key == 'left':
            self.graphics[self.active_plot]['visible_idx'] -= 1
            if self.graphics[self.active_plot]['visible_idx'] < -spectrum_sets.size:
                self.graphics[self.active_plot]['visible_idx'] = spectrum_sets.size - 1
            new_visible_set = spectrum_sets[self.graphics[self.active_plot]['visible_idx']]
            self._update_figure(new_visible_set)
        elif event.key == 'right':
            self.graphics[self.active_plot]['visible_idx'] += 1
            if self.graphics[self.active_plot]['visible_idx'] >= spectrum_sets.size:
                self.graphics[self.active_plot]['visible_idx'] = -spectrum_sets.size
            new_visible_set = spectrum_sets[self.graphics[self.active_plot]['visible_idx']]
            self._update_figure(new_visible_set)
        else:
            print('Press the arrow keys')

    def _draw_active_plot(self):
        """
        Highlights active axes

        INPUT:
            self.graphics (dict): dictionary containing all required figure info
            self.active_plot (str): indicator which plot (fname/signal combination) is active
        """
        for active_axes in self.graphics[self.active_plot]['related_axes']:
            for child in active_axes.get_children():
                if isinstance(child, Spine):
                    child.set_linewidth(2)

        # update figure
        self.graphics['fig_comb'].canvas.draw()

    def _update_figure(self, new_visible_set):
        """
        Updates figure canvas

        INPUT:
            self.graphics (dict): dictionary containing all required figure info
            self.active_plot (str): indicator which plot (fname/signal combination) is active
            new_visible_set (int): time step (corresponding to dict key) which has to be shown
        OUTPUT:
            sets all lines corresponding to the previous time step invisible and
            makes all lines corresponding to the new_visible_set visible
        """
        visible_set = self.graphics[self.active_plot]['visible_set']

        for artist in self.graphics[self.active_plot]['spectrum_lines'][visible_set]:
            artist.set_visible(False)
        for artist in self.graphics[self.active_plot]['time_dom_lines'][visible_set]:
            artist.set_visible(False)

        for artist in self.graphics[self.active_plot]['spectrum_lines'][new_visible_set]:
            artist.set_visible(True)
        for artist in self.graphics[self.active_plot]['time_dom_lines'][new_visible_set]:
            artist.set_visible(True)

        self.graphics[self.active_plot]['visible_set'] = new_visible_set

        # update figure
        self.graphics['fig_comb'].canvas.draw()


################################
####---- Input handling ----####
################################
def FFT_user_interface(config, datasets, plotting):
    """
    - gets user config and data
    - operates FrequencyDomainAnalysis and InteractivePlot

    INPUT:
        config (dict): already parsed configuration file with user settings
        datasets (dict): dictionary with all content of the input file
        plotting (bool): flag whether the interactive plot should be made or not
    OUTPUT:
        window centres: where FFT has been made
        fft_y_array: FFT magnitude spectrum for each window
    """
    #############################
    ####---- READ CONFIG ----####
    #############################
    # user input which will be stored in the signals dictionary
    desired_signals = config['FFT']['desired_signals']
    desired_radpos = config['FFT']['desired_radpos']
    spectrum_sets_input = config['FFT']['spectrum_sets']
    spectrum_sets = np.arange(float(spectrum_sets_input[0]),
                              float(spectrum_sets_input[1]),
                              float(spectrum_sets_input[2]))
    # user input which will be stored in settings dictionary
    settings = dict()
    settings['mode'] = config['FFT']['mode']
    settings['window_length'] = config['FFT']['window_length']  # [s] - time length for FFT
    settings['db_scale'] = config['FFT']['db_scale']
    settings['nr_rotor_multiples'] = config['FFT']['nr_rotor_multiples']
    settings['window'] = config['FFT']['window']
    settings['show_phase'] = config['FFT']['show_phase']
    settings['contourplot'] = config['FFT']['contourplot']

    data_dict = _make_data_dict(datasets, desired_signals, settings['window_length'],
                                spectrum_sets, settings['mode'], desired_radpos)

    if plotting is True:
        analysis = InteractivePlot(data_dict, settings)
        analysis.init_interactive_figure()
        analysis.generate_freq_dom_data()
        analysis.complete_graphics_data()
    else:
        analysis = FrequencyDomainAnalysis(data_dict, settings)
        analysis.generate_freq_dom_data()

    return settings['window_centres'], \
           analysis.data_dict['set_1']['freq_dom_results'][analysis.data_dict['set_1']['legend'][0]]['fft_y_array']


##########################################################
####---- Initializing data_dict from config input ----####
##########################################################
def _make_data_dict(datasets, desired_signals, window_length, spectrum_sets, mode, desired_radpos):
    """
    Structures the input in the standard data_dict lay-out. Each comparison set
    indicates new axes in the figure. In the cpsd mode all signals are plotted
    in the same axis, so only one set will be used. In the FFT mode all signals
    are by default plotted in different axes, with the exception of different
    radial positions of the same signal (f.ex. out-of-plane defl. at 20, 50,
    70m along the blade).

    INPUT:
        datasets (dict): dictionary with all content of the input file
        desired_signals (list of strings): list of desired signals which have
        to be analysed in the frequency domain.
        window_length (float): length of each time to be used for Fourier transform
        spectrum_sets (array): starting points for each Fourier transform
        mode (string): 'cpsd', 'FFT' or 'No'
        desired_radpos (list of int): radial positions to which the data has to
        be interpolated if a 2d (time x radial positions) input array is given
    OUTPUT:
        data_dict (dict): structured data for the frequency domain analysis

    |- data_dict
    |   |- set_1 (goes in one of the axes)
    |   |   |- ds_name (name)
    |   |   |- time (1D time vector)
    |   |   |- rotspeed (1D rotor speed vector)
    |   |   |- spectrum_sets (array with start points for FFT)
    |   |   |- time_dom_results (array with time domain signals. Can be 1D or 2D. Time on axis=0)
    |   |   |- legend (list with name for each of the columns in time_dom_results)
    |   |   |- (freq_dom_results) (dictionary with results of frequency domain analysis)
    |   |- set_2
    ...
    """
    data_dict = dict()

    set_nr = 0
    # different datasets always go in different axes
    for fname in datasets:
        if check_min_req_data(datasets[fname], ['time', 'rotspeed']):
            print('WARNING time or rotspeed not available in data for', fname)
            continue

        time = datasets[fname]['time']
        rotspeed = datasets[fname]['rotspeed']

        ###################################################################
        ####---- Check if spectrum set does not exceed time signal ----####
        ###################################################################
        # I assume that:
        #   1. all time signals within the same file have the same length
        #   2. time signal is positive and strictly increasing
        spectrum_sets_tool = np.copy(spectrum_sets)
        if datasets[fname]['time'][0] > np.min(spectrum_sets_tool) or datasets[fname]['time'][-1] < np.max(
                spectrum_sets_tool) + window_length:
            spectrum_sets_tool = spectrum_sets_tool[(datasets[fname]['time'][0] < spectrum_sets_tool) & (
                    spectrum_sets_tool + window_length < datasets[fname]['time'][-1])]
            print('WARNING: FFT discretization adapted for', fname)
            print('New discretization:', spectrum_sets_tool)

        if mode == 'cpsd':  # put all time series in the same set
            set_nr += 1
            set_name = 'set_' + str(set_nr)
            data_dict[set_name] = dict()
            data_dict[set_name]['ds_name'] = fname
            data_dict[set_name]['time'] = time
            data_dict[set_name]['rotspeed'] = rotspeed
            data_dict[set_name]['spectrum_sets'] = spectrum_sets_tool
            data_dict[set_name]['title'] = fname
            data_dict[set_name]['legend'] = []  # channel list

            data_dict[set_name]['time_dom_results'] = np.zeros((time.size, 1))  # just a dummy to append to

            for desired_signal in desired_signals:
                if check_min_req_data(datasets[fname], [desired_signal]):
                    print('WARNING:', desired_signal, 'not available in dataset:', fname)
                    data_dict[set_name]['time_dom_results'] = np.zeros(time.shape).reshape(time.size, 1)
                    data_dict[set_name]['legend'].append(desired_signal + ' not available in dataset: ' + fname)
                    continue

                # all desired signals are structured as column vectors in 'time_dom_results'
                # so row dimension = time dimension
                sgnl_tmp, lgnd_tmp = _make_time_dom_results_array(datasets[fname], desired_signal,
                                                                  desired_radpos, time)
                data_dict[set_name]['time_dom_results'] = np.hstack((data_dict[set_name]['time_dom_results'], sgnl_tmp))
                data_dict[set_name]['legend'].append(lgnd_tmp[0])

            # remove zeros dummy
            data_dict[set_name]['time_dom_results'] = data_dict[set_name]['signals'][:, 1:]

        elif mode == 'FFT':  # make each desired signal a new set
            for desired_signal in desired_signals:
                set_nr += 1
                set_name = 'set_' + str(set_nr)
                data_dict[set_name] = dict()
                data_dict[set_name]['ds_name'] = fname
                data_dict[set_name]['time'] = time
                data_dict[set_name]['rotspeed'] = rotspeed
                data_dict[set_name]['spectrum_sets'] = spectrum_sets_tool
                data_dict[set_name]['title'] = fname + ': ' + desired_signal

                if check_min_req_data(datasets[fname], [desired_signal]):
                    print('WARNING:', desired_signal, 'not available in dataset:', fname)
                    data_dict[set_name]['time_dom_results'] = np.zeros(time.shape).reshape(time.size, 1)
                    data_dict[set_name]['legend'] = [desired_signal + ' not available in dataset: ' + fname]
                    continue

                # all desired signals are structured as column vectors in 'time_dom_results'
                # so row dimension = time dimension
                data_dict[set_name]['time_dom_results'], data_dict[set_name]['legend'] = _make_time_dom_results_array(
                    datasets[fname],
                    desired_signal,
                    desired_radpos,
                    time)
    return data_dict


def _make_time_dom_results_array(fdata, desired_signal, desired_radpos, time):
    """
    Makes time_dom_results vector or array based on config input and available datasets

    INPUT:
        fdata (dict): dataset of 1 input file
        desired_signal (str): signal which has to be added to the signals matrix
        desired_radpos (list): radial positions which have to be used for the desired_signal
        time (array): time array
    OUTPUT:
        time_dom_results (array): Time series which will be used for the freq. dom. analysis.
                                  Row dimension == time dimension, multiple columns can be
                                  multiple radial positions for certain desired_signal
        lgnd (list): list with description of each signal (column signals).
    """
    # if desired signal is 2d array
    signal_data = fdata[desired_signal]
    if signal_data.squeeze().ndim == 2:
        # force time to be row dimension
        if find_corresponding_axis(signal_data, time.size, 2) == 1:
            signal_data = signal_data.T

        if desired_radpos == ['all']:
            time_dom_results = signal_data
            lgnd = [desired_signal + ' at ' + str(rp) + 'm' for rp in fdata['radpos']]
        else:
            f = interp1d(fdata['radpos'], signal_data, axis=1, fill_value='extrapolate')
            time_dom_results = f(desired_radpos)
            lgnd = [desired_signal + ' at ' + rp + 'm' for rp in desired_radpos]
    else:
        time_dom_results = signal_data
        # force to be column vector
        time_dom_results = time_dom_results.reshape(time_dom_results.size, 1)
        lgnd = [desired_signal]

    return time_dom_results, lgnd


def make_FFT(s, t, window, start_idx, N, fs):
    """
    Do FFT on signal 's' with time series 't'

    INPUT:
        s (vector): signal
        t (vector): time
    OUTPUT:
        fft_x (vector): frequency distribution Fourier transform
        fft_y (complex vector): result of the Fourier transform
        phase_y (vector): phase of the Fourier transform
    """

    fft_x = np.linspace(0.0, 1.0 / (2.0 * fs), int(N / 2))

    if window == 'No':
        fft_y = scipy.fftpack.fft(s[start_idx: start_idx + N])
    elif window == 'Hamming':
        fft_y = np.hamming(N) * scipy.fftpack.fft(s[start_idx: start_idx + N])
    elif window == 'Hanning':
        fft_y = np.hanning(N) * scipy.fftpack.fft(s[start_idx: start_idx + N])

    fft_y = fft_y[1:N // 2]
    phase_y = (180 / np.pi) * np.angle(fft_y)

    return fft_x, fft_y, phase_y


def make_cpsd(s1, s2, t, window, start_idx, N, fs):
    """
    Make cross power spectral density of signals 1 and 2 with time series 't'

    INPUT:
        s1 (vector): signal 1
        s2 (vector): signal 2
        t (vector): time
    OUTPUT:
        FFT_x_1 (vector): frequency distribution Fourier transform
        CPSD (complex vector): cross-power spectral density
        CPSD_p (vector): phase of the cross-power signal in degrees
    """
    FFT_x_1, FFT_y_1, phase_y_1 = make_FFT(s1, t, window, start_idx, N, fs)
    FFT_x_2, FFT_y_2, phase_y_2 = make_FFT(s2, t, window, start_idx, N, fs)

    f_real_1 = FFT_y_1.real
    f_imag_1 = FFT_y_1.imag
    f_real_2 = FFT_y_2.real
    f_imag_2 = FFT_y_2.imag

    nhalf = f_real_1.size

    A_complex_conjugate = np.zeros(nhalf, 'complex')
    B_complex = np.zeros(nhalf, 'complex')

    for i in range(nhalf):
        A_complex_conjugate[i] = complex(f_real_1[i], -f_imag_1[i])
        B_complex[i] = complex(f_real_2[i], f_imag_2[i])

    CPSD = A_complex_conjugate * B_complex / fs
    CPSD_p = (180 / np.pi) * np.angle(CPSD)

    return FFT_x_1, CPSD, CPSD_p
