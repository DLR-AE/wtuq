"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 09.11.2020

Collection of methods to define frequency content and damping of transient time signals
"""

import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy.signal import find_peaks  # new feature (scipy 1.1.0)
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import logging

from wtuq_framework.helperfunctions import find_corresponding_axis
from libFFT import FFT_user_interface


def linear_fie(x, a, b):
    return a * x + b


def exponential_fie(x, a, k, b):
    return a * np.exp(x*k) + b 


class PostProcessor(object):
    """
    Collection of methods for the frequency and damping determination of time signals

    Parameters
    ----------
    resultdict : dict
        Collected simulation results. Key names have to match with var_names specified in the postprocessor
        settings in the config
    time : array, optional
        time vector
    sampling_time : float, optional
        sampling time of the signal
    signal : array, optional
        1D or 2D array of time signals, time dimension on axis=0
    result_dir : string, optional
        output directory for plots and data
        default is cwd

    Attributes
    ----------
    resultdict : dict
        Collected simulation results. Key names have to match with var_names specified in the postprocessor
        settings in the config
    time : array
        time vector
    sampling_time : float
        sampling time of the signal
    signal : array
        1D or 2D array of time signals, time dimension on axis=0
    result_dir : string
        output directory for plots and data
        default is cwd
    radpos : array
        1D array describing at which radial positions the signals are measured
    """
    def __init__(self, resultdict, time=np.array([]), sampling_time=float(), signal=np.array([]), result_dir='.'):

        self.resultdict = resultdict
        self.time = time
        self.sampling_time = sampling_time
        self.signal = signal
        self.result_dir = result_dir

    def read_results(self, var_names, interp_radpos=None, nr_radpos=None):
        """
        Read result dictionary

        Parameters
        ----------
        var_names : list
            List of strings, variable names to be used as signal, if multiple var names are given their signals will
            be appended
        interp_radpos : 1D array of floats, optional
            Radial positions for interpolation of a 2D data set
        nr_radpos : int, optional
            Number of equidistantly spaced radial positions to be used for each of the variables

        Raises
        ------
        NotImplementedError
            If a certain signal in the resultdict has to many dimensions

        Notes
        -----
        self.resultdict requires the variables:
            - time
            - var_name
            - (radpos: only if var_name is a 2D signal)
        """
        logger = logging.getLogger('quexus.uq.postprocessor.read_results')
        # read time and determine sampling time
        self.time = np.squeeze(self.resultdict['time'])
        # determine sampling time from time signal
        # sometimes the time stamps are not accurate enough (in Bladed f.ex.: 0.0, 0.009990785, 0.02, 0.03, ...), this
        # might lead to problems later on
        dt1 = self.time[1] - self.time[0]
        dtfull = (self.time[-1] - self.time[0]) / (np.size(self.time) - 1)
        if dt1 != dtfull:
            logger.debug('sampling time determined as time[1] - time[0] is not the same as '
                         'time[end] - time[0] / total time')
            logger.debug(str(dtfull) + 's used as sampling time')
        self.sampling_time = dtfull
        logger.debug('Sampling time = {}'.format(self.sampling_time))
        
        radpos_list = []
        signal_list = []
        for var_name in var_names:
            var_signal = np.squeeze(self.resultdict[var_name])
            if var_signal.ndim == 1:
                logger.debug('1D input signal:' + var_name)
                signal_list.append(var_signal.reshape((var_signal.size, 1)))
            elif var_signal.ndim == 2:
                logger.debug('2D input signal:' + var_name)
                if interp_radpos is not None or nr_radpos is not None:
                    orig_radpos = np.squeeze(self.resultdict['radpos'])
                    interp_signal, new_radpos = self.interpolate_signals(var_signal,
                                                                         orig_radpos,
                                                                         interp_radpos=interp_radpos,
                                                                         nr_radpos=nr_radpos)
                    radpos_list.append(new_radpos)
                else:
                    # keep 2D signal                
                    radpos_list.append(np.squeeze(self.resultdict['radpos']))
                    interp_signal = var_signal
                    
                # force time to be first dimension (row) f.ex. (# timesteps x # radpos)
                if find_corresponding_axis(interp_signal, np.size(self.time), 2)==0:
                    signal_list.append(interp_signal)
                else:
                    signal_list.append(interp_signal.T)
            else:
                logger.exception('Only 1D or 2D signals can be post-processed')
                raise NotImplementedError
  
        # compose signal and radpos arrays
        if len(signal_list) == 1:
            self.signal = signal_list[0]
        else:
            self.signal = np.concatenate(signal_list, axis=1)

        if len(radpos_list) > 1:
            self.radpos = np.concatenate(radpos_list, axis=0)
        elif len(radpos_list) == 1:
            self.radpos = np.array(radpos_list)
        else:
            logger.debug('only 1D arrays used in input, no radpos available')
            self.radpos = np.array([])      

    @staticmethod
    def interpolate_signals(signal, radpos, interp_radpos=None, nr_radpos=None):
        """
        Reduce radial positions by interpolation. Interpolation is either done
        to equidistantly spaced nr_radpos or to interp_radpos positions.
        interp_radpos overwrites nr_radpos if both are given.

        Parameters
        ----------
        signal : 2D array
            signal to be interpolated
        radpos : 1D array
            radial positions -> length has to correspond to one of the signal axes
        interp_radpos : 1D array, optional
            new radial positions, overwrites nradpos
        nr_radpos : int, optional
            linear spacing of nradpos stations between min(radpos) and max(radpos)

        Returns
        -------
        f(interp_radpos) : 2D array
            interpolated signal at the interp_radpos
        interp_radpos : 1D array
            new radial positions
        """
        logger = logging.getLogger('quexus.uq.postprocessor.interpolate_signals')

        if interp_radpos is None:
            interp_radpos = np.linspace(radpos[0], radpos[-1], nr_radpos)
            
        logger.debug('2D input signal interpolated to {} meter radial position'.format(interp_radpos))
        f = interp1d(radpos, signal, 
                     axis=find_corresponding_axis(signal, np.size(radpos), 2), 
                     fill_value='extrapolate')
        
        return f(interp_radpos), interp_radpos 

    @staticmethod
    def run_FFT(time, signal, plotting=True):
        """
        Call FFT module

        for now the config settings are hardcoded here

        Parameters
        ----------
        time : 1D array
            time vector
        signal : 1D or 2D array
            signal array
        plotting : bool, optional
            plotting flag, default is True

        Returns
        -------
        window_centres : 1D array
            Centres of the moving time windows on the signal where an FFT is performed
        fft_array : 2D array
            FFT magnitude spectrum for each of the moving windows
        """

        # config setup
        config = dict()
        config['FFT'] = dict()
        config['FFT']['mode'] = 'FFT'
        config['FFT']['window_length'] = 10
        config['FFT']['desired_signals'] = ['selected_signal']
        config['FFT']['desired_radpos'] = 0
        config['FFT']['spectrum_sets'] = [0, 2000, 1]
        config['FFT']['db_scale'] = False
        config['FFT']['nr_rotor_multiples'] = 20
        config['FFT']['window'] = 'No'
        config['FFT']['show_phase'] = False
        config['FFT']['contourplot'] = True
        
        # FFT module needs a specific 'datasets' data structure
        datasets = dict()
        datasets['manual'] = dict()
        datasets['manual']['time'] = time
        datasets['manual']['selected_signal'] = signal
        datasets['manual']['rotspeed'] = 10 * np.ones(time.shape)
        
        # run FFT
        window_centres, fft_array = FFT_user_interface(config, datasets, plotting=plotting)
        
        if plotting:
            # post-processing fft_array
            max_magn_loc = np.where(fft_array == np.amax(fft_array))         
                
            fig_test = plt.figure('fft magn at crit freq')
            ax = fig_test.gca()
            ax.plot(window_centres, fft_array[max_magn_loc[0][0], :])
            ax.set_xlabel('time [s]')
            ax.set_ylabel('FFT magnitude')
            plt.close(fig_test)

            #tempdata = moving_average(fft_array[max_magn_loc[0][0], :], 13)
            #ax.plot(contourplot_x, tempdata)
                   
            fig_test = plt.figure('deriv')
            ax = fig_test.gca()
            ax.plot(window_centres[:-1] + np.diff(window_centres), np.diff(fft_array[max_magn_loc[0][0], :]))
            ax.set_xlabel('time [s]')
            ax.set_ylabel('Gradient FFT magnitude')
            plt.close(fig_test)

        return window_centres, fft_array
    
    def prepare_signals(self, window_method=0, crop_window=[0, 1e10]):
        """
        crop and demean signals

        Parameters
        ----------
        window_method : int, optional
            Identifier for which method has to be used to select a time window. Options are 0-7.
            Default is 0
        crop_window : list, optional
            start and end time of time window in case if method=0, start time and max. oscillation amplitude factor if
            method=5
        """
        self.signal = self.demean_signals(self.signal)
        if window_method != 0 and window_method != 6 and window_method != 7:
            if self.signal.ndim > 1:
                crop_windows = np.zeros((2, self.signal.shape[1]))
                # iterate over columns (time is on axis=0)
                for iter, signal_1D in enumerate(self.signal.T):
                    crop_windows[:, iter] = self.find_window(signal_1D, window_method, 'signal_nr_'+str(iter),
                                                             plotting=True, compare_methods=False,
                                                             crop_window_for_m5=crop_window)
                crop_window = [np.median(crop_windows[0, :]), np.median(crop_windows[1, :])]
                # crop_window = [np.max(crop_windows[0, :]), np.min(crop_windows[1, :])]
        elif window_method == 6:
            # manually select window
            fig = plt.figure('Choose window')
            self.plot_windowing(fig.gca(), 'Choose window', self.time, self.signal[:, 0],
                                    [0, 0])  # use first signal if a 2D data array is used
            plt.draw()
            plt.pause(0.001)

            print('\nNote: Only 1 signal is visualized')
            window_selected = False
            while not window_selected:
                print('Start window:')
                start_window = input()
                try:
                    start_window = float(start_window)
                except ValueError:
                    print('Start window has to be a number')
                    continue
                print('End window:')
                end_window = input()
                try:
                    end_window = float(end_window)
                except ValueError:
                    print('End window has to be a number')
                    continue

                self.plot_windowing(fig.gca(), 'Choose window', self.time, self.signal[:, 0],
                                        [start_window, end_window])  # use first signal if a 2D data array is used
                plt.draw()

                print('OK? [y or n]')
                if input() == 'y':
                    window_selected = True
                else:
                    window_selected = False
            plt.close()

        elif window_method == 7:
            # manually select window
            try:
                data = np.loadtxt(os.path.join(os.path.split(self.result_dir)[0], 'uq_results',
                                               'INTERMEDIATE_manual_window_selection.txt'))
            except OSError:
                data = np.loadtxt('manual_window_selection.txt')

            crop_window = data.reshape(-1, 2)[0]
            np.savetxt(os.path.join(os.path.split(self.result_dir)[0], 'uq_results',
                                    'INTERMEDIATE_manual_window_selection.txt'), data[1:])

        # make plot with final chosen window (use first signal if a 2D data array is used)
        fig = plt.figure('final chosen window')
        self.plot_windowing(fig.gca(), 'final window', self.time, self.signal[:, 0], crop_window)
        fig.savefig(os.path.join(self.result_dir, 'final_chosen_window.pdf'))
        plt.close(fig)

        self.time, self.signal = self.crop_signals(self.time, self.signal, crop_window)
        
    def find_window(self, signal_1D, method=1, iteration_name='', plotting=False, compare_methods=True,
                    crop_window_for_m5=[0, 1e10]):
        """
        Find the window in the signal with the clean exponential increase

        Parameters
        ----------
        signal_1D : array
            1D signal vector
        method : int, optional
            flag indicating which method has to be used for the window selection, default is 1
        iteration_name : string, optional
            additional string which will be appended to figure titles, default is ''
        plotting : bool, optional
            plotting flag, default is False
        compare_methods : bool, optional
            flag to compare all window searching methods with each other, default is True
        crop_window_for_m5 : list, optional
            initial crop window (only used for method 5), default is [0, 1e10]

        Notes
        -----
        method 1: find peaks -> where do they increase consistently?
        method 2: linear curve fit on moving window -> where is the fit the best?
        method 3: exp curve fit on moving window -> where is the fit the best?
        method 4: highest gradient on FFT of critical freq
        method 5: identify start of the window and stop when oscillation amplitude becomes too big (manual threshold)

        Note: method 0, 6 and 7 do not require a window search, because they are user defined

        method 0: user determined window (from config settings)
        method 6: user determined window (interactive selection of the window for each iteration in runtime)
        method 7: user determined window (user defines window for each iteration prior to simulation with
                                            manual_window_selection.txt file in source directory)
        """

        peaks, _, _ = self.find_peaks_wrap(self.time, signal_1D, self.sampling_time)

        ########################## method 1 ###################################
        # find first window where at least 5 (and optimally as many as possible)
        # peaks are consistently increasing (or decreasing?)
        if method == 1 or compare_methods is True:
            minimum_required_window_length = 41
            end_idx_window = None
            while minimum_required_window_length > 5 and end_idx_window is None:
                minimum_required_window_length -= 1
                start_idx_window = 0
                window_length_found = 0  # number of peak samples consistently increasing
                for idx, (peak_value, peak_value_next) in enumerate(zip(signal_1D[peaks[:-1]], signal_1D[peaks[1:]])):
                    if peak_value_next > peak_value:
                        window_length_found += 1
                    elif window_length_found < minimum_required_window_length:
                        window_length_found = 0
                        start_idx_window = idx + 1
                    else:
                        end_idx_window = idx
                        break

            real_idx_start = peaks[start_idx_window]
            real_idx_end = peaks[end_idx_window]

            selected_window_method_1 = [self.time[real_idx_start], self.time[real_idx_end]]
            if plotting is True:
                fig = plt.figure('Window finding: method 1')
                ax = fig.gca()
                ax.plot(self.time, signal_1D)
                ax.plot(self.time[peaks], signal_1D[peaks], 'r*')
                ax.plot(self.time[real_idx_start:real_idx_end+1], signal_1D[real_idx_start:real_idx_end+1])
                fig.savefig(os.path.join(self.result_dir, 'window_finding_method1_' + iteration_name + '.pdf'))
                plt.close(fig)

        ########################## method 2 ###################################
        # make linear curve fit for a moving set of fit_length points
        if method == 2 or compare_methods is True:
            window_found = False
            # determining fit_length:
            # - all options (# of peaks -> 2) is computationally impossible
            # - some signals like long time signals, such that contribution of
            # noise is reduced. F.ex. Bladed
            # - other signals like short time signals
            # -> use a standard set of lengths depending on peaks.size and
            # manually add some interesting ones
            possible_fit_length = np.linspace(5, peaks.size, 10, dtype=int)
            possible_fit_length = np.append(possible_fit_length, [40, 35, 30, 25, 20, 15, 10, 5])
            for fit_length in np.sort(possible_fit_length)[::-1]:
                possible_relative_error = []
                possible_window = []
                possible_fit = []
                for peak_start_idx in range(peaks.size-fit_length):
                    selected_peaks = peaks[peak_start_idx:peak_start_idx + fit_length]
                    popt, pcov, _, success = self.linear_fit(self.time[selected_peaks],
                                                             np.log(signal_1D[selected_peaks]))

                    if success:
                        perr = np.sqrt(np.diag(pcov))
                        relative_error = np.abs(perr/popt)
                        relative_error = relative_error[0]  # only look at error on exponent
                        if np.max(relative_error) < 0.01:
                            possible_relative_error.append(np.max(relative_error))
                            possible_window.append(self.time[selected_peaks])
                            possible_fit.append(popt)

                if possible_window:
                    selected_window = possible_window[possible_relative_error.index(min(possible_relative_error))]
                    selected_fit = possible_fit[possible_relative_error.index(min(possible_relative_error))]
                    selected_window_method_2 = [selected_window[0], selected_window[-1]]
                    window_found = True
                    break

            if not window_found:
                selected_window_method_2 = [0, 0]

            if plotting is True:
                fig = plt.figure('Window finding: method 2')
                ax = fig.gca()
                # ax.plot(self.time, self.signal)
                ax.plot(self.time[peaks], np.log(signal_1D[peaks]), 'r*')
                ax.plot(selected_window, linear_fie(selected_window, selected_fit[0], selected_fit[1]))
                fig.savefig(os.path.join(self.result_dir, 'window_finding_method2_' + iteration_name + '.pdf'))
                plt.close(fig)

        ########################## method 3 ###################################
        # make exponential curve fit for a moving set of fit_length points
        if method == 3 or compare_methods is True:
            window_found = False
            possible_fit_length = np.linspace(5, peaks.size, 10, dtype=int)
            possible_fit_length = np.append(possible_fit_length, [40, 35, 30, 25, 20, 15, 10, 5])
            for fit_length in np.sort(possible_fit_length)[::-1]:
                possible_relative_error = []
                possible_window = []
                possible_fit = []
                for peak_start_idx in range(peaks.size-fit_length):
                    selected_peaks = peaks[peak_start_idx:peak_start_idx + fit_length]
                    popt, pcov, _, success = self.exponential_fit(self.time[selected_peaks],
                                                                  signal_1D[selected_peaks])

                    if success:
                        perr = np.sqrt(np.diag(pcov))
                        relative_error = np.abs(perr/popt)
                        relative_error = relative_error[1]  # only look at error on exponent
                        if np.max(relative_error) < 0.01:
                            possible_relative_error.append(np.max(relative_error))
                            possible_window.append(self.time[selected_peaks])
                            possible_fit.append(popt)

                if possible_window:
                    selected_window = possible_window[possible_relative_error.index(min(possible_relative_error))]
                    selected_fit = possible_fit[possible_relative_error.index(min(possible_relative_error))]
                    selected_window_method_3 = [selected_window[0], selected_window[-1]]
                    window_found = True
                    break

            if not window_found:
                selected_window_method_3 = [0, 0]

            if plotting is True:
                fig = plt.figure('Window finding: method 3')
                ax = fig.gca()
                ax.plot(self.time[peaks], signal_1D[peaks], 'r*')
                ax.plot(selected_window,
                        exponential_fie(selected_window, selected_fit[0], selected_fit[1], selected_fit[2]))
                fig.savefig(os.path.join(self.result_dir, 'window_finding_method3_' + iteration_name + '.pdf'))
                plt.close(fig)

        ########################## method 4 ###################################
        # highest gradient on FFT of critical freq
        if method == 4 or compare_methods is True:
            window_centres, fft_array = self.run_FFT(time=self.time, signal=signal_1D, plotting=False)
            max_magn_loc = np.where(fft_array == np.amax(fft_array))

            fft_at_crit_freq = fft_array[max_magn_loc[0][0], :]
            max_fft_magn = np.max(fft_at_crit_freq)

            # step 1: cut off unstable region.
            instability_found = False
            for iter, fft_magn in enumerate(fft_at_crit_freq):
                if fft_magn > 0.5 * max_fft_magn:
                    instability_found = True
                if instability_found and fft_magn < fft_at_crit_freq[iter-1]:
                    break
            last_idx = iter  # last iter before break, out-of-loop such that it also
            # works if the FFT signal does not reduce before the end of the signal
            window_centres_cutoff = window_centres[:last_idx]
            fft_at_crit_freq = fft_at_crit_freq[:last_idx]

            # step 2: window with highest gradient of fft_magn
            window_centres_gradient = window_centres_cutoff[:-1] + np.diff(window_centres_cutoff)
            gradient = np.diff(fft_at_crit_freq)
            max_gradient = np.max(gradient)
            idx_max_gradient = np.where(gradient == max_gradient)[0][0]
            gradient_high_enough = gradient > max_gradient/4

            idx = idx_max_gradient
            lower_bound_found = False
            while not lower_bound_found:
                idx -= 1
                if not gradient_high_enough[idx]:
                    lower_bound_found = True
                    lower_bound_idx = idx + 1
                if idx == 0:
                    lower_bound_found = True
                    lower_bound_idx = idx

            idx = idx_max_gradient
            upper_bound_found = False
            while not upper_bound_found:
                idx += 1
                if not gradient_high_enough[idx]:
                    upper_bound_found = True
                    upper_bound_idx = idx - 1
                if idx == gradient_high_enough.size - 1:
                    upper_bound_found = True
                    upper_bound_idx = idx

            selected_window_method_4 = [window_centres_gradient[lower_bound_idx],
                                        window_centres_gradient[upper_bound_idx]]
            # NOTE: for now I take the centre of the FFT windows, to be correct
            # maybe half of the window width should be added

        ########################## method 5 ###################################
        # User predefined start of the window. End of the window defined where the signal amplitude exceeds
        # crop_window_for_m5[1] times the initial oscillation amplitude.
        if method == 5 or compare_methods is True:
            init_idx = list(self.time >= crop_window_for_m5[0]).index(True)
            peaks_after_init_crop = peaks[peaks > init_idx]
            init_amplitude = max(signal_1D[peaks_after_init_crop[0]:peaks_after_init_crop[1]]) - \
                             min(signal_1D[peaks_after_init_crop[0]:peaks_after_init_crop[1]])
            peak_values = signal_1D[peaks_after_init_crop]
            peaks_above_threshold = list(peak_values > peak_values[0] + crop_window_for_m5[1]*init_amplitude)
            if any(peaks_above_threshold) is True:
                last_idx = peaks_above_threshold.index(True)
                last_idx = peaks_after_init_crop[last_idx]
            else:
                last_idx = -1

            selected_window_method_5 = [crop_window_for_m5[0], self.time[last_idx]]
            if plotting is True:
                fig = plt.figure('Window finding: method 5')
                ax = fig.gca()
                ax.plot(self.time, signal_1D)
                ax.plot(self.time[peaks], signal_1D[peaks], 'r*')
                ax.plot(self.time[init_idx:last_idx+1], signal_1D[init_idx:last_idx+1])
                fig.savefig(os.path.join(self.result_dir, 'window_finding_method5_' + iteration_name + '.pdf'))
                # plt.show()
                plt.close(fig)

        if (plotting and compare_methods) is True:
            # todo: plotting comparison atm only implemented for methods 1-4
            self.plot_windowing_methods_overview(self.time, signal_1D,
                                                 'Window finding algorithms ' + iteration_name,
                                                 selected_window_method_1=selected_window_method_1,
                                                 selected_window_method_2=selected_window_method_2,
                                                 selected_window_method_3=selected_window_method_3,
                                                 selected_window_method_4=selected_window_method_4)
        
        if method == 1:
            return selected_window_method_1
        elif method == 2:
            return selected_window_method_2
        elif method == 3:
            return selected_window_method_3
        elif method == 4:
            return selected_window_method_4
        elif method == 5:
            return selected_window_method_5

    def plot_windowing_methods_overview(self, time, signal_1D, title,
                                        selected_window_method_1=None, selected_window_method_2=None,
                                        selected_window_method_3=None, selected_window_method_4=None):
        """
        Make comparison plot for 4 windowing methods

        Parameters
        ----------
        time : array
            time vector
        signal_1D : array
            1D signal vector
        title : string
            title for the figure
        selected_window_method_1 : list
            list with time start point and end point of window for method 1
        selected_window_method_2 : list
            list with time start point and end point of window for method 2
        selected_window_method_3 : list
            list with time start point and end point of window for method 3
        selected_window_method_4 : list
            list with time start point and end point of window for method 4
        """
        fig = plt.figure(title, figsize=[14, 10])
        ax1 = fig.add_subplot(311)  # original signal and peaks
        ax2 = fig.add_subplot(323)  # method 1
        ax3 = fig.add_subplot(324)  # method 2
        ax4 = fig.add_subplot(325)  # method 3
        ax5 = fig.add_subplot(326)  # method 4

        ax1.set_title('Original signal')
        ax1.plot(time, signal_1D, label='original signal')

        if selected_window_method_1 is not None:
            self.plot_windowing(ax2, 'Method 1: consistently increasing peaks', time, signal_1D,
                                selected_window_method_1)
        if selected_window_method_2 is not None:
            self.plot_windowing(ax3, 'Method 2: best linear fit', time, signal_1D, selected_window_method_2)
        if selected_window_method_3 is not None:
            self.plot_windowing(ax4, 'Method 3: best exponential fit', time, signal_1D, selected_window_method_3)
        if selected_window_method_4 is not None:
            self.plot_windowing(ax5, 'Method 4: highest FFT gradient', time, signal_1D, selected_window_method_4)

        fig.savefig(os.path.join(self.result_dir, title + '.pdf'))
        plt.close(fig)

    def plot_windowing(self, ax, title, time, signal_1D, window):
        """ plots the original signal and the 'windowed' signal in axes """
        ax.set_title(title)
        ax.plot(time, signal_1D, label='original signal', color='C0')
        t, s = self.crop_signals(time, signal_1D, window)
        ax.plot(t, s, color='C1')

    def compare_analysis_methods_wrap(self, plotting):
        """ wrapper around compare_analysis_methods to allow 2D signal arrays """
        logger = logging.getLogger('quexus.uq.postprocessor.compare_analysis_methods_wrap')
        if self.signal.ndim > 1:
            damp_ratio_log_decr = np.zeros((self.signal.shape[1]))
            damp_ratio_lin_fit = np.zeros((self.signal.shape[1]))
            damp_ratio_exp_fit = np.zeros((self.signal.shape[1]))

            # iterate over columns (time is on axis=0)
            for iter, signal_1D in enumerate(self.signal.T):
                damp_ratio_log_decr[iter], damp_ratio_lin_fit[iter], damp_ratio_exp_fit[
                    iter] = self.compare_analysis_methods(signal_1D, '_signal_nr_' + str(iter), plotting=plotting)

            logger.debug('!! Still in development !!')
            logger.debug(
                '!! median value taken for multiple signals, no care taken of outliers/failed damping determination !!')
            self.damp_ratio_log_decr = np.median(damp_ratio_log_decr)
            self.damp_ratio_lin_fit = np.median(damp_ratio_lin_fit)
            self.damp_ratio_exp_fit = np.median(damp_ratio_exp_fit)

            with open(os.path.join(self.result_dir, 'damping_for_multiple_sensors.txt'), 'a') as f:

                f.write('\nDamping ratios logarithmic decrement\n')
                np.savetxt(f, damp_ratio_log_decr)

                f.write('\nMedian damping ratio logarithmic decrement\n')
                np.savetxt(f, [self.damp_ratio_log_decr])

                f.write('\nDamping ratios linear fit\n')
                np.savetxt(f, damp_ratio_lin_fit)

                f.write('\nMedian damping ratios linear fit\n')
                np.savetxt(f, [self.damp_ratio_lin_fit])

                f.write('\nDamping ratios exponential fit\n')
                np.savetxt(f, damp_ratio_exp_fit)

                f.write('\nMedian damping ratios exponential fit\n')
                np.savetxt(f,  [self.damp_ratio_exp_fit])

    def compare_analysis_methods(self, signal_1D, iteration_name='', plotting=True):
        """
        comparison of different methods for damping determination

        Args:
            signal_1D (np.array)
            iteration_name (str): additional string which will be appended to figure titles (f.ex. if this fie is
            called in a loop
            plotting (bool): flag for plotting
        """
        logger = logging.getLogger('quexus.uq.postprocessor.compare_analysis_methods')
        if signal_1D.ndim > 1:
            logger.warning('only 1D signals can be tested by all analysis methods')
            return

        # log decr. from peaks
        peaks, mean_vibr_freq_rads, _ = self.find_peaks_wrap(self.time,
                                                             signal_1D,
                                                             self.sampling_time)
        damp_ratio_log_decr = self.log_decr_each_period(signal_1D[peaks])

        # linear curve fit
        popt_linear, pcov_linear, damp_ratio_lin_fit, _ = self.linear_fit(self.time[peaks],
                                                                          np.log(signal_1D[peaks]),
                                                                          mean_vibr_freq_rads)
        perr = np.sqrt(np.diag(pcov_linear))
        relative_error_linear = np.abs(perr/popt_linear)
        logger.debug('Relative error linear curve fit: {}'.format(relative_error_linear))

        # exponential curve fit
        popt_exponential, pcov_exponential, damp_ratio_exp_fit, _ = self.exponential_fit(self.time[peaks],
                                                                                         signal_1D[peaks],
                                                                                         mean_vibr_freq_rads)
        perr = np.sqrt(np.diag(pcov_exponential))
        relative_error_exponential = np.abs(perr/popt_exponential)
        logger.debug('Relative error exponential curve fit: {}'.format(relative_error_exponential))
        
        if plotting:
            time = self.time
            fig = plt.figure('Comparison analysis methods', figsize=[14, 10])
            ax1 = fig.add_subplot(221)  # original signal and peaks
            ax2 = fig.add_subplot(222)  # logarithmic peaks and linear fit
            ax3 = fig.add_subplot(223)  # exponential fie from linear fit
            ax4 = fig.add_subplot(224)  # exponential fie from exponential fit
            
            ax1.plot(time, signal_1D, label='original signal')
            ax1.plot(time[peaks], signal_1D[peaks], '*', label='peaks')
            ax1.set_title('Original signal and peak finding')
            ax1.legend()
            
            ax2.plot(time[peaks], np.log(signal_1D[peaks]), '*', label='log. peaks')
            ax2.plot(time[peaks], 
                          linear_fie(time[peaks], 
                                     popt_linear[0],
                                     popt_linear[1]), label='linear fit')
            ax2.set_title('Logarithmic peaks and linear fit')
            ax2.legend()
            
            ax3.plot(time, signal_1D, label='original signal')
            ax3.plot(time[peaks], signal_1D[peaks], '*', label='peaks')
            ax3.plot(time[peaks], np.exp(linear_fie(time[peaks],
                                                    popt_linear[0],
                                                    popt_linear[1])), label='linear fit')
            ax3.plot(time[peaks], -np.exp(linear_fie(time[peaks],
                                                     popt_linear[0],
                                                     popt_linear[1])), label='linear fit')
            ax3.set_title('Exponential function based on linear fit')
            ax3.legend()
            
            ax4.plot(time, signal_1D, label='original signal')
            ax4.plot(time[peaks], signal_1D[peaks], '*', label='peaks')
            ax4.plot(time[peaks], exponential_fie(time[peaks],
                                                  popt_exponential[0],
                                                  popt_exponential[1],
                                                  popt_exponential[2]), label='exponential fit')
            ax4.set_title('Direct exponential fitting')
            ax4.legend()

            fig.savefig(os.path.join(self.result_dir, 'ComparisonAnalysisMethods' + iteration_name + '.pdf'))
            plt.close(fig)

        return damp_ratio_log_decr, damp_ratio_lin_fit, damp_ratio_exp_fit

    @staticmethod
    def log_decr_each_period(peak_values):
        """ logarithmic decrement estimation for each vibration period """
        logger = logging.getLogger('quexus.uq.postprocessor.log_decr_each_period')
        logger.debug('*** logarithmic decrement for each period ***')
        damp_ratio_list = []
        for idx in range(peak_values.size-1):
            log_decr = np.log(peak_values[idx+1]/peak_values[idx])
            damp_ratio = log_decr/(np.sqrt(4*np.pi**2 + log_decr**2)) * 100  # in %
            damp_ratio_list.append(damp_ratio)
            logger.debug('logarithmic decrement = {:10.4f}, damping ratio = {:10.3f}%'.format(log_decr, damp_ratio))
        damp_ratio = np.mean(np.asarray(damp_ratio_list))
        logger.debug('Average damping ratio = {:38.3f}%'.format(damp_ratio))
        return damp_ratio

    @staticmethod
    def linear_fit(time, signal, mean_vibr_freq_rads=-999.):
        """ curve fit of linear function on logarithmic peak values """
        logger = logging.getLogger('quexus.uq.postprocessor.linear_fit')
        logger.debug('*** Damping determination by linear curve fitting on logarithmic data ***')
        
        try:
            popt, pcov = curve_fit(linear_fie, time, signal)
            success = True
        except:
            logger.warning('linear fitting failed')
            popt = np.zeros(2)
            pcov = np.zeros(2)
            success = False
            
        logger.debug('logarithmic curve fit = {}'.format(popt))
        damping_ratio = 100 * popt[0]/mean_vibr_freq_rads
        logger.debug('damping_ratio = {:8.3f}%'.format(damping_ratio))
    
        return popt, pcov, damping_ratio, success

    @staticmethod
    def exponential_fit(time, signal, mean_vibr_freq_rads=-999.):
        """ curve fit of exponential function on original peak values """
        logger = logging.getLogger('quexus.uq.exponential_fit')
        logger.debug('*** Damping determination by exponential curve fitting on original data ***')
        
        satisfied = False
        init_coef = 0
        init_ampl = 0
        init_offset = 0
        while not satisfied:
            try:
                init_coef -= 0.0001
                popt, pcov = curve_fit(exponential_fie, 
                                       time, 
                                       signal, 
                                       p0=[init_ampl, init_coef, init_offset])
                success = True
            except RuntimeError:
                logger.warning('exponential curve_fit optimum not found')
                success = False
                popt = np.zeros(3)
                pcov = np.zeros(3)
                # continue # keep trying 
               
            __verbose = False  # TODO soft-code verbose
            if __verbose:
                fig_curvefit = plt.figure('exponential curve fitting')
                ax = fig_curvefit.gca()
            
                ax.plot(time, signal, '*')
            
                ax.plot(time, 
                        exponential_fie(time, 
                                        popt[0], popt[1], popt[2]), 'r')
                plt.show()
            
                answer = input("Satisfied with the curve fitting? (y or n)")
                if answer == 'y':
                    satisfied = True
                    print('popt =', popt)
                else:
                    satisfied = False
                    print('p0 was =', [init_ampl, init_coef, init_offset])
                    print('give new initial value for curve_fit function (a * np.exp(x*k) + b)')
                    print('popt =', popt)
                    init_ampl = float(input("Initial amplitude "))
                    init_coef = float(input("Initial exponent "))
                    init_offset = float(input("Initial offset "))
            else:
                satisfied = True
                logger.debug('popt = {}'.format(popt))
                damping_ratio = 100 * popt[1]/mean_vibr_freq_rads
                logger.debug('damping_ratio = {:10.3f}%'.format(damping_ratio))
                return popt, pcov, damping_ratio, success

    @staticmethod
    def crop_signals(time, signal, crop_window=[0, 1e10]):
        """
        crop signal
        
        Args:
            time (1D array)
            signal (1D or 2D array): time on 1st axis
            crop_window (list, optional): start and end time of time window
            
        Note:
            time is always the first dimension of the signal attribute
        """    
        mask = [(time >= crop_window[0]) & (time <= crop_window[1])]
        mask = mask[0]
        
        return time[mask], signal[mask]

    @staticmethod
    def demean_signals(signal):
        """           
        Note: time is always the first dimension of the signal attribute
        """ 
        return signal - np.mean(signal, axis=0)

    @staticmethod
    def find_peaks_wrap(time, signal, sampling_time):
        """           
        find peaks of the 1D signal and define mean vibration frequency
        """
        logger = logging.getLogger('quexus.uq.postprocessor.find_peaks_wrap')
        logger.debug('*** Find peaks ***')
        # introduce required width for peak finding to filter oscillations higher than 10 Hz
        peaks, _ = find_peaks(signal, width=1/(10*sampling_time)) 
        
        mean_vibr_period = np.mean(time[peaks][1:] - time[peaks][:-1])
        mean_vibr_freq_hz = 1 / mean_vibr_period
        mean_vibr_freq_rads = mean_vibr_freq_hz * 2 * np.pi
        logger.debug('mean vibration frequency from peaks = {:-8.3f} Hz'.format(mean_vibr_freq_hz))
        
        return peaks, mean_vibr_freq_rads, mean_vibr_freq_hz


if __name__ == "__main__":

    from wtuq_framework.helperfunctions import load_dict_h5py
    from libDMD import DMDanalysis

    result_dict = load_dict_h5py('<path to result_dict .hdf5 file>')

    # initialize postprocessing object
    postprocess = PostProcessor(result_dict, result_dir=r'<path to output directory>')

    # define which signal will be analyzed
    # args:
    # - variable name (key of resultdict),
    # - interp_radpos = optional radpos positions for interpolation of 2D array
    postprocess.read_results(['torsion_b1', 'torsion_b2', 'torsion_b3'],
                             interp_radpos=[50, 55, 60, 65, 70, 75])
    # do some pre-processing on the signals: de-mean, crop_window
    postprocess.prepare_signals(window_method=5,
                                crop_window=[10, 300])

    # initialize DMD object
    # args:
    # required: time array
    # required: sampling_time
    # required: signal array (1D or 2D, time on axis=0)
    # optional: plotting -> True or False
    # optional: result_dir
    # optional: svd_rank for DMD determination (0 -> let DMD module decide)
    DMDobj = DMDanalysis(postprocess.time,
                         postprocess.sampling_time,
                         postprocess.signal,
                         plotting=True,
                         result_dir=r'D:\test',
                         svd_rank=0)

    # do dmd analysis
    DMDobj.do_full_hodmd_analysis()
    damping = -DMDobj.DMDdamping_criticalmode