"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 09.11.2020

Wrapper around pyDMD Higher-Order Dynamic Mode Decomposition package
https://mathlab.github.io/PyDMD/
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from pydmd import HODMD


class DMDresult(object):
    """
    Structured way to store and postprocess results of a DMD analysis

    Parameters
    ----------
    hodmd : pyDMD HODMD object, optional
        default is None
    eigs : array, optional
        eigenvalues in continuous time, default is None
    formatted_result : array, optional
        array with summarised results of the DMD analysis (participation factors, frequency, damping), default is None

    Attributes
    ----------
    hodmd : pyDMD HODMD object
    eigs : array
        eigenvalues in continuous time
    formatted_result : array
        array with summarised results of the DMD analysis (participation factors, frequency, damping)
    DMDdamping_criticalmode : float
        most negative damped DMD mode
    """
    def __init__(self, hodmd=None, eigs=None, formatted_result=None):
        self.hodmd = hodmd
        self.eigs = eigs
        self.formatted_result = formatted_result
        self.DMDdamping_criticalmode = None

    def summarise_and_output_results(self, output_dir):
        """
        Export DMD result data to text files.

        Parameters
        ----------
        output_dir : string
            Path to directory where DMD results have to be exported
        """
        np.savetxt(os.path.join(output_dir, 'DMD_results', 'DMD_summary.txt'), self.formatted_result, fmt='%19.2f',
                   header='{:>17}{:>20}{:>20}'.format('participation', 'frequency [Hz]', 'damping ratio [%]'))
        np.savetxt(os.path.join(output_dir, 'DMD_results', 'DMD_damping_criticalmode.txt'),
                   np.array([self.DMDdamping_criticalmode]))
        np.savetxt(os.path.join(output_dir, 'DMD_results', 'DMD_NRMSD.txt'),
                   np.array([self.nrmsd_total]))


class DMDanalysis(object):
    """
    wrapper for DMD (Dynamic Mode Decomposition) module

    Parameters
    ----------
    time : array
        1D time vector
    signal : array
        (1D or 2D array): time dimension on axis=0
    sampling_time : float
        sampling time of the time vector
    plotting : bool, optional
        plotting flag, default is False
    result_dir : string, optional
        path to result directory, default is cwd
    radpos : array, optional
        1D vector with radial positions, default is np.array([0])
    svd_rank : int, optional
        Truncation rank for the DMD algorithm, default is 0 (automatic truncation selection)

    Attributes
    ----------
    time : array
        1D time vector
    signal : array
        (1D or 2D array): time dimension on axis=0
    sampling_time : float
        sampling time of the time vector
    plotting : bool
        plotting flag
    result_dir : string
        path to result directory
    radpos : array
        1D vector with radial positions
    svd_rank : int
        Truncation rank for the DMD algorithm
    DMD_result : DMDresult object
        Summary of DMD results
    """
    def __init__(self, time, sampling_time, signal, plotting=False, result_dir='.', radpos=np.array([0]), svd_rank=0):
        self.time = time
        self.sampling_time = sampling_time
        self.signal = signal
        self.plotting = plotting
        self.radpos = radpos
        self.svd_rank = svd_rank
        self.result_dir = result_dir
        self.DMDresult = DMDresult()
        os.makedirs(os.path.join(self.result_dir, 'DMD_results'), exist_ok=True)

    def do_full_hodmd_analysis(self):
        """
        Run through all steps of the hodmd analysis on the signal attribute. Run pyDMD HODMD method, analyse the results
        and make post-processed plots
        """
        self.execute_HODMD()
        self.analyse_eigs()
        self.compute_NRMSD(self.signal, self.DMDresult.hodmd.reconstructed_data.real.T)  # compute DMD error
        self.DMDresult.summarise_and_output_results(self.result_dir)

        if self.plotting:
            # fig_full -> all signals in one figure
            fig_full = plt.figure(figsize=[14, 10])
            ax2 = fig_full.add_subplot(211)  # reconstructed signal (for 2D only a couple positions)
            ax3 = fig_full.add_subplot(212)  # dmd dynamics
        
            self.plot_reconstructed_signal(ax2,
                                           self.DMDresult.hodmd.original_timesteps,
                                           self.signal, 
                                           self.DMDresult.hodmd.dmd_timesteps,
                                           self.DMDresult.hodmd.reconstructed_data.real.T)
            self.plot_alldynamics(ax3)

            fig_full.savefig(os.path.join(self.result_dir, 'DMD_results', 'DMDanalysis.pdf'))
            plt.close(fig_full)

            # fig_per_signal -> each signals gets its own figure
            if self.signal.ndim > 1:
                for signal_id in range(self.signal.shape[1]):
                    fig_per_signal = plt.figure(figsize=[14, 7])
                    ax_per_signal = fig_per_signal.gca()
                    self.plot_reconstructed_signal(ax_per_signal,
                                                   self.DMDresult.hodmd.original_timesteps,
                                                   self.signal[:, signal_id], 
                                                   self.DMDresult.hodmd.dmd_timesteps,
                                                   self.DMDresult.hodmd.reconstructed_data.real.T[:, signal_id])
                    fig_per_signal.savefig(os.path.join(self.result_dir, 'DMD_results',
                                                        'DMDplot_per_signal_'+str(signal_id) + '.pdf'))
                    plt.close(fig_per_signal)
                # self.plot_modes()

    def execute_HODMD(self):
        """
        Execute pyDMD HODMD method for modal decomposition of the signal array
        """
        logger = logging.getLogger('quexus.uq.postprocessor.dmd.execute_HODMD')
        logger.debug('Execute HODMD method')
        self.DMDresult.hodmd = HODMD(svd_rank=self.svd_rank, exact=True, opt=True, d=30).fit(self.signal.T)
            
    def analyse_eigs(self):
        """
        Post-process eigenvalues of the DMD decomposed signal
        """
        logger = logging.getLogger('quexus.uq.postprocessor.dmd.analyse_eigs')

        # convert eigenvalues to continuous time
        eigs = np.log(self.DMDresult.hodmd.eigs)/self.sampling_time
        self.DMDresult.eigs = eigs
        
        # modal participation for now determined by the norm of the dynamics signal
        modal_participation = np.linalg.norm(self.DMDresult.hodmd.dynamics, axis=1)
        modal_participation = 100 * modal_participation / np.sum(modal_participation)

        # Join modal participation / frequency / damping in numpy array
        logger.debug('*** Dynamic Mode Decomposition ***')
        formatted_DMDresult = np.zeros((eigs.size, 3))
        for iter, (eig, eig_discr, part) in enumerate(zip(eigs, self.DMDresult.hodmd.eigs, modal_participation)):
            freq = eig.imag  # rad/s
            damp_ratio = 1/self.sampling_time * ((np.abs(eig_discr)-1) / np.abs(freq))
            logger.debug('Mode {:-2d}: '
                         'modal participation = {:-8.1f}%, '
                         'eigenvalue = {:-15.3f}, '
                         'vibr. freq. = {:-8.3f} Hz, '
                         'damping ratio = {:-8.3f}%'.format(iter + 1, part, eig, freq / 2 / np.pi, 100 * damp_ratio))
            formatted_DMDresult[iter, 0] = part
            formatted_DMDresult[iter, 1] = freq/2/np.pi
            formatted_DMDresult[iter, 2] = 100*damp_ratio

        # remove modes with too high or too low frequency
        # in case of a periodic signal with only a small instability, the periodic mode could have the highest
        # participation
        for row in range(formatted_DMDresult.shape[0]):
            if formatted_DMDresult[row, 1] < 0.4 or formatted_DMDresult[row, 1] > 5:
                formatted_DMDresult[row, 0] = 0
        self.DMDresult.formatted_result = formatted_DMDresult

        # Critical mode determination by looking at mode with highest participation
        self.DMDresult.DMDdamping_criticalmode = formatted_DMDresult[np.where(formatted_DMDresult[:, 0] ==
                                                                     np.max(formatted_DMDresult[:, 0])), 2][0][0]

    def plot_reconstructed_signal(self, ax, orig_timesteps, orig_signals, dmd_timesteps, reconstructed_signals):
        """
        plot reconstructed signal and original signal in axes. signals and reconstructed_signals can be both 1D or 2D
        """
        ax.plot(orig_timesteps, orig_signals, '.', label='snapshots')
        ax.plot(dmd_timesteps, reconstructed_signals, '--', label='DMD output')
        ax.set_title('original and reconstructed DMD signal')
        ax.legend()
        
    def plot_alldynamics(self, ax):
        """
        Plot the dynamics part of the DMD in axes

        Parameters
        ----------
        ax : matplotlib axes
        """
        for dynamics in self.DMDresult.hodmd.dynamics:
            ax.plot(self.time, dynamics)
        ax.set_title('DMD dynamics')
    
    def plot_modes(self):
        """ plot DMD modes and dynamics side by side """
        # TODO this should plot the modes in order of importance (participation)
        print('Left column: mode shapes / right column: dynamics')
        for mode, dynamics, eig in zip(self.DMDresult.hodmd.modes.T, self.DMDresult.hodmd.dynamics,
                                       self.DMDresult.eigs):
            fig_modes, ax = plt.subplots(1,2)
            ax[0].plot(self.radpos, mode.real)
            ax[0].set_title('Eig: {} / Freq: {} Hz'.format(eig, eig.imag/2/np.pi))
            ax[1].plot(self.time, dynamics.real)
            plt.show()

    def compute_NRMSD(self, signal1, signal2):
        """
        Computes the NRMSD between two (possibly multi-dimensional) time signals. The root mean square error is
        normalized by the amplitude (max-min) of each signal.
        Assumed that    axis=0 -> time
                        axis=1 -> multiple D signals

        Parameters
        ----------
        signal1 : array
            1D or 2D signal array
        signal2 : array
            1D or 2D signal array
        """
        logger = logging.getLogger('quexus.uq.postprocessor.dmd.compute_NRMSD')
        logger.debug('compute NRMSD DMD reconstructed signal')
        rmsd = np.sqrt(((signal1-signal2)**2).sum(axis=0) / signal1.shape[0])
        max_amplitude = signal1.max(axis=0) - signal1.min(axis=0)
        nrmsd = rmsd / max_amplitude
        self.DMDresult.nrmsd_total = nrmsd.mean()
        logger.debug('Normalized RMSD between reconstructed DMD and original signal = {}%'.format(
            self.DMDresult.nrmsd_total * 100))

    def reduce_data(self, nradpos=16, new_radpos=None):
        """
        Reduce radial positions by interpolation

        Parameters
        ----------
        nradpos : int, optional
            linear spacing of nradpos stations between min(radpos) and max(radpos), default is 16
        new_radpos : array
            1D array with new radial positions, overwrites nradpos, default is None
        """
        if new_radpos is None:
            new_radpos = np.linspace(self.radpos[0], self.radpos[-1], nradpos)
        f = interp1d(self.radpos, self.signal, axis=1, fill_value='extrapolate')
        self.signal = np.squeeze(f(new_radpos))
        self.radpos = new_radpos            

    def make_stabilisation_diagram(self):
        """
        Makes the DMD for increasing SVD rank, plots frequencies/damping/participation/NRMSD

        For debugging/understanding purposes
        """
        fig_stab_diagram = plt.figure(figsize=[14, 10])
        ax1 = fig_stab_diagram.add_subplot(311)  # svd_rank vs frequency
        ax2 = fig_stab_diagram.add_subplot(312)  # svd_rank vs frequency
        ax3 = fig_stab_diagram.add_subplot(313)  # svd_rank vs NRMSD

        if self.svd_rank == 0:
            max_svd_rank = self.DMDresult.formatted_result.shape[0]
        else:
            max_svd_rank = self.svd_rank

        for svd_rank in range(1, max_svd_rank+1):
            dummy_obj = DMDanalysis(self.time, self.sampling_time, self.signal, plotting=False, svd_rank=svd_rank)
            dummy_obj.do_full_hodmd_analysis()
            dummy_obj.compute_NRMSD(dummy_obj.signal, dummy_obj.DMDresult.hodmd.reconstructed_data.real.T)

            ax1.scatter(dummy_obj.DMDresult.formatted_result[:, 1],
                        np.ones(dummy_obj.DMDresult.formatted_result.shape[0], dtype=int)*svd_rank, color='C0',
                        s=30*dummy_obj.DMDresult.formatted_result[:, 0]/max(dummy_obj.DMDresult.formatted_result[:, 0]))
            ax2.scatter(- dummy_obj.DMDresult.formatted_result[:, 2],
                        np.ones(dummy_obj.DMDresult.formatted_result.shape[0], dtype=int)*svd_rank, color='C0',
                        s=30*dummy_obj.DMDresult.formatted_result[:, 0]/max(dummy_obj.DMDresult.formatted_result[:, 0]))
            ax3.scatter(dummy_obj.DMDresult.nrmsd_total * 100, svd_rank, color='C0')

        for ax in [ax1, ax2, ax3]:
            ax.grid(True)
            ax.set_ylabel('SVD rank')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax1.set_xlabel('Frequency / Hz')
        ax2.set_xlabel('Damping ratio / %')
        ax3.set_xlabel('NRMSD / %')

        ax2.autoscale(False)
        ax2.fill_between(x=[-100, 0], y1=0, y2=1000, facecolor='lightgrey', alpha=0.3)
        plt.tight_layout()

        plt.show()
        ax1.set_xlim([0, 5])
        ax2.set_xlim([-5, 5])
        fig_stab_diagram.savefig(os.path.join(self.result_dir, 'DMD_results', 'DMDstabilisationdiagram.pdf'))
        plt.close(fig_stab_diagram)


