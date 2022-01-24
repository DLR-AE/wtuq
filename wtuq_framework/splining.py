"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>, Oliver Hach <oliver.hach@dlr.de>
@date: 11.02.2020

Functionalities for NURBS curve splining or linear interpolation
"""

import numpy as np
from geomdl import BSpline, knotvector
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import uuid


class QuexusSpline(object):
    """
    Interface to translate the uncertain parameters (spline coefficients) to splines. Each spline is represented by an
    individual object.

    Parameters
    ----------
    param_subset : dict
        Dictionary with spline coefficients

    Attributes
    ----------
    spline : geomdl BSpline object
        NURBS curve representation of the spline coefficients given in param_subset

    Notes
    -----
    Example:

    param_subset = {'cp01_x': 0, 'cp01_y': 0, ...}

    The subset "cp01_x, cp01_y, cp02_y" defines a spline with 3 control points
    since defaults for the point apply. Numbering starts at 00.

    The current implementation builds the spline exclusively on the basis of the control points locations. Future
    development could include slopes at the control points.

    x=0, y=0, (a=0)
    x=cp01_x, y=cp01_y, (a=cp01_a)
    x=1, y=cp02_y, (a=0)
    """
    def __init__(self, param_subset):
        self.spline = BSpline.Curve()
        self.spline.degree = 2
        self.spline.ctrlpts = self._get_ctrlpts(param_subset)
        self.spline.knotvector = knotvector.generate(self.spline.degree, len(self.spline.ctrlpts))
        # self.plot()

    @staticmethod
    def _get_ctrlpts(param_subset):
        """
        Extracts a list of spline point parameters (x, y) from the dict.
        Current implementation only includes control point positioning. Spline gradient at control points could be a
        future improvement.

        Parameters
        ----------
        param_subset : dict
            Dictionary with spline coefficients. Example: param_subset = {'cp01_x': 0, 'cp01_y': 0, ...}

        Returns
        -------
        ctrlpts : list
            List of lists. Each value is an x, y coordinate list
        """
        ctrlpts = list()
        end_reached = False
        ctrlpt_i = -1
        while not end_reached:
            ctrlpt_i += 1

            cp_x_key = 'cp{:02d}_x'.format(ctrlpt_i)
            cp_y_key = 'cp{:02d}_y'.format(ctrlpt_i)

            # cp_x_key

            if cp_x_key not in param_subset.keys():
                end_reached = True
            else:
                cp_x = param_subset[cp_x_key]
                cp_y = param_subset[cp_y_key]
            ctrlpts.append([cp_x, cp_y])

        return ctrlpts

    def as_array(self, where):
        """
        Evaluates the spline at x=where.

        Parameters
        ----------
        where : float
            location of evaluation (0<=x<=1)

        Returns
        -------
        f(where) : float
            Spline value at <where> coordinate
        """
        basis_for_interp = np.asarray(self.spline.evaluate_list(np.linspace(0, 1, 1000).tolist()))
        f = interp1d(basis_for_interp[:, 0], basis_for_interp[:, 1], fill_value='extrapolate')
        return f(where)

    def plot(self):
        """
        Create debugging plots
        """
        evalpts = np.array(self.spline.evalpts)
        control_points = np.array(self.spline.ctrlpts)

        # plot for each spline individually
        if True:
            fig = plt.figure('Splining example')
            ax = fig.gca()
            # ax.plot(evalpts[:, 0], evalpts[:, 1], '-+', label='Spline')
            ax.plot(evalpts[:, 0], evalpts[:, 1], label='Spline')
            ax.scatter(control_points[:, 0], control_points[:, 1], color='red', marker='^', s=15.,
                       label='Control points')
            ax.set_ylim([0.85, 1.15])
            ax.legend()
            ax.grid(True)
            fig_id = uuid.uuid1()  # id containing a time attribute
            # plt.show()
            plt.savefig(r'.\spline_plots\figure_' + str(fig_id))
            plt.close()

        # make a plot with all splines together
        if False:
            fig_all_splines = plt.figure('All splines together', figsize=[20, 10])
            ax_all_splines = fig_all_splines.gca()
            ax_all_splines.plot(evalpts[:, 0], evalpts[:, 1], label='Spline')
            ax_all_splines.scatter(control_points[:, 0], control_points[:, 1], color='red', marker='^', s=15.,
                                   label='Control points')
            ax_all_splines.set_ylim([0.85, 1.15])
            ax_all_splines.grid(True)
            plt.savefig(r'.\spline_plots\all_splines_together.png')
            plt.savefig(r'.\spline_plots\all_splines_together.pdf')
            # this plot will continuously be overwritten until all splines are there -> not optimal, but
            # has the benefit that the entire plotting script is here (and not in the run_analysis or so)
            # don't close figure


class LinearInterp(object):
    """
    Interface to translate the uncertain parameters to Linear Interpolation objects. Each interpolation is represented
    by an individual object.

    Parameters
    ----------
    param_subset : dict
        Dictionary with interpolation coefficients,
        example = {'fixed_distr': 0.5, 'cp01_y_fix': 0, 'cp02_y_max': 1.1, ...}

    Attributes
    ----------
    ctrlpts : dict
        Coordinates of the control points
    linear_interp : np.interp1d object
    """

    def __init__(self, param_subset):
        self.ctrlpts = self._get_ctrlpts(param_subset)
        self.linear_interp = interp1d(self.ctrlpts['x'], self.ctrlpts['y'], fill_value='extrapolate')
        # self.plot()

    @staticmethod
    def _get_ctrlpts(param_subset):
        """
        Extracts the control points for the fixed distribution (x, y) from the dict.

        Parameters
        ----------
        param_subset : dict
            Dictionary with interpolation coefficients,
            example = {'fixed_distr': 0.5, 'cp01_y_fix': 0, 'cp02_y_max': 1.1, ...}

        Returns
        -------
        ctrlpts : dict
            Dictionary with 'x' and 'y' coordinates of control points

        """
        ctrlpts = dict()
        ctrlpts['x'] = list()
        ctrlpts['y'] = list()

        for ax in ['x', 'y']:
            end_reached = False
            ctrlpt_i = -1
            while not end_reached:
                ctrlpt_i += 1

                cp_key = 'cp{:02d}_{}'.format(ctrlpt_i, ax)

                if cp_key + '_fix' in param_subset.keys():
                    ctrlpts[ax].append(param_subset[cp_key + '_fix'])
                elif cp_key + '_min' in param_subset.keys():
                    f = interp1d([0, 1], [param_subset[cp_key + '_min'], param_subset[cp_key + '_max']],
                                 fill_value='extrapolate')
                    ctrlpts[ax].append(f(param_subset['fixed_distr']))
                else:
                    end_reached = True

        return ctrlpts

    def as_array(self, where):
        """
        Evaluates the interpolation at x=where.

        Parameters
        ----------
        where : float
            location of evaluation (0<=x<=1)

        Returns
        -------
        f(where) : float
            Interpolation value at <where> coordinate
        """
        return self.linear_interp(where)

    def plot(self):
        """
        Create debugging plots
        """
        evalpts_x = np.linspace(0, 1, 1000)
        evalpts_y = np.array(self.linear_interp(evalpts_x))

        # plot for each spline individually
        if False:
            fig = plt.figure('Linear interpolation example')
            ax = fig.gca()
            ax.plot(evalpts_x, evalpts_y, label='linear interpolation')
            ax.scatter(self.ctrlpts['x'], self.ctrlpts['y'], color='red', marker='^', s=15.,
                       label='Control points')
            ax.set_ylim([0.85, 1.15])
            ax.legend()
            ax.grid(True)
            fig_id = uuid.uuid1()  # id containing a time attribute
            plt.show()
            plt.savefig(r'.\spline_plots\figure_' + str(fig_id))
            plt.close()

        # make a plot with all splines together
        if True:
            fig_all_splines = plt.figure('All splines together', figsize=[20, 10])
            ax_all_splines = fig_all_splines.gca()
            ax_all_splines.plot(evalpts_x, evalpts_y, label='Spline')
            ax_all_splines.scatter(self.ctrlpts['x'], self.ctrlpts['y'], color='red', marker='^', s=15.,
                                   label='Control points')
            ax_all_splines.set_ylim([0.85, 1.15])
            ax_all_splines.grid(True)
            plt.savefig(r'.\spline_plots\all_splines_together.png')
            plt.savefig(r'.\spline_plots\all_splines_together.pdf')
            # this plot will continuously be overwritten until all splines are there -> not optimal, but
            # has the benefit that the entire plotting script is here (and not in the run_analysis or so)
            # don't close figure
