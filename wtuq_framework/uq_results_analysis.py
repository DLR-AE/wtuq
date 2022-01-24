"""
@author: Hendrik Verdonck <hendrik.verdonck@dlr.de>
@date: 23.08.2021

This module contains scripts for additional postprocessing of the uncertainty quantification results.
"""

import os
import numpy as np
import numpoly
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import uncertainpy as un
import pandas as pd
from configobj import ConfigObj
from validate import Validator
import argparse
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno, plasma, viridis
from bokeh.models import HoverTool
import itertools


class UQResultsAnalysis(object):
    """
    Additional plotting routines on uncertainpy UQ results. Comparison between different runs possible.

    Parameters
    ----------
    uncertainpy_results : list
        List with paths to the standard uncertainpy .h5 output files
    label_names : list
        List with legend label names corresponding to uncertainpy_results
    output_dir : str
        Output directory where figures are saved
    param_name_translation : dict
        Conversion of input uncertain parameter names into text for plots

    Attributes
    ----------
    uncertainpy_results : list
        List with paths to the standard uncertainpy .h5 output files
    run_label : dict
        Dictionary with legend name for each of the uncertainpy results
    output_dir : dict
        Output directory where figures are saved
    param_name_translation : dict
        Conversion of input uncertain parameter names into text for plots
    merged_results : dict
        Merged all results to one dictionary to serve as input for boxplots
    evaluations : dict
        Training data values for each of the uncertainpy results
    evaluations_hat : dict
        Evaluations of the surrogate model at the training data coordinates for each of the uncertainpy results
    evaluations_loo : dict
        Evaluations of the leave-one-out surrogate model at the training data coordinates for each of the uncertainpy results
    sobol_first : dict
        First order Sobol indices for each of the uncertainpy results
    sobol_total : dict
        Total Sobol indices for each of the uncertainpy results
    uncertain_param_names : dict
        Names of the uncertain parameters for each of the uncertainpy results
    uncertain_parameters : dict
        Uncertain parameter values for each of the uncertainpy results
    RMSD_evaluations : dict
        Root mean square deviation between surrogate model and training data values for each of the uncertainpy results
    NRMSD_evaluations : dict
        Normalised root mean square deviation between surrogate model and training data values for each of the uncertainpy results
    MAE_evaluations : dict
        Mean absolute error between surrogate model and training data values for each of the uncertainpy results
    RMSD_loo : dict
        Root mean square deviation between the leave-one-out surrogate model and training data values for each of the uncertainpy results
    NRMSD_loo : dict
        Normalised root mean square deviation between the leave-one-out surrogate model and training data values for each of the uncertainpy results
    MAE_loo : dict
        Mean absolute error between the leave-one-out surrogate model and training data values for each of the uncertainpy results
    """
    def __init__(self, uncertainpy_results, label_names=[], output_dir=None, param_name_translation={}):
        # check if uncertainpy result exists
        self.uncertainpy_results = list()
        self.run_label = dict()
        if len(label_names) != len(uncertainpy_results):
            print('No label names requested or length of the requested label names does not match length of uncertainpy'
                  'results. uncertainpy_results names will be used as label legend')
            label_names = uncertainpy_results

        for uq_result, label_name in zip(uncertainpy_results, label_names):
            if os.path.isfile(uq_result) is True:
                self.uncertainpy_results.append(uq_result)
                self.run_label[uq_result] = label_name
            else:
                print(uq_result, 'can not be found')
        self.param_name_translation = param_name_translation
        if output_dir is None:
            if len(uncertainpy_results) == 1:
                self.output_dir = os.path.split(uncertainpy_results[0])[0]
            else:
                self.output_dir = '.'
        else:
            self.output_dir = output_dir

        # data which can be extracted from uncertainpy simulations
        self.evaluations = dict()
        self.evaluations_hat = dict()
        self.evaluations_loo = dict()
        self.sobol_first = dict()
        self.sobol_total = dict()
        self.uncertain_param_names = dict()
        self.uncertain_parameters = dict()
        self.RMSD_evaluations = dict()
        self.NRMSD_evaluations = dict()
        self.MAE_evaluations = dict()
        self.RMSD_loo = dict()
        self.NRMSD_loo = dict()
        self.MAE_loo = dict()

    def read_results(self, model_name='Model'):
        """
        Read all uncertainpy results from the uncertainpy results files specified in the self.uncertainpy_results
        attribute.

        Parameters
        ----------
        model_name : string, optional
            Name of the feature which should be found in the uncertainpy.h5 file.
            Default is 'Model'

        Raises
        ------
        OSError
            If a requested uncertainpy .h5 file does not exist
        """
        for uq_result in self.uncertainpy_results:
            # get info from uncertainpy .h5 file
            try:
                uq_results = self.get_UQ_results(uq_result)

                # get input nodes and training data evaluations
                self.evaluations[uq_result] = uq_results.data[model_name].evaluations[~np.isnan(uq_results.data[model_name].evaluations)]
                self.uncertain_parameters[uq_result] = uq_results.data[model_name].nodes
                # get surrogate model evaluations and verification data
                if uq_results.data[model_name].evaluations_hat.any():
                    self.evaluations_hat[uq_result] = uq_results.data[model_name].evaluations_hat
                    self.evaluations_loo[uq_result] = uq_results.data[model_name].evaluations_loo
                    self.RMSD_evaluations[uq_result] = uq_results.data[model_name].RMSD_evaluations
                    self.NRMSD_evaluations[uq_result] = uq_results.data[model_name].NRMSD_evaluations
                    self.MAE_evaluations[uq_result] = uq_results.data[model_name].MAE_evaluations
                    self.RMSD_loo[uq_result] = uq_results.data[model_name].RMSD_loo
                    self.NRMSD_loo[uq_result] = uq_results.data[model_name].NRMSD_loo
                    self.MAE_loo[uq_result] = uq_results.data[model_name].MAE_loo
                # get sobol values
                self.sobol_first[uq_result] = uq_results.data[model_name].sobol_first
                self.sobol_total[uq_result] = uq_results.data[model_name].sobol_total

                # get uncertain param names and translate if desired
                self.uncertain_param_names[uq_result] = []
                for param in uq_results.uncertain_parameters:
                    if param in self.param_name_translation:
                        self.uncertain_param_names[uq_result].append(self.param_name_translation[param])
                    else:
                        self.uncertain_param_names[uq_result].append(param)

            except OSError:
                print(uq_result, ' file not found')
                raise

    def merge_results(self):
        """
        Merge the evaluations and sobol indices of multiple runs together to generate data for boxplot presentation
        """
        self.merged_results = dict()
        self.merged_results['evaluations'] = np.vstack([self.evaluations[x] for x in self.evaluations])
        self.merged_results['sobol_first'] = np.vstack([self.sobol_first[x] for x in self.sobol_first])
        self.merged_results['sobol_total'] = np.vstack([self.sobol_total[x] for x in self.sobol_total])

    @staticmethod
    def get_UQ_results(uq_h5_file):
        """
        load the Model Data object from uncertainpy

        Parameters
        ----------
        uq_h5_file : string
            Path to standard output file of the uq framework (saved as a .h5 file)

        Returns
        -------
        data : un.Data
            Parsed data object
        """
        data = un.Data()
        data.load(uq_h5_file)
        return data

    @staticmethod
    def redo_UQ_plotting(uq_h5_file, output_dir):
        """
        Redo the standard UQ plots and save to output_dir

        Parameters
        ----------
        uq_h5_file : string
            Path to standard output file of the uq framework (saved as a .h5 file)
        output_dir : string
            Path to output directory
        """
        uq_plotting = un.plotting.PlotUncertainty(filename=uq_h5_file, folder=output_dir)
        uq_plotting.plot_all()

    def show_input_parameters_per_iter(self):
        """
        Show the uq uncertain parameters per iteration.
        """
        fig = plt.figure('input_parameters')
        ax = fig.gca()
        for uq_result in self.uncertainpy_results:
            for params, names in zip(self.uncertain_parameters[uq_result], self.uncertain_param_names[uq_result]):
                  ax.plot(params, label=self.run_label[uq_result]+' - '+names)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Input value')
        fig.savefig(os.path.join(self.output_dir, 'input_parameters.png'), bbox_inches='tight')
        plt.close(fig)
    
    def compare_QoI_evaluations_per_iter(self):
        """
        Compare the QoI evaluations of two or more simulations.
        """
        # fig 1: QoI evaluations per iteration
        fig1 = plt.figure('Comparison QoI evaluations')
        ax = fig1.gca()
        for uq_result in self.evaluations:
            ax.plot(self.evaluations[uq_result], '-*', markersize=10, label=self.run_label[uq_result])
        ax.grid(True)
        ax.set_xlabel('Framework iteration')
        ax.set_ylabel('QoI evaluation')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig1.savefig(os.path.join(self.output_dir, 'QoI_evaluations.png'), bbox_inches='tight', dpi=300)
        plt.close(fig1)

        # fig 2: de-meaned QoI evaluations per iteration
        fig2 = plt.figure('DE-MEANED: Comparison QoI evaluations')
        ax2 = fig2.gca()
        for uq_result in self.evaluations:
            ax2.plot((self.evaluations[uq_result][~np.isnan(self.evaluations[uq_result])] - np.mean(np.array(self.evaluations[uq_result][~np.isnan(self.evaluations[uq_result])]))) / np.mean(np.array(self.evaluations[uq_result][~np.isnan(self.evaluations[uq_result])])) * 100, '-*',
                     markersize=10, label=self.run_label[uq_result]+' - DE-MEANED')
        ax2.grid(True)
        ax2.set_xlabel('Framework iteration')
        ax2.set_ylabel('(QoI - mean(QoI)/mean(QoI) [%]')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        fig2.savefig(os.path.join(self.output_dir, 'QoI_evaluations_demeaned.png'), bbox_inches='tight', dpi=300)
        plt.close(fig2)

    def surrogate_model_verification(self):
        """
        Plots with verification of the surrogate model fit
        1) direct comparison training QoI evaluations vs (leave-one-out) surrogate QoI evaluations
        2) RMSD/NRMSD/MAE error metrics between training data and (leave-one-out) surrogate model evaluations
        """
        fig = plt.figure('Comparison training QoI evaluations vs. surrogate model evaluations vs. leave-one-out surrogate model evaluations', figsize=(12, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for uq_result in self.evaluations:
            if uq_result in self.evaluations_hat:
                ax1.scatter(self.evaluations[uq_result], self.evaluations_hat[uq_result],
                            label=self.run_label[uq_result])
                ax2.scatter(self.evaluations[uq_result], self.evaluations_loo[uq_result],
                            label=self.run_label[uq_result])
        ax1.set_ylabel('Surrogate Model evaluations')
        ax2.set_ylabel('Leave-one-out surrogate model evaluations')
        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.set_xlabel('Model evaluations (training data)')
            ax.legend(loc='upper left')
            ax.set_axisbelow(True)
            # add diagonal x=y line on plot
            ax.autoscale(False)
            ax.plot([-10, 10], [-10, 10], color='C3')
        fig.savefig(os.path.join(self.output_dir, 'surrogate_model_verif.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        fig = plt.figure('Comparison training evaluations vs. surrogate model evaluations', figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        for uq_result in self.evaluations:
            if uq_result in self.evaluations_hat:
                ax1.scatter(self.evaluations[uq_result], self.evaluations_hat[uq_result],
                            label=self.run_label[uq_result])
        ax1.set_ylabel('Surrogate Model evaluations')
        ax1.grid(True)
        ax1.set_xlabel('Model evaluations (training data)')
        ax1.legend(loc='upper left')
        ax1.set_axisbelow(True)
        # add diagonal x=y line on plot
        ax1.autoscale(False)
        ax1.plot([-10, 10], [-10, 10], color='C3')
        fig.savefig(os.path.join(self.output_dir, 'surrogate_model_verif_only_surrogate.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        fig = plt.figure('Comparison training evaluations vs. leave-one-out surrogate model evaluations', figsize=(6, 4))
        ax2 = fig.add_subplot(111)
        for uq_result in self.evaluations:
            if uq_result in self.evaluations_hat:
                ax2.scatter(self.evaluations[uq_result], self.evaluations_loo[uq_result],
                            label=self.run_label[uq_result])
        ax2.set_ylabel('Leave-one-out surrogate model evaluations')
        ax2.grid(True)
        ax2.set_xlabel('Model evaluations (training data)')
        ax2.legend(loc='upper left')
        ax2.set_axisbelow(True)
        # add diagonal x=y line on plot
        ax2.autoscale(False)
        ax2.plot([-10, 10], [-10, 10], color='C3')
        fig.savefig(os.path.join(self.output_dir, 'surrogate_model_verif_only_loo.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        for name, error_evaluations, error_loo in zip(['NRMSD', 'RMSD', 'MAE'],
                                                      [self.NRMSD_evaluations, self.RMSD_evaluations, self.MAE_evaluations],
                                                      [self.NRMSD_loo, self.RMSD_loo, self.MAE_loo]):
            fig = plt.figure(name)
            ax = fig.gca()
            error_norm = dict()  # key => legend label
            for uq_result in self.sobol_first:
                error_norm[self.run_label[uq_result]] = [error_evaluations[uq_result], error_loo[uq_result]]
            index = ['Surrogate model', 'Leave-one-out surrogate model']
            df1 = pd.DataFrame(error_norm, index=index)
            # useful settings: rot -> rotation xlabel, color={"<label1>": "green", "<label2>": "red"}
            df1.plot.bar(ax=ax, rot=0)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y')
            ax.set_axisbelow(True)
            ax.set_ylabel('{} surrogate model vs. training data / %'.format(name))
            fig.savefig(os.path.join(self.output_dir, '{}.png'.format(name)), bbox_inches='tight', dpi=300)
            plt.close(fig)

        fig = plt.figure('Comparison QoI evaluations training vs. surrogate vs. LOO surrogate', figsize=(10, 4))
        ax = fig.gca()
        for uq_result in self.evaluations:
            ax.plot(self.evaluations[uq_result], 'o', markersize=10, label=self.run_label[uq_result])
            if uq_result in self.evaluations_hat:
                ax.plot(self.evaluations_hat[uq_result], '*', markersize=10, label='surrogate model ' + self.run_label[uq_result])
            if uq_result in self.evaluations_loo:
                ax.plot(self.evaluations_loo[uq_result], '+', markersize=10, label='leave-one-out surrogate model ' + self.run_label[uq_result])
        ax.grid(True)
        ax.set_xlabel('Framework iteration')
        ax.set_ylabel('QoI evaluation')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        fig.savefig(os.path.join(self.output_dir, 'QoI_evaluations_surrogate_model.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

    def compare_sobol_indices(self):
        """
        Compare sobol indices of multiple framework runs with each other. Assumed that those runs are made with the
        same uncertain parameter sets. Bar plot made with pandas.
        """
        fig = plt.figure('Comparison Sobol Indices')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        sobol_first_pd_dict = dict()  # key => legend label
        sobol_total_pd_dict = dict()  # key => legend label
        for uq_result in self.sobol_first:
            sobol_first_pd_dict[self.run_label[uq_result]] = self.sobol_first[uq_result]
            sobol_total_pd_dict[self.run_label[uq_result]] = self.sobol_total[uq_result]

        # assumed that all uncertain_param_names are the same, so just last uq_result used
        index = self.uncertain_param_names[uq_result]
        df1 = pd.DataFrame(sobol_first_pd_dict, index=index)
        df1.plot.bar(ax=ax1, rot=0)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        df2 = pd.DataFrame(sobol_total_pd_dict, index=index)
        df2.plot.bar(ax=ax2, rot=0, legend=False)

        ax1.set_ylabel('First Order Sobol Index')
        ax1.grid(axis='y')
        ax1.set_axisbelow(True)
        ax2.set_ylabel('Total Sobol Index')
        ax2.grid(axis='y')
        ax2.set_axisbelow(True)
        fig.savefig(os.path.join(self.output_dir, 'comparison_sobol_indices.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

    def compare_QoI_evals_per_iter_bokeh(self):
        """
        Make interactive bokeh plots of:
        - QoI evaluations per iteration
        - de-meaned QoI evaluations per iteration.
        - RMSD/NRMSD error metrics
        - comparison of QoI evaluations and (leave-one-out) surrogate model evaluations
        """
        colors = plasma(len(self.evaluations))  # other possible color palettes: inferno, plasma, viridis
        # todo: the HoverTool model has to be initialized separately for each plot, I don't know how to make it cleaner
        hover = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])
        hover2 = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])
        hover3 = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])
        hover4 = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])
        hover5 = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])
        hover6 = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])
        hover7 = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])
        hover8 = HoverTool(tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)")])

        #####################################
        # fig 1: QoI evaluations per iteration
        #####################################
        p = figure(plot_width=1800, plot_height=900, title='Comparison QoI evaluations')
        p.add_tools(hover)
        for uq_result, color in zip(self.evaluations, colors):
            p.line(np.arange(len(self.evaluations[uq_result])), self.evaluations[uq_result],
                   color=color, legend_label=self.run_label[uq_result], line_width=2)
        p.add_layout(p.legend[0], 'right')  # get legend from plot and put it beside plot, legend in plot -> p.legend.location = "right"
        p.legend.click_policy = "hide"  # hide line if legend clicked
        if len(self.evaluations) > 35:
            self.bokeh_legend_as_small_as_possible(p)
        p.xaxis.axis_label = 'Framework iteration'
        p.yaxis.axis_label = 'QoI evaluation'
        output_file(os.path.join(self.output_dir, 'QoI_evaluations.html'))
        show(p)

        ################################################################
        # fig 2: de-meaned QoI evaluations per iteration
        ################################################################
        p = figure(plot_width=1800, plot_height=900, title='RELATIVE DE-MEANED: Comparison QoI evaluations')
        p.add_tools(hover2)
        for uq_result, color in zip(self.evaluations, colors):
            p.line(np.arange(len(self.evaluations[uq_result])),
                   (self.evaluations[uq_result] - np.mean(
                       np.array(self.evaluations[uq_result]))) / abs(
                       np.mean(np.array(self.evaluations[uq_result]))) * 100,
                   color=color, legend_label=self.run_label[uq_result], line_width=2)
        p.add_layout(p.legend[0], 'right')
        p.legend.click_policy = "hide"
        if len(self.evaluations) > 35:
            self.bokeh_legend_as_small_as_possible(p)
        p.xaxis.axis_label = 'Framework iteration'
        p.yaxis.axis_label = '(QoI - mean(QoI)/mean(QoI) [%]'
        output_file(os.path.join(self.output_dir, 'QoI_evaluations_relative_meaned.html'))
        show(p)

        p = figure(plot_width=1800, plot_height=900, title='ABSOLUTE DE-MEANED: Comparison QoI evaluations')
        p.add_tools(hover3)
        for uq_result, color in zip(self.evaluations, colors):
            p.line(np.arange(len(self.evaluations[uq_result])),
                   self.evaluations[uq_result] - np.mean(
                       np.array(self.evaluations[uq_result])),
                   color=color, legend_label=self.run_label[uq_result], line_width=2)
        p.add_layout(p.legend[0], 'right')
        p.legend.click_policy = "hide"
        if len(self.evaluations) > 35:
            self.bokeh_legend_as_small_as_possible(p)
        p.xaxis.axis_label = 'Framework iteration'
        p.yaxis.axis_label = 'QoI - mean(QoI) [%]'
        output_file(os.path.join(self.output_dir, 'QoI_evaluations_absolute_meaned.html'))
        show(p)

        #########################################
        # fig 3/4: NRMSD for each surrogate model
        #########################################
        plot_surro_bool = False
        p = figure(plot_width=1800, plot_height=900, title='NRMSD surrogate model')
        p.add_tools(hover4)
        for i, (uq_result, color) in enumerate(zip(self.evaluations, colors)):
            if uq_result in self.NRMSD_evaluations:
                plot_surro_bool = True
                p.diamond_cross(i, self.NRMSD_evaluations[uq_result],
                                color=color, size=20, legend_label=self.run_label[uq_result])
        if plot_surro_bool:
            p.add_layout(p.legend[0], 'right')  # get legend from plot and put it beside plot, legend in plot -> p.legend.location = "right"
            p.legend.click_policy = "hide"  # hide line if legend clicked
            if len(self.evaluations) > 35:
                self.bokeh_legend_as_small_as_possible(p)
            p.xaxis.axis_label = 'iteration'
            p.yaxis.axis_label = 'NRMSD in %'
            output_file(os.path.join(self.output_dir, 'NRMSD_evaluations.html'))
            show(p)

        plot_surro_bool = False
        p = figure(plot_width=1800, plot_height=900, title='RMSD')
        p.add_tools(hover5)
        for i, (uq_result, color) in enumerate(zip(self.evaluations, colors)):
            if uq_result in self.NRMSD_loo:
                plot_surro_bool = True
                p.diamond_cross(i, self.NRMSD_loo[uq_result],
                                color=color, size=20, legend_label=self.run_label[uq_result])
        if plot_surro_bool:
            p.add_layout(p.legend[0], 'right')  # get legend from plot and put it beside plot, legend in plot -> p.legend.location = "right"
            p.legend.click_policy = "hide"  # hide line if legend clicked
            if len(self.evaluations) > 35:
                self.bokeh_legend_as_small_as_possible(p)
            p.xaxis.axis_label = 'iteration'
            p.yaxis.axis_label = 'Leave-one.out NRMSD in %'
            output_file(os.path.join(self.output_dir, 'NRMSD_leave_one_out.html'))
            show(p)

        ############################################################
        # fig 5/6: Training data QoI evaluations vs. surrogate model QoI evaluations
        ############################################################
        plot_surro_bool = False
        p = figure(plot_width=1800, plot_height=900, title='training data vs. surrogate model')
        p.add_tools(hover6)
        for uq_result, color in zip(self.evaluations_hat, colors):
            plot_surro_bool = True
            p.circle(self.evaluations[uq_result], self.evaluations_hat[uq_result],
                     color=color, size=20, legend_label=self.run_label[uq_result])
        if plot_surro_bool:
            p.add_layout(p.legend[0], 'right')  # get legend from plot and put it beside plot, legend in plot -> p.legend.location = "right"
            p.legend.click_policy = "hide"  # hide line if legend clicked
            if len(self.evaluations) > 35:
                self.bokeh_legend_as_small_as_possible(p)
            p.xaxis.axis_label = 'training QoI evaluation'
            p.yaxis.axis_label = 'surrogate model QoI evaluation'
            output_file(os.path.join(self.output_dir, 'surrogate_evaluations.html'))
            show(p)

        plot_surro_bool = False
        p = figure(plot_width=1800, plot_height=900, title='training data vs. leave-one-out surrogate model')
        p.add_tools(hover7)
        for uq_result, color in zip(self.evaluations_loo, colors):
            plot_surro_bool = True
            p.circle(self.evaluations[uq_result], self.evaluations_loo[uq_result],
                     color=color, size=20, legend_label=self.run_label[uq_result])
        if plot_surro_bool:
            p.add_layout(p.legend[0], 'right')  # get legend from plot and put it beside plot, legend in plot -> p.legend.location = "right"
            p.legend.click_policy = "hide"  # hide line if legend clicked
            if len(self.evaluations) > 35:
                self.bokeh_legend_as_small_as_possible(p)
            p.xaxis.axis_label = 'training QoI evaluation'
            p.yaxis.axis_label = 'leave-one-out surrogate model QoI evaluation'
            output_file(os.path.join(self.output_dir, 'surrogate_evaluations_loo.html'))
            show(p)

        ############################################################
        # fig 8: Training data evaluation vs. surrogate model evaluation
        ############################################################
        plot_surro_bool = False
        p = figure(plot_width=1800, plot_height=900, title='training data/surrogate model/LOO surrogate model vs. iteration')
        p.add_tools(hover8)
        for uq_result, color in zip(self.evaluations_hat, colors):
            plot_surro_bool = True
            p.circle(np.arange(len(self.evaluations[uq_result])), self.evaluations[uq_result],
                     color=color, size=20, legend_label=self.run_label[uq_result], alpha=0.5)
            p.star(np.arange(len(self.evaluations[uq_result])), self.evaluations_hat[uq_result],
                   color=color, size=20, legend_label='surrogate model ' + self.run_label[uq_result])
            p.cross(np.arange(len(self.evaluations[uq_result])), self.evaluations_loo[uq_result],
                    color=color, size=20, legend_label='leave-one-out surrogate model ' + self.run_label[uq_result])
        if plot_surro_bool:
            p.add_layout(p.legend[0], 'right')  # get legend from plot and put it beside plot, legend in plot -> p.legend.location = "right"
            p.legend.click_policy = "hide"  # hide line if legend clicked
            if len(self.evaluations) > 35:
                self.bokeh_legend_as_small_as_possible(p)
            p.xaxis.axis_label = 'iteration'
            p.yaxis.axis_label = 'QoI evaluation'
            output_file(os.path.join(self.output_dir, 'QoI_evaluations_surrogate_model.html'))
            show(p)

    @staticmethod
    def bokeh_legend_as_small_as_possible(p):
        """
        Set the margins in the bokeh plot as small as possible (still readable) 

        Parameters
        ----------
        p : bokeh plot
        """
        print('Bokeh plot legend adjusted to include as many entries as possible')
        p.legend.label_text_font_size = '6pt'
        p.legend.spacing = -7
        p.legend.padding = 0
        p.legend.margin = 0

    def compare_QoI_evaluations_per_iter_boxplot(self):
        """
        Plot the QoI evaluations of the different uq_results in a boxplot per iteration (can f.ex. be used to see the
        spread of different postprocessing runs).
        merge_results() has to be executed earlier.
        """
        if hasattr(self, 'merged_results') is False:
            self.merge_results()
        # fig: QoI evaluations per iteration - boxplot
        fig = plt.figure('Comparison QoI evaluations - boxplot', figsize=(14, 7), dpi=300)
        ax = fig1.gca()
        ax.boxplot(self.merged_results['evaluations'])
        ax.grid(True)
        ax.set_xlabel('Framework iteration')
        ax.set_ylabel('QoI evaluation')
        fig1.savefig(os.path.join(self.output_dir, 'QoI_evaluations_boxplot.png'), bbox_inches='tight')
        plt.close(fig)

    def compare_sobol_indices_boxplot(self):
        """
        Plot the sobol results of the different rundirs in a boxplot (can f.ex. be used to see the
        spread of different postprocessing runs).
        merge_results() has to be executed earlier.
        """
        if hasattr(self, 'merged_results') is False:
            self.merge_results()
        fig = plt.figure('Comparison Sobol Indices - boxplot', dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # assumed that all uncertain_param_names are the same, so just take random uncertainpy result
        index = self.uncertain_param_names[self.uncertainpy_results[0]]

        ax1.boxplot(self.merged_results['sobol_first'])
        ax2.boxplot(self.merged_results['sobol_total'])

        ax1.set_ylabel('First Order Sobol Index')
        ax1.grid()
        ax1.set_xticklabels(index)
        ax2.set_ylabel('Total Sobol Index')
        ax2.grid()
        ax2.set_xticklabels(index)

        fig.savefig(os.path.join(self.output_dir, 'comparison_sobol_indices_boxplot.png'), bbox_inches='tight')
        plt.close(fig)

    def plot_surface(self, U_hat, distribution, model_name='Model'):
        """
        Plot 2D (line) and 3D (surface) representation of the PCE polynomial.

        2D plots: polynomial is sampled for each uncertain parameter separately. All other uncertain parameters are
        set to their mean value (1st statistical moment)
        3D plots: polynomial is sampled for each combination of two uncertain parameters. All other uncertain parameters
        are set to their mean value (1st statistical moment)

        Information on statistical moments:

        zeroth order = area under pdf -> always 1
        first order = mean
        second central moment = variance

        chaospy distributions (cp.Distribution.mom()) gives the raw, uncentralized moments. So to get the variance, the
        second moment has to be centralized by var = mom([2]) - mean ** 2

        It is not trivial to determine where to sample the polynomial:
        - uniform distribution : at lower and upper bounds of the distribution?
        - normal distribution : at mean +- variance? -> lower and upper bounds are very small and big
        - between minimum and maximum training nodes?
        -> first order plot shows a comparison between sampling between min. and max. training nodes and a
        quasi-random sampling of the full input distribution range (with a huge amount of samples)

        Parameters
        ----------
        U_hat : numpoly
            PCE polynomial object
        distribution : chaospy.Distribution
            distributions belonging to the uncertain parameter input of U_hat
        model_name : string
            name of the model saved in the uncertainpy data
        """
        # USER SETTINGS
        with_shading = True
        colormap = 'magma'  # , 'inferno', 'viridis', "autumn_r", "RdYlBu"
        nr_samples_plot_1D = 100000  # dense sampling (can not be used for 2D surface plots)
        nr_samples_plot_2D = 100  # sampling for 2D surface plot

        uncertain_param_names = self.uncertain_param_names[list(self.uncertain_param_names.keys())[0]]

        fig = plt.figure('First order effects')
        ax = fig.gca()
        fig2 = plt.figure('First order effects - scaled to -1 - 1')
        ax2 = fig2.gca()

        for iter, (uncertain_param_name, distr) in enumerate(zip(uncertain_param_names, distribution)):
            lower_bound = min(self.uncertain_parameters[self.uncertainpy_results[0]][iter, :])
            upper_bound = max(self.uncertain_parameters[self.uncertainpy_results[0]][iter, :])

            # quasi-random samples over the full input distribution range
            random_samples = distr.sample(nr_samples_plot_1D)
            # array bounded within training data points where polynomial will be sampled
            samples_plot = np.linspace(lower_bound, upper_bound, nr_samples_plot_1D)
            # simple normalization of the data to a range between -1 to 1, to ease a comparison between uncertain params
            scale_to_minus1_plus1 = np.linspace(-1, 1, nr_samples_plot_1D)

            # mean values of all input distributions
            sample_means = [distr.mom([1])[0] for distr in distribution]
            # replace mean value of this iter with the sample points
            sample_means[iter] = random_samples
            evaluated_polynom = numpoly.call(U_hat[model_name], sample_means)
            ax.plot(random_samples, evaluated_polynom, '.', label=uncertain_param_name + ': quasi-random sampling full input distribution range')

            sample_means[iter] = samples_plot
            evaluated_polynom = numpoly.call(U_hat[model_name], sample_means)
            ax.plot(samples_plot, evaluated_polynom, label=uncertain_param_name + ': sampling within bounds training data')

            ax2.plot(scale_to_minus1_plus1, evaluated_polynom, label=uncertain_param_name)

        for axx in [ax, ax2]:
            axx.grid()
            axx.legend()

        fig.savefig(os.path.join(self.output_dir, 'first_order_effects.png'), bbox_inches='tight')
        fig2.savefig(os.path.join(self.output_dir, 'first_order_effects_scaled_to_-1_to_1.png'), bbox_inches='tight')
        # plt.show()

        for uncertain_param_name, distr, iters in zip(itertools.combinations(uncertain_param_names, 2),
                                                      itertools.combinations(distribution, 2),
                                                      itertools.combinations(range(len(distribution)), 2)):
            # mean values of all input distributions
            sample_means = [distr.mom([1])[0] for distr in distribution]

            samples_plot = []
            for dd, iter in zip(distr, iters):
                # lower_bound = dd.lower[0]
                # upper_bound = dd.upper[0]
                lower_bound = min(self.uncertain_parameters[self.uncertainpy_results[0]][iter, :])
                upper_bound = max(self.uncertain_parameters[self.uncertainpy_results[0]][iter, :])

                # array where polynomial will be sampled
                samples_plot.append(np.linspace(lower_bound, upper_bound, nr_samples_plot_2D))

            grid = np.meshgrid(samples_plot[0], samples_plot[1])

            # replace mean value of this iter with the sample points
            sample_means[iters[0]] = grid[0]
            sample_means[iters[1]] = grid[1]

            evaluated_polynom = numpoly.call(U_hat[model_name], sample_means)

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection="3d")
            ax2.set_xlabel(uncertain_param_name[0])
            ax2.set_ylabel(uncertain_param_name[1])

            if with_shading:
                ls = LightSource(azdeg=0, altdeg=65)
                # Shade data, creating an rgb array.
                rgb = ls.shade(evaluated_polynom, cmap=plt.get_cmap(colormap) )
                ax2.plot_surface(grid[0], grid[1], evaluated_polynom, cmap=colormap, rstride=1, cstride=1,
                                 antialiased=False, facecolors=rgb, zorder=-1)
            else:
                ax2.plot_surface(grid[0], grid[1], evaluated_polynom, cmap=colormap, rstride=1, cstride=1,
                                 antialiased=False, shade=True, zorder=-1)

            ax2.contour(grid[0], grid[1], evaluated_polynom, 12, colors="k", linestyles="solid")

            offset = np.min(evaluated_polynom) - (np.max(evaluated_polynom) - np.min(evaluated_polynom)) * 0.1
            ax2.contour(grid[0], grid[1], evaluated_polynom, 12, cmap=colormap, linestyles="solid",
                        offset=offset)
            ax2.set_zlim(bottom=offset)

            ax2.grid(False)

            fig2.savefig(os.path.join(self.output_dir,
                                      'surface_{}_{}.png'.format(uncertain_param_name[0], uncertain_param_name[1])),
                         dpi=300)
            ax2.azim = 30
            fig2.savefig(os.path.join(self.output_dir,
                                      'surface_{}_{}_2.png'.format(uncertain_param_name[0], uncertain_param_name[1])),
                         dpi=300)
            # plt.show()

    def plot_distributions(self, U_hat, distribution, model_name='Model'):
        """
        Plot input distributions and QoI distributions depending on each input parameter individually, on combinations
        of multiple input parameters and on the combination of all input parameters.

        Parameters
        ----------
        U_hat : numpoly
            PCE polynomial object
        distribution : chaospy.Distribution
            distributions belonging to the uncertain parameter input of U_hat
        model_name : string
            name of the model saved in the uncertainpy data
        """
        # SETTINGS
        nr_samples = 100000
        only_show_within_training_samples = False
        output_distribution_same_xlimits = False

        uncertain_param_names = self.uncertain_param_names[list(self.uncertain_param_names.keys())[0]]

        # quasi-random sampling
        random_samples = distribution.sample(nr_samples)

        if only_show_within_training_samples:
            lower_bound = np.min(self.uncertain_parameters[self.uncertainpy_results[0]], axis=1)
            upper_bound = np.max(self.uncertain_parameters[self.uncertainpy_results[0]], axis=1)
            to_remove = list()
            for iter, sample in enumerate(random_samples.T):
                if np.all(np.greater_equal(sample, lower_bound)) and np.all(np.less_equal(sample, upper_bound)):
                    continue
                else:
                    to_remove.append(iter)
            print('{} out of {} random samples for the distribution plot are removed, because they fall outside of the '
                  'range of training data'.format(len(to_remove), nr_samples))
            random_samples = np.delete(random_samples, to_remove, axis=1)

        fig = plt.figure('Full output distribution')
        ax = fig.gca()
        evaluated_polynom = numpoly.call(U_hat[model_name], random_samples)
        ax.hist(evaluated_polynom, alpha=0.5, bins=100, label='distr. sampling', density=True)
        xlimit = ax.get_xlim()
        fig.savefig(os.path.join(self.output_dir, 'full_output_distribution.png'), dpi=300)

        fig = plt.figure('Distributions')
        nr_uncertain_param = len(distribution)

        for iter, (uncertain_param_name, distr) in enumerate(zip(uncertain_param_names, distribution)):
            ax_input_distr = fig.add_subplot(nr_uncertain_param, 2, iter * 2 + 1)
            ax_output_distr = fig.add_subplot(nr_uncertain_param, 2, iter * 2 + 2)

            x_range_pdf = np.linspace(np.min(random_samples[iter, :]), np.max(random_samples[iter, :]), 100)  # 100 samples for plot
            pdf_y = distr.pdf(x_range_pdf)
            ax_input_distr.hist(random_samples[iter, :], alpha=0.5, bins=100, label='quasi-random sampling',
                                density=True)
            ax_input_distr.plot(x_range_pdf, pdf_y, label='pdf: ' + uncertain_param_name)

            sample_means = [distr.mom([1])[0] for distr in distribution]
            # replace mean value of this iter with the sample points
            sample_means[iter] = random_samples[iter, :]
            evaluated_polynom = numpoly.call(U_hat[model_name], sample_means)
            ax_output_distr.hist(evaluated_polynom, alpha=0.5, bins=100, label='quasi-random sampled damping distribution', density=True)
            ax_input_distr.legend(loc='lower left')
            ax_output_distr.legend(loc='lower left')

            if output_distribution_same_xlimits:
                ax_output_distr.set_xlim(xlimit)

        fig.savefig(os.path.join(self.output_dir, 'partial_contributions_to_output_distribution.png'), dpi=300)
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare UQ Framework results')
    parser.add_argument('-c', '--config', metavar='cfg', type=str, nargs='?',
                        help='Config file for comparison plots', default='config.txt')
    args = parser.parse_args()

    config_spec = ConfigObj(infile=os.path.join(os.path.realpath(os.path.split(__file__)[0]),
                                                'uq_results_analysis_config.spec'),
                            interpolation=False,
                            list_values=False,
                            _inspec=True,
                            file_error=True)

    config = ConfigObj(args.config, configspec=config_spec)
    validation_result = config.validate(Validator())
    if validation_result is not True:
        raise ValueError('Validation failed (result: {0})'.format(validation_result))

    uncertainpy_results = [os.path.join(config['base_directory'], uq_result) for uq_result in config['uncertainpy_results']]
    os.makedirs(config['output_dir'], exist_ok=True)
    uq_plots = UQResultsAnalysis(uncertainpy_results,
                                 label_names=config['label_names'],
                                 output_dir=config['output_dir'],
                                 param_name_translation=config['param_name_translation'])
    uq_plots.read_results(model_name=config['model_name'])
    uq_plots.show_input_parameters_per_iter()
    uq_plots.compare_sobol_indices()
    uq_plots.compare_QoI_evaluations_per_iter()
    uq_plots.surrogate_model_verification()
    if config['bokeh_plot'] is True:
        uq_plots.compare_QoI_evals_per_iter_bokeh()

    # extra boxplots on QoI evaluations and sobol indices, can be interesting when comparing postprocessing runs
    if config['boxplot_plot'] is True:
        uq_plots.merge_results()
        uq_plots.compare_QoI_evaluations_per_iter_boxplot()
        uq_plots.compare_sobol_indices_boxplot()
