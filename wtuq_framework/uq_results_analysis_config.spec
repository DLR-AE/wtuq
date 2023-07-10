
# base dir for results (absolute or relative to cwd)
base_directory = string(default='./reference_data')

# uncertainpy results (standard .h5 output files) which have to be compared, relative to reference_data directory.
uncertainpy_results = list(default=list())

# uncertainpy model name, name of the model object used to run the uncertainty quantification in uncertainpy
model_name = string(default='Model')

# label name for in the legends of the plots
# list needs to have same length as uncertainpy results
label_names = list(default=list())

# predefined colors for each of the datasets
# list needs to have same length as uncertainpy results
predefined_colors = list(default=list())

# name of output directory, relative to working directory
output_dir = string(default='results')

# flag for interactive bokeh plot for damping per iteration
bokeh_plot = boolean(default=False)

# flag for boxplot plots for the damping per iteration and the sobol indices
# this could be an interesting plot to show the influence/variation of different postprocessing settings on the results
boxplot_plot = boolean(default=False)

# uncertain parameter name translation
# f.ex. flap_stiffness_cp01_y -> Flap Stiffness, edge_stiffness_cp01_y -> Edge Stiffness
[param_name_translation]
__many__ = string()