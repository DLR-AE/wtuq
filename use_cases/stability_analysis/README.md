# use case: wind turbine stability analysis

## Purpose
This use case was established to compute the uncertainty of engineering beam properties of wind turbine blades on its
stability properties. A comparison between different wind turbine simulation tools was made. These tools were:

- Bladed: both linearization and time domain simulation
- HAWCStab2: linearization
- HAWC2: time domain simulation
- alaska/Wind: time domain simulation
- Simpack (+AeroDyn): time domain simulation
- OpenFAST: time domain simulation, (interface established, but no uncertainty quantification performed)

The IWT-7.5-164 RWT was used as baseline turbine. The blade stiffness was reduced to force an instability under nominal
operating conditions. This reference condition was presented at the WESC2021 conference (Code-to-code comparison of
realistic wind turbine instability phenomena, https://doi.org/10.5281/zenodo.5874658). 
The reference models in the different tools are in part available here
and in part available upon request.

A range of engineering beam properties could be used as uncertain parameters, with variability along the blade provided
by the splining routines of the framework. Two case studies with different uncertain parameters were investigated.
These results will be published at a later point. The damping of the most critical mode (edgewise BW mode) was used as
Quantity of Interest to represent the stability of the turbine. This was a direct output of the linearization models,
the time domain models needed extra postprocessing to extract the damping of the time domain signals.

The general use case workflow is as follows:

1. Definition of the uncertain parameters with the wtuq_framework routines
2. Modification of the BECAS cross section properties with the *preprocessor.py* and *libBECAS* modules and export in
different formats.
3. Setting up and running simulations in different tools, interfaces provided in the tool_interface directory.
4. Postprocessing of the time domain signals to determine the damping with the *postprocessor.py* and *libDMD* modules.
Multiple methods have been implemented in these modules.
5. Uncertainty quantification with uncertainpy

The modules in this use case are dedicated to this specific turbine, available data, tools and research question.
Nevertheless, most of the code is independent of these prerequisites and therefore more widely applicable without need
for modification.

## Structure
The main functionalities are on the top level of the directory. The sub-directories contain:
- config: use case specific config-file specifications and example config files for different tools
- reference_data: reference results for two case studies in different tools and data of the reference turbine
- result_comparison: config files for the comparison of different results with the uq_framework.uq_results_analysis
module
- tool_interfaces: interface modules (used by the StabilityAnalysisModel class) to different aeroelastic wind turbine
simulation tools

## Usage

The general usage description is given in the README file of the wtuq_framework. 

The usage of this use case is in part tool specific, because the interfaces to the wind turbine aeroelastic models 
often require specific packages and of course the required licenses/installed tools. 

| Tool            |      Instructions      |  Contact person |
|-----------------|:-------------|-------------|
| Bladed          | The Bladed interface requires the installation of the pyBladed package, which is a wrapper for the BladedAPI. This package can be installed by `pip install --user https+git://github.com/DLR-AE/pyBladed.git`. Further requirements by the BladedAPI are a 32-bit Python interpreter and administrator rights when executing the python script. Bladed has to be installed (we used v4.9), a Bladed license has to be active and the location of the Bladed API DLL is in your PYTHONPATH | hendrik.verdonck@dlr.de |
| Simpack         | We have dedicated scripts for the setup of a SIMBEAM model, the pre- and postprocessing of the simulations. They are not part of this repository but are available upon request   |  hendrik.verdonck@dlr.de |
| HAWC2/HAWCStab2 | HAWC2 has to be installed and a correct license has to be available | jelmer.polman@iwes.uni-hannover.de |
| OpenFAST        | The pyFAST package available at https://github.com/OpenFAST/python-toolbox has to be installed | jelmer.polman@iwes.uni-hannover.de |
| alaska/Wind     | Dedicated postprocessing scripts are available upon request | smueller2@nordex-online.com |

## Re-run test cases

All test cases can be recomputed starting from the uncertainpy data saved in .h5 files.

Run: `python run_analysis -c config\<tool_name>\config_<tool_name>_case_study_1.txt
                         -rh5 reference_data\<tool_name>\case_study_1\uq_results\StabilityAnalysisModel.h5`
                         
Example: `python run_analysis -c config\alaska\config_alaska_case_study_1.txt
                             -rh5 reference_data\alaska\case_study_1\uq_results\StabilityAnalysisModel.h5`


## Comparison of different runs

The UQ results of different tools/runs can be compared by running the uq_results_analysis script with a special config
file.

Example:

* set working directory: `<...>/use_cases/stability_analysis`
* Run: `python ../../wtuq_framework/uq_results_analysis.py -c ./result_comparison/config_case_study_1.txt`


## Contributing
This use case was the joint work of the DLR (Institute of Aeroelasticity), Leibniz University Hannover (Institut für
Windenergiesysteme), and Nordex GmbH. Many thanks go to: Oliver Hach, Claudio Balzani, Jelmer Polman, Otto Braun,
Sarah Müller, and Johannes Rieke
