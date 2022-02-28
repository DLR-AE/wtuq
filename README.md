# wtuq (Wind Turbine Uncertainty Quantification framework)

## Purpose
This Uncertainty Quantification (UQ) framework is a wrapper around the open-source uncertainpy package.
It provides extensions and interfaces especially useful for the implementation of UQ in aeroelastic
wind turbine simulations.

## Structure
The framework consists of two main parts. wtuq_framework is an installable package which is independent of any wind
turbine or research question specific input. The modules in this package can rather be seen as direct extensions
to the uncertainpy package and might in fact (in part) be merged into the uncertainpy package in the future.
The use_cases directory contains wind turbine and research question specific routines. For now, only one use case has
been implemented.

    - wtuq_framework -> installable package
      | - uq_framework.py
      | - splining.py
      | - helperfunctions.py
      | - UQResultsAnalysis.py
    - use_cases
      | - stability_analysis
        | - config
        | - reference_data
        | - result_comparison
        | - tool_interfaces
        | - run_analysis.py
        | - preprocessor.py
        | - postprocessor.py
        | - etc etc.py
      | - other use cases
    - doc
    - tests

## Usage

1. Make a config for the framework run

    The config has two parts. The [framework] part provides the settings for uncertainpy and the definition of the
    uncertain parameters. The specification of this part is given in wtuq_framework/config.spec.
    The [use_case] part contains use_case specific information. A new file of config specifications has to be generated
    for new use cases. An example can be found in use_cases/stability_analysis/config/use_case_config.spec

    (For an example config file, see use_cases/config/examples/example_config_case_study1.txt)

2. Run the framework

    Basic use: `python run_analysis.py -c path_to_your_config`

    There are two alternative 'restart' functionalities.

    1. `python run_analysis.py -c path_to_your_config -rh5 ../reference_data/bladed/run123/uq_results/Model.h5`

        The framework can extract the training data of a previous run from the standard uncertainpy .h5 output file.
        The script will check that the requested Model.h5 file is available and that it has the same uncertain
        parameter settings as requested in the config file.
    
        This setting is useful when changes are made to uncertainpy itself or if new postprocessing scripts need to be
        tested.

    2. `python run_analysis.py -c path_to_your_config -r ../reference_data/bladed/run123/*`

        The framework also has a restart capability that recognizes already computed results and omits the corresponding
        simulations. This way, different post-processing algorithms and settings can be applied and compared.
        It also helps to recover an analysis in case of a software crash.
    
        The restart directory are defined as a pattern (as understood by the `glob` package) and must be given with a
        full valid path.

3. Compare different runs

    The UQ results of different runs (e.g., different models/tools) can be compared by running the
    wtuq_framework.uq_results_analysis script with a special config file.

    Basic use: `python uq_results_analysis.py -c path_to_your_uq_results_analysis_config`

    The config specification can be found in wtuq_framework/uq_results_analysis_config.spec. An example config file
    can be found in use_cases/result_comparison/config_torque_paper.txt

## Requirements

#### General

* OS: Windows & Linux
* Python >=3.7

#### Required Python packages

* numpy
* matplotlib
* pandas
* configobj
* bokeh
* uncertainpy @ git+https://github.com/DLR-AE/uncertainpy.git
* pydmd
* geomdl

## Installing
* to use: `pip install --user https+git://github.com/DLR-AE/wtuq.git`
* to develop
  * clone repository
  * `pip install -e <local_repo_path>`
  
Only the general wtuq_framework package will be installed automatically. To work on the available use cases, the 
repository has to be cloned. 

## Support
Please do not hesitate to contact us if you have any questions on the framework or if you want more information. 
We are open for collaboration and welcome further development or modification of the framework.
You can contact us on hendrik.verdonck@dlr.de / oliver.hach@dlr.de, or by reaching out to any of the other collaborators.

## Contributing
This framework was the joint work of the DLR (Institute of Aeroelasticity), Leibniz University Hannover (Institut für
Windenergiesysteme), and Nordex GmbH. 
Active colloborators on the project were: 
* [Hendrik Verdonck](https://github.com/hendrikverdonck), hendrik.verdonck@dlr.de
* [Oliver Hach](https://github.com/hach-ol-dlr), oliver.hach@dlr.de
* [Claudio Balzani](https://github.com/claudiobalzani), claudio.balzani@iwes.uni-hannover.de
* Jelmer Polman, Jelmer.polman@iwes.uni-hannover.de
* Sarah Müller, SMueller2@nordex-online.com
* [Johannes Rieke](https://github.com/RiekeJ), JRieke@nordex-online.com
* Otto Braun

## Acknowledgment
This software package was developed in the frame of the German national research project QuexUS.
This project is funded by the German Federal Ministry for Economic Affairs and Climate Action, grant no. 03EE3011A/B.
