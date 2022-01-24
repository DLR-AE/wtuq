
[use_case]
    [[tool]]
        # select a tool
        tool_name = option('alaska', 'bladed', 'bladed-lin', 'dummy-tool', 'hawc2', 'hawcstab2', 'openfast', 'simpack', default='dummy-tool')

        # Tool specific inputs can be given here. they will be forwarded to the tool interface.

    [[preprocessor]]

        # location of the BECAS cross section reference data directory (original IWT-175-6.4 blade) (absolute or relative to cwd)
        BECAS_directory = string(default=./reference_data/BECAS/reference_blade/c2_coordinate_system)
        # location of HAWC2-BECAS geometry data (absolute or relative to cwd)
        h2_becas_geometry_path = string(default=./reference_data/BECAS/hawc2_becas_luh.dat)
        # location of FOCUS geometry data (absolute or relative to cwd)
        focus_geometry_path = string(default=./reference_data/BECAS/focus_geometry.csv)
        # location of the HAWC2 beam properties template data (reduced stiffness blade) - only values which can not be uniquely derived from the BECAS cross section matrices will be used (absolute or relative to cwd)
        template_h2_beam_properties_path = string(default=./reference_data/BECAS/reference_blade_reduced_stiffness/BECAS2HAWC2_30_30_70_st_HV.dat)

    [[postprocessor]]

        # signal selection
        # var_name = list of variable names, which have to be available in the result_dict!
        var_name = list(default=list('torsion_b1'))
        # interp_radpos = list of radial positions (along pitch axis) where the signal has to be interpolated to
        interp_radpos = list(default=list(70))

        # windowing method:
        # 0 -> windowing based on user defined crop_window
        # 1 -> windowing based on consistently increasing peaks signal
        # 2 -> windowing based on best linear curve fit
        # 3 -> windowing based on best exponential curve fit
        # 4 -> windowing based on FFT
        # 5 -> fixed window start, window end when oscillation exceeds threshold. Threshold = crop_window[1] * initial_amplitude
        # 6 -> manual window setting for each iteration individually
        # 7 -> manual window setting through text file 'manual_window_selection.txt' in source directory
        window_method = option(0, 1, 2, 3, 4, 5, 6, 7, default=0)
        # crop_window -> used directly if window_method = 0
        #             -> used by method 5 to define fixed window start (crop_window[0]) and to define oscillation threshold (crop_window[1])
        crop_window = list(default=list(0, 1E10))

        # damping determination method
        damping_method = option('DMD', 'log_decr', 'linear_fit', 'exp_fit', default='DMD')
        # number of svd modes to be retained in DMD, 0 -> automatic determination by pyDMD
        svd_rank = integer(default=0)

        # generate window search and damping determination plots
        plotting = boolean(default=True)

        # do MBC transformation on signals before damping determination
        MBC_transf = boolean(default=True)