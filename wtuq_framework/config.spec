
[framework]
    [[uncertainty]]

        # base dir for results (absolute or relative to cwd)
        base_directory = string(default='./reference_data')
        # run dir (common for all results of a study), relative to base dir
        run_directory = string(default='runs')

        # select run type:
        # full: normal framework run
        # test: only 1 random sample iteration
        # reference: only 1 iteration with the unaltered reference model
        run_type = option('full', 'test', 'reference', default='full')

        # set number of CPUs
        n_CPU = integer(default=1)

        log_level = option('notset', 'debug', 'info', 'warning', 'error', 'critical', default='debug')

        # UQ method specification
        uq_method = string(default=pc)

        # PCE specifications
        pc_sampling_method = option('sobol', 'hammersley', 'random', default='sobol')
        pc_regression_model = option('least_squares', 'lasso', 'lars', 'lasso_lars', default='least_squares')
        polynomial_order = integer(default=4)
        nr_collocation_nodes = integer(default=None)

        # Monte-Carlo specifications
        nr_mc_samples = integer(default=10000)

        # Custom UQ methods
        morris_nr_of_repetitions = integer(default=10)
        morris_oat_linear_disturbance = boolean(default=True)
        morris_oat_linear_disturbance_factor = float(default=1E-3)

    [[parameters]]

        # All uncertain parameters are defined here:
        #
        # if method==multiplicative
        #   - spline/interpolation values are used as factor on the reference value (e.g. 1.1 = 110% reference value)
        # if method==additive
        #   - spline/interpolation values is simply added to the reference value
        #
        # if the radial_distribution==spline
        #   - uncertain parameters (range defined by min and max) and fixed parameters that are unchanged during the UQ.
        # if the radial_distribution==fixed
        #   - control points are used to build a linear interpolation. A minimal and maximal distribution have to be made.
        #     The uncertain parameter determines where the distribution lays between the max. and min. distribution.
        # if the radial_distribution==none
        #   - no radial distribution, uncertain parameter is a single, scalar property
        #
        # if param_distribution==uniform
        #   - min and max control points describe the outer bounds of the uniform distribution
        # if param_distribution==normal
        #   - normal distribution with min = mean - std and max = mean + std
        #
        # Example:
        # Description:
        # Flap stiffness has a normal distribution with +- 10% standard deviation around the reference value.
        # The uncertainty in radial direction is governed by a two fixed control points at the root and tip and
        # a single variable control point in the middle of the blade, which determines the spline shape.
        # Edge stiffness has a uniform distribution of +- 5% around the reference value. The uncertainty in radial
        # direction is also uniform over the blade.
        #   [[parameters]]
        #       [[[flap_stiffness]]]
        #           radial_distribution = spline
        #           param_distribution = normal
        #           [[[[cp00]]]]
        #               [[[[[ x ]]]]]
        #               fix = 0.0
        #               [[[[[ y ]]]]]
        #               fix = 1.0
        #           [[[[cp01]]]]
        #               [[[[[ x ]]]]]
        #               fix = 0.5
        #               [[[[[ y ]]]]]
        #               min = 0.9
        #               max = 1.1
        #           [[[[cp02]]]]
        #               [[[[[ x ]]]]]
        #               fix = 1.0
        #               [[[[[ y ]]]]]
        #               fix = 1.0
        #       [[[edge_stiffness]]]
        #           radial_distribution = fixed
        #           param_distribution = uniform
        #           [[[[cp00]]]]
        #               [[[[[ x ]]]]]
        #               fix = 0.0
        #               [[[[[ y ]]]]]
        #               min = 0.95
        #               max = 1.05
        #           [[[[cp01]]]]
        #               [[[[[ x ]]]]]
        #               fix = 1.0
        #               [[[[[ y ]]]]]
        #               min = 0.95
        #               max = 1.05

        [[[__many__]]]
            method  = option('multiplicative', 'additive', default='multiplicative')
            radial_distribution = option('spline', 'fixed', 'none', default='spline')
            param_distribution = option('uniform', 'normal', default='uniform')
            [[[[__many__]]]]
                # two sections are needed for the dofs of each control point
                [[[[[x]]]]]
                [[[[[y]]]]]
                  # the control point can either used to describe the uncertain distribution or can be fixed
                  # ### either ###
                  # min = <number>
                  # max = <number>
                  # ### or ###
                  # fix = <number>
