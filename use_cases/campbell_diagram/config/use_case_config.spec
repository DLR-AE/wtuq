
[use_case]
    [[tool]]
        # select a tool
        tool_name = option('bladed-lin', 'dummy-tool', 'hawc2', 'hawcstab2', default='dummy-tool')

        # HAWCStab2 data
        n_op_points = integer(default=0)
        nmodes = integer(default=0)
        ndofs = integer(default=0)
        mode_indices_ref = list(default=list(0))

        postpro_method = option('MAC_based_mode_picking_and_tracking', 'MAC_based_mode_picking_and_tracking_mode_specific_thresholds', default='MAC_based_mode_picking_and_tracking_mode_specific_thresholds')

        # These thresholds have to be given if postpro_method = MAC_based_mode_picking_and_tracking
        minimum_MAC_mode_picking = float(default=0)
        minimum_MAC_mode_tracking = float(default=0)
        minimum_MAC_mode_tracking_wrt_ref = float(default=0)

        # These thresholds have to be given if postpro_method = MAC_based_mode_picking_and_tracking_mode_specific_thresholds
        minimum_MAC_mode_picking_mode_specific = list(default=list(0))
        minimum_MAC_mode_tracking_mode_specific = list(default=list(0))
        minimum_MAC_mode_tracking_wrt_ref_mode_specific = list(default=list(0))

        with_damp = boolean(default=True)  # full eigenvalue will be used for MAC correction instead of only frequency

        # Tool specific inputs can be given here. they will be forwarded to the tool interface.

    [[preprocessor]]

        # for now nothing yet