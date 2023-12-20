
[use_case]
    [[tool]]
        # select a tool
        tool_name = option('bladed-lin', 'dummy-tool', 'hawc2', 'hawcstab2', default='dummy-tool')

        # HAWCStab2 data
        n_op_points = integer(default=0)
        nmodes = integer(default=0)
        ndofs = integer(default=0)
        mode_indices_ref = list(default=list(0))
        minimum_MAC_mode_picking = float(default=0)
        minimum_MAC_mode_tracking = float(default=0)

        # Tool specific inputs can be given here. they will be forwarded to the tool interface.

    [[preprocessor]]

        # for now nothing yet