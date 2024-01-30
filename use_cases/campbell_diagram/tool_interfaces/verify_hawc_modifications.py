import numpy as np
import matplotlib.pyplot as plt
import os
from hawc2_interface import HAWC2Model


def read_geom_file(fname):

    header_length_geom = 2
    blade_geometry = np.loadtxt(fname, skiprows=header_length_geom)
    return blade_geometry


def read_ref_axis(fname):

    with open(fname, 'r') as htc_main_file:
        htc_content = htc_main_file.readlines()

    line_idx_bra, line = HAWC2Model.find_anchor(None, htc_content, '#blade_ref_axis')
    nsec = int(line.split()[1])
    blade_ref_axis = np.loadtxt(fname, skiprows=line_idx_bra + 1, max_rows=nsec, usecols=(1, 2, 3, 4, 5))

    return blade_ref_axis


def read_blade_struc(fname):

    blade_structure = np.loadtxt(fname, skiprows=5, max_rows=26)
    return blade_structure


def read_tower_struc(fname):

    tower_structure = np.loadtxt(fname, skiprows=3, max_rows=20)
    return tower_structure


def read_polars(fname):

    test = HAWC2Model(None, None)
    test.pc_file = fname
    polars, aero_data, _, _ = test.get_polars()

    return polars, aero_data


def basic_plot(r_orig, param_orig, r_modif, param_modif, ylabel, save_file):
    plt.plot(r_orig, param_orig, label='reference')
    plt.plot(r_modif, param_modif, label='modified')
    plt.xlabel('radius / m')
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(save_file)
    plt.close()


def verify_modifications(original_dir, modified_dir):

    main_model_htc_file = r'IEA_15MW_RWT_Onshore.htc'
    data_shaft_st_file = r'IEA_15MW_RWT_Shaft_st.dat'
    data_tower_st_file = r'IEA_15MW_RWT_Tower_st.dat'
    data_blade_geom_file = r'IEA_15MW_RWT_ae.dat'
    data_blade_st_file = r'IEA_15MW_RWT_Blade_st_noFPM.st'
    pc_file = r'IEA_15MW_RWT_pc_OpenFASTpolars_3dcorr.dat'

    # COMPARE polars
    orig_polars, orig_aero_data = read_polars(os.path.join(original_dir, 'data', pc_file))
    modif_polars, modif_aero_data = read_polars(os.path.join(modified_dir, 'data', pc_file))

    for ii, (orig_data, modif_data) in enumerate(zip(orig_aero_data, modif_aero_data)):
        plt.plot(orig_data[:, 0], orig_data[:, 1], label='reference cl')
        plt.plot(modif_data[:, 0], modif_data[:, 1], label='modified cl')
        plt.plot(orig_data[:, 0], orig_data[:, 2], label='reference cd')
        plt.plot(modif_data[:, 0], modif_data[:, 2], label='reference cd')
        plt.xlabel('AOA / deg')
        plt.ylabel('Cl Cd / -')
        plt.grid()
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(modified_dir, 'polar_'+str(ii)))
        plt.close()

    # COMPARE blade reference axis
    orig_blade_ref_axis = read_ref_axis(os.path.join(original_dir, 'htc', main_model_htc_file))
    modif_blade_ref_axis = read_ref_axis(os.path.join(modified_dir, 'htc', main_model_htc_file))

    basic_plot(orig_blade_ref_axis[:, 3], orig_blade_ref_axis[:, 1], modif_blade_ref_axis[:, 3], modif_blade_ref_axis[:, 1], 'pre-sweep / m', os.path.join(modified_dir, 'geom_sweep'))
    basic_plot(orig_blade_ref_axis[:, 3], orig_blade_ref_axis[:, 2], modif_blade_ref_axis[:, 3], modif_blade_ref_axis[:, 2], 'prebend / m', os.path.join(modified_dir, 'geom_prebend'))
    basic_plot(orig_blade_ref_axis[:, 3], orig_blade_ref_axis[:, 4], modif_blade_ref_axis[:, 3], modif_blade_ref_axis[:, 4], 'twist / deg', os.path.join(modified_dir, 'geom_twist'))

    # COMPARE blade structure
    orig_blade_struc = read_blade_struc(os.path.join(original_dir, 'data', data_blade_st_file))
    modif_blade_struc = read_blade_struc(os.path.join(modified_dir, 'data', data_blade_st_file))

    r_orig = orig_blade_struc[:, 0]
    r_modif = modif_blade_struc[:, 0]
    basic_plot(r_orig, orig_blade_struc[:, 1], r_modif, modif_blade_struc[:, 1], 'mass / kg/m', os.path.join(modified_dir, 'blade_struc_mass'))
    basic_plot(r_orig, orig_blade_struc[:, 2], r_modif, modif_blade_struc[:, 2], 'x_cg / m', os.path.join(modified_dir, 'blade_struc_x_cg'))
    basic_plot(r_orig, orig_blade_struc[:, 3], r_modif, modif_blade_struc[:, 3], 'y_cg / m', os.path.join(modified_dir, 'blade_struc_y_cg'))
    basic_plot(r_orig, orig_blade_struc[:, 4], r_modif, modif_blade_struc[:, 4], 'ri_x / m', os.path.join(modified_dir, 'blade_struc_ri_x'))
    basic_plot(r_orig, orig_blade_struc[:, 5], r_modif, modif_blade_struc[:, 5], 'ri_y / m', os.path.join(modified_dir, 'blade_struc_ri_y'))
    basic_plot(r_orig, orig_blade_struc[:, 6], r_modif, modif_blade_struc[:, 6], 'x_sh / m', os.path.join(modified_dir, 'blade_struc_x_sh'))
    basic_plot(r_orig, orig_blade_struc[:, 7], r_modif, modif_blade_struc[:, 7], 'y_sh / m', os.path.join(modified_dir, 'blade_struc_y_sh'))
    basic_plot(r_orig, orig_blade_struc[:, 8], r_modif, modif_blade_struc[:, 8], 'E / N/m^2', os.path.join(modified_dir, 'blade_struc_E'))
    basic_plot(r_orig, orig_blade_struc[:, 9], r_modif, modif_blade_struc[:, 9], 'G / N/m^2', os.path.join(modified_dir, 'blade_struc_G'))
    basic_plot(r_orig, orig_blade_struc[:, 10], r_modif, modif_blade_struc[:, 10], 'I_x / m^4', os.path.join(modified_dir, 'blade_struc_I_x'))
    basic_plot(r_orig, orig_blade_struc[:, 11], r_modif, modif_blade_struc[:, 11], 'I_y / m^4', os.path.join(modified_dir, 'blade_struc_I_y'))
    basic_plot(r_orig, orig_blade_struc[:, 12], r_modif, modif_blade_struc[:, 12], 'I_p / m^4', os.path.join(modified_dir, 'blade_struc_I_p'))
    basic_plot(r_orig, orig_blade_struc[:, 13], r_modif, modif_blade_struc[:, 13], 'k_x / -', os.path.join(modified_dir, 'blade_struc_k_x'))
    basic_plot(r_orig, orig_blade_struc[:, 14], r_modif, modif_blade_struc[:, 14], 'k_y / -', os.path.join(modified_dir, 'blade_struc_k_y'))
    basic_plot(r_orig, orig_blade_struc[:, 15], r_modif, modif_blade_struc[:, 15], 'A / m^2', os.path.join(modified_dir, 'blade_struc_A'))
    basic_plot(r_orig, orig_blade_struc[:, 16], r_modif, modif_blade_struc[:, 16], 'pitch / deg', os.path.join(modified_dir, 'blade_struc_pitch'))
    basic_plot(r_orig, orig_blade_struc[:, 17], r_modif, modif_blade_struc[:, 17], 'x_e / m', os.path.join(modified_dir, 'blade_struc_x_e'))
    basic_plot(r_orig, orig_blade_struc[:, 18], r_modif, modif_blade_struc[:, 18], 'y_e / m', os.path.join(modified_dir, 'blade_struc_y_e'))

    # COMPARE blade geom
    orig_blade_geom = read_geom_file(os.path.join(original_dir, 'data', data_blade_geom_file))
    modif_blade_geom = read_geom_file(os.path.join(modified_dir, 'data', data_blade_geom_file))

    basic_plot(orig_blade_geom[:, 0], orig_blade_geom[:, 1], modif_blade_geom[:, 0], modif_blade_geom[:, 1], 'chord / m', os.path.join(modified_dir, 'geom_chord'))


    # COMPARE tower structure
    orig_tower_struc = read_tower_struc(os.path.join(original_dir, 'data', data_tower_st_file))
    modif_tower_struc = read_tower_struc(os.path.join(modified_dir, 'data', data_tower_st_file))

    r_orig = orig_tower_struc[:, 0]
    r_modif = modif_tower_struc[:, 0]
    basic_plot(r_orig, orig_tower_struc[:, 1], r_modif, modif_tower_struc[:, 1], 'mass / kg/m', os.path.join(modified_dir, 'tower_struc_mass'))
    basic_plot(r_orig, orig_tower_struc[:, 2], r_modif, modif_tower_struc[:, 2], 'x_cg / m', os.path.join(modified_dir, 'tower_struc_x_cg'))
    basic_plot(r_orig, orig_tower_struc[:, 3], r_modif, modif_tower_struc[:, 3], 'y_cg / m', os.path.join(modified_dir, 'tower_struc_y_cg'))
    basic_plot(r_orig, orig_tower_struc[:, 4], r_modif, modif_tower_struc[:, 4], 'ri_x / m', os.path.join(modified_dir, 'tower_struc_ri_x'))
    basic_plot(r_orig, orig_tower_struc[:, 5], r_modif, modif_tower_struc[:, 5], 'ri_y / m', os.path.join(modified_dir, 'tower_struc_ri_y'))
    basic_plot(r_orig, orig_tower_struc[:, 6], r_modif, modif_tower_struc[:, 6], 'x_sh / m', os.path.join(modified_dir, 'tower_struc_x_sh'))
    basic_plot(r_orig, orig_tower_struc[:, 7], r_modif, modif_tower_struc[:, 7], 'y_sh / m', os.path.join(modified_dir, 'tower_struc_y_sh'))
    basic_plot(r_orig, orig_tower_struc[:, 8], r_modif, modif_tower_struc[:, 8], 'E / N/m^2', os.path.join(modified_dir, 'tower_struc_E'))
    basic_plot(r_orig, orig_tower_struc[:, 9], r_modif, modif_tower_struc[:, 9], 'G / N/m^2', os.path.join(modified_dir, 'tower_struc_G'))
    basic_plot(r_orig, orig_tower_struc[:, 10], r_modif, modif_tower_struc[:, 10], 'I_x / m^4', os.path.join(modified_dir, 'tower_struc_I_x'))
    basic_plot(r_orig, orig_tower_struc[:, 11], r_modif, modif_tower_struc[:, 11], 'I_y / m^4', os.path.join(modified_dir, 'tower_struc_I_y'))
    basic_plot(r_orig, orig_tower_struc[:, 12], r_modif, modif_tower_struc[:, 12], 'I_p / m^4', os.path.join(modified_dir, 'tower_struc_I_p'))
    basic_plot(r_orig, orig_tower_struc[:, 13], r_modif, modif_tower_struc[:, 13], 'k_x / -', os.path.join(modified_dir, 'tower_struc_k_x'))
    basic_plot(r_orig, orig_tower_struc[:, 14], r_modif, modif_tower_struc[:, 14], 'k_y / -', os.path.join(modified_dir, 'tower_struc_k_y'))
    basic_plot(r_orig, orig_tower_struc[:, 15], r_modif, modif_tower_struc[:, 15], 'A / m^2', os.path.join(modified_dir, 'tower_struc_A'))
    basic_plot(r_orig, orig_tower_struc[:, 16], r_modif, modif_tower_struc[:, 16], 'pitch / deg', os.path.join(modified_dir, 'tower_struc_pitch'))
    basic_plot(r_orig, orig_tower_struc[:, 17], r_modif, modif_tower_struc[:, 17], 'x_e / m', os.path.join(modified_dir, 'tower_struc_x_e'))
    basic_plot(r_orig, orig_tower_struc[:, 18], r_modif, modif_tower_struc[:, 18], 'y_e / m', os.path.join(modified_dir, 'tower_struc_y_e'))


def verify_all_modifications():
    preprocessed_data = {'nacelle_mass': 0.1,
                         'nacelle_cog_vertical': 0.1,
                         'nacelle_cog_horizontal': 0.1,
                         'nacelle_yaw_inertia': 0.1,
                         'nacelle_nodding_inertia': 0.1,
                         'nacelle_rolling_inertia': 0.1,
                         'cone_angle': 0.1,
                         'LSS_stiffness': 0.1,
                         'LSS_damping': 0.1,
                         'tower_bending_stiffness': 0.1,
                         'tower_torsional_stiffness': 0.1,
                         'tower_shear_stiffness': 0.1,
                         'tower_mass': 0.1,
                         'blade_damping_1': 0.1,
                         'blade_damping_2': 0.1,
                         'blade_damping_3': 0.1,
                         'blade_damping_4': 0.1,
                         'tower_damping_1': 0.1,
                         'tower_damping_2': 0.1,
                         'blade_chord_root': 0.1,
                         'blade_chord_tip': 0.1,
                         'blade_twist': 0.1,
                         'blade_na_x': 0.1,
                         'blade_na_y': 0.1,
                         'blade_flap_stiffness': 0.1,
                         'blade_edge_stiffness': 0.1,
                         'blade_pa_orientation': 0.1,
                         'blade_mass': 0.1,
                         'blade_cog_x': 0.01,
                         'blade_cog_y': 0.01,
                         'blade_sc_x': 0.01,
                         'blade_sc_y': 0.01,
                         'polar_clalpha': 0.1,
                         'polar_alpha_max': 0.1,
                         'polar_cd0': 0.1,
                         'ds_a1': 0.1,
                         'ds_a2': 0.1,
                         'ds_b1': 0.1,
                         'ds_b2': 0.1,
                         'ds_taupre': 0.1,
                         'ds_taubly': 0.1}

    for key, val in preprocessed_data.items():
        subset_preprocessed_data = {key: val}
        iteration_run_directory = os.path.join(r'C:\Users\verd_he\Desktop\remove', key)
        config = {'master_model_directory_path': r'Z:\projects\IEA-15-240-RWT\HAWC2\IEA-15-240-RWT-Onshore-master',
                  'main_model_htc_file': r'IEA_15MW_RWT_Onshore.htc',
                  'data_shaft_st_file': r'IEA_15MW_RWT_Shaft_st.dat',
                  'data_tower_st_file': r'IEA_15MW_RWT_Tower_st.dat',
                  'data_blade_geom_file': r'IEA_15MW_RWT_ae.dat',
                  'data_blade_st_file': r'IEA_15MW_RWT_Blade_st_noFPM.st',
                  'pc_file': r'IEA_15MW_RWT_pc_OpenFASTpolars_3dcorr.dat'}

        test = HAWC2Model(iteration_run_directory, config)
        test.create_simulation(subset_preprocessed_data)

        verify_modifications(r'Z:\projects\IEA-15-240-RWT\HAWC2\IEA-15-240-RWT-Onshore-master', iteration_run_directory)


if __name__ == '__main__':
    verify_all_modifications()
