from slab.experiments.PulseExperiments_M8195A_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_M8195A_PXI.pulse_experiment_with_switch import Experiment
import json

import os
import numpy as np
from h5py import File
path = os.getcwd()


def take_controls_and_measure(times, controls, taus, measure_with_piby2s=True):
    filepath = "S:\\KevinHe\\Optimal Control and Blockade\\Aditya work\\221117_hardware_looping\\"
    path = os.getcwd()

    ##### Create file that will be accessed by the experiment code #####
    total_time = times[-1]
    measure_pulse_fracs = []
    # taus = [len(times)]  # comment out later
    for tau in taus:
        measure_pulse_fracs.append(min(1.0, times[tau - 1] / total_time))
    steps = len(times)
    uneven_tlist = True  # assume for generality that times is not equally spaced, should still work
    # even if times is evenly spaced

    file_number = 0
    pulse_filename = "hardware_looping_tests_v2.h5"
    while os.path.exists(os.path.join(filepath, str(file_number).zfill(5) + "_" + pulse_filename)):
        file_number += 1
    pulse_filename = str(file_number).zfill(5) + "_" + pulse_filename 

    with File(filepath + pulse_filename, 'w') as hf:
        hf.create_dataset('uks', data=np.array([np.array(controls).T]))
        hf.create_dataset('total_time', data=total_time)
        hf.create_dataset('steps', data=steps)
        if uneven_tlist:
            hf.create_dataset('times', data=times)

    ##### Load system parameters #####
    with open('quantum_device_config.json', 'r') as f:
        quantum_device_cfg  = json.load(f)
    with open('experiment_config.json', 'r') as f:
        experiment_cfg = json.load(f)
    with open('hardware_config.json', 'r') as f:
        hardware_cfg = json.load(f)

    ##### Run the experiment which will generate data files #####
    experiment_name = 'optimal_control_test_1step'
    show = 'I'
    TRIGGER_TIME = 15000
    if measure_with_piby2s:
        settings_list = [[False, 0.0], [True, 0.0], [True, -np.pi/2]]
    else:
       settings_list = [[False, 0.0]]

    final_output = []
    for measure_pulse_frac in measure_pulse_fracs:
        data_filenames = []
        for settings in settings_list:
            experiment_cfg['optimal_control_test_1step']['filename'] = filepath + pulse_filename
            experiment_cfg['optimal_control_test_1step']['piby2_before_photon_dist_meas'] = settings[0]
            experiment_cfg['optimal_control_test_1step']['piby2_before_photon_dist_meas_phase'] = settings[1]
            experiment_cfg['optimal_control_test_1step']['pulse_frac'] = measure_pulse_frac
            hardware_cfg['trigger']['period_us'] = TRIGGER_TIME
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg,plot_visdom=True)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            # print(sequences)
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=False)
            I, Q, data_filename = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
                                                       data_file_path="S:\\_Data\\2021-10-22 Multimode cooldown 16 with JPA as of 2022-05-04\\",
                                                       return_filename=True)
            exp.post_analysis(experiment_name, P=show, show=False)
            data_filenames.append(data_filename)

        ##### Get data from the files #####
        Is = []
        for data_filename in data_filenames:
            with File(data_filename, 'r') as a:
                # hardware_cfg =  (json.loads(a.attrs['hardware_cfg']))
                # experiment_cfg =  (json.loads(a.attrs['experiment_cfg']))
                # quantum_device_cfg = (json.loads(a.attrs['quantum_device_cfg']))
                # expt_cfg = (json.loads(a.attrs['experiment_cfg']))[experiment_names[0]]
                I, Q = np.array(a['I']), np.array(a['Q'])
            data_to_look_at = I
            if experiment_cfg['optimal_control_test_1step']['singleshot']:
                data_means = np.mean(data_to_look_at, axis=0)
                g_val = data_means[-2]
                e_val = data_means[-1]
                Is.append((data_means[:-2] - g_val) / (e_val - g_val))
                # Additional things, not sure what quantities are desired
                # measurements formatted with each row corresponding to a photon peak, so actual_meas_dat[0] will be
                # the 0 photon peak, etc.
                actual_meas_dat = np.array(data_to_look_at).T[:-2]
                # all values corresponding to qubit in g for normalization
                g_dat = np.array(data_to_look_at).T[-2]
                # all values corresponding to qubit in e for normalization
                e_dat = np.array(data_to_look_at).T[-1]
            else:
                Is.append(data_to_look_at)
        Is = np.array(Is)
        e_pop = np.mean(Is[0][-3:])  # higher photon numbers should be flat at baseline value, average last 3
        # g_pop = 1 - e_pop
        # output = [g_pop, e_pop]
        output = [e_pop]

        ##### Data analysis and manipulation #####
        look_at_n_peaks = 10  # number of peaks to look at
        cav_pops_collected = []
        for I in Is:
            baseline = np.mean(I[-3:])
            cav_pops_part = np.abs(I[:look_at_n_peaks] - baseline)
            cav_pops_collected.append(cav_pops_part)
        cav_pops = np.sum(cav_pops_collected, axis=0)
        cav_pops = cav_pops / np.sum(cav_pops)  # normalize to total population of 1

        norm_scale = np.mean(cav_pops[-3:])  # normalize out the last 3 levels
        cav_pops_norm = np.abs(cav_pops - norm_scale)
        cav_pops_norm = cav_pops_norm / np.sum(cav_pops_norm)

        cav_cov = np.cov(cav_pops_collected)

        for pop in cav_pops_norm:
            output.append(pop)
            
        final_output.append(output)

    return np.array(final_output)

# # Testing that the code runs
# filename = "S:\\KevinHe\\Optimal Control and Blockade\\Aditya work\\221114_Aaron_pulses\\converted\\g0_to_g1_T_102_dt_4.0_Q_200.0_R_0.1_u_bound_0.0001_iter_10000_00005_noflip.h5"
# with File(filename, 'r') as f:
#     controls = f['uks'][()][0]
#     times = f['times'][()]
#
# print(take_controls_and_measure(times, controls, "test.h5", 1.0))
   