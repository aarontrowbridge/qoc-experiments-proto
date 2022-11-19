from slab.experiments.PulseExperiments_M8195A_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_M8195A_PXI.pulse_experiment_with_switch import Experiment
import json

import os
import numpy as np
from h5py import File
path = os.getcwd()

############# Load system paramters #############
with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)

experiment_names = ['optimal_control_test_1step']

show = 'I'

trigger_time = 15000

# working directory
filepath = "S:\\KevinHe\\Optimal Control and Blockade\\Aditya work\\221117_hardware_looping\\"

iterations = 5

for _ in range(iterations):
    ############# Optimization part #############

    #************** TO DO ******************


    # Parameters that are needed from the optimization
    filename = ""  # output filename from the optimization
    uneven_tlist = False  # whether the optimized t_list is uneven or not
    total_time = 0  # length of the pulse
    delta_t = 0  # time step between optimization points
    times = []  # list of times, if uneven
    uks = [[[]]]  # controls

    with File(filepath + filename, 'w') as hf:  # creating pulse file
        hf.create_dataset('uks', data=uks)
        hf.create_dataset('total_time', data=total_time)
        hf.create_dataset('steps', data=int(total_time/delta_t))
        if uneven_tlist:
            hf.create_dataset('times', data=times)

    ############# Run the experiment #############
    for experiment_name in experiment_names:
        # experiment_cfg['optimal_control_test_1step']['filename'] = filename
        hardware_cfg['trigger']['period_us'] = trigger_time
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg,plot_visdom=True)
        sequences = ps.get_experiment_sequences(experiment_name)
        #sequences = ps.get_experiment_sequences(experiment_name,save=True,filename='00001_g0_to_g1_blockade_omega2_filt3_frac0.2')
        print("Sequences generated")
        # print(sequences)
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=False)
        I,Q,data_filename = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
                               data_file_path="S:\\_Data\\2021-10-22 Multimode cooldown 16 with JPA as of 2022-05-04\\",
                               return_filename=True)
        exp.post_analysis(experiment_name, P=show, show=True)

    ############# Get data #############
    with File(data_filename, 'r') as a:
        hardware_cfg =  (json.loads(a.attrs['hardware_cfg']))
        experiment_cfg =  (json.loads(a.attrs['experiment_cfg']))
        quantum_device_cfg =  (json.loads(a.attrs['quantum_device_cfg']))
        expt_cfg = (json.loads(a.attrs['experiment_cfg']))[experiment_names[0]]
        I,Q = np.array(a['I']), np.array(a['Q'])

    data_to_look_at = I

    ############# Manipulate data into cost function or compare with expected #############
    #************** TO DO ******************