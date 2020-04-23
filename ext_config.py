import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"]="3"
gpu_config = tf.ConfigProto()
#gpu_config.gpu_options.visible_device_list= '1,2,0,3'
gpu_config.gpu_options.visible_device_list= '3'
gpu_config.gpu_options.allow_growth = True
#gpu_config.gpu_options.per_process_gpu_memory_fraction = 1.0
#gpu_config.gpu_options.allocator_type = 'BFC'
#gpu_config.log_device_placement = True
#gpu_config.allow_soft_placement = True

# batch size and number of epochs
batch_size = 2
n_epochs = 20

# whether to continue from last checkpoint or not
continue_from_chkpt = False

ann_type = 'word_ann' # 'char_ann' 'word_ann' 'line_ann' 'para_ann'

# number of epochs to train in between validations
n_eps_for_eval = 5

# training accuracy threshold needed for validation to run
acc_for_eval = 0.5

# number of epochs until validation can start
n_eps_until_eval = 0

# learning rate and beta1 are used in the Adam optimizer.
learning_rate, beta1 = 0.0001, 0.5

# Used to prevent numerical instability (dividing by zero or log(0))
epsilon = 1e-6

use_c3d_weights = False

# number of classes for the network to predict
n_classes = 12

# resolution
vid_h = 128
vid_w = 240

# model number, output file name, save file directory, and save file name
model_num = 2
output_file_name = './ext_output_inference_%d.txt' % model_num
#network_save_dir = './older_models/save_with_bad_split/'
#network_save_dir = './older_models/plate_nums_20/network_saves/'
network_save_dir = './network_saves/'
if not os.path.exists(network_save_dir):  # creates the directory if it does not exist
    os.mkdir(network_save_dir)
save_file_name = network_save_dir + ('model_%d.ckpt' % model_num)

# coefficient for the segmentation loss
segment_coef = 0.0002

# margin for classification loss, how much it is incremented by, and how often it is incremented by
start_m = 0.2
m_delta = 0.1
n_eps_for_m = 5

# number of frames to skip in the data
frame_skip = 2

# time to wait for data to load when dataloader is created
wait_for_data = 5

# number of batches to train on before statistics are printed to stdio
batches_until_print = 100

# parameters for the EM-routing operation
inv_temp = 0.5
inv_temp_delta = 0.1

# size of the pose matrix height and width
pose_dimension = 4

# determines if the network layers will be printed when network is initialized
print_layers = True


def clear_output():
    """
    Clears the text file which the training/validation/testing metrics will be printed to
    """
    with open(output_file_name, 'w') as f:
        print('Writing to ' + output_file_name)


def write_output(string):
    """
    Writes a given string to the text output file. Used to write the different metrics.
    """
    try:
        output_log = open(output_file_name, 'a')
        output_log.write(string)
        output_log.close()
    except:
        print('Unable to save to output log')
        
labels = {0: '_tr_l_r_',
          1: '_tr_r_l_',
          2: '_tr_t_b_',
          3: '_tr_b_t_',
          4: '_roll_ac_',
          5: '_roll_c_',
          6: '_Pan1_',
          7: '_Panrev1_',
          8: '_tilt1_',
          9: '_tilt1rev_',
          10: '_zoomout_',
          11: '_zoomin_',
          }
