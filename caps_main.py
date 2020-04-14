import config
import tensorflow as tf
from caps_network import Caps3d
from get_iou import iou
from load_synth_data import SynthTrainDataGenDet as TrainDataGen, SynthTestDataGenDet as TestDataGen


def get_num_params():
    # prints out the number of trainable  parameters in the TensorFlow graph
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Num of parameters:', total_parameters)


def train_network(gpu_config):
    #with tf.device("/gpu:3"):
    capsnet = Caps3d()
        
    with tf.Session(graph=capsnet.graph, config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        
        if config.continue_from_chkpt:
            capsnet.load(sess, config.network_save_dir)
            capsnet.cur_m += config.m_delta * config.prev_epochs / config.n_eps_for_m
            capsnet.cur_m = min(capsnet.cur_m, 0.9)
        else:
            config.clear_output()
            config.prev_epochs = 0

        get_num_params()
        

        n_eps_after_acc, best_loss = -1, 100000
        print('Training on Synthetic Videos')
        for ep in range(1, config.n_epochs+1):
            print(20 * '*', 'epoch', ep, 20 * '*')

            # trains network for one epoch
            data_gen = TrainDataGen(config.wait_for_data, frame_skip=config.frame_skip)
            margin_loss, seg_loss, acc = capsnet.train(sess, data_gen)
            config.write_output('Training\tCL: %.4f. SL: %.4f. Acc: %.4f.\n' % (margin_loss, seg_loss, acc))

            # increments the margin
            if (ep + config.prev_epochs) % config.n_eps_for_m == 0:
                capsnet.cur_m += config.m_delta
                capsnet.cur_m = min(capsnet.cur_m, 0.9)
                
            try:
                capsnet.save(sess, config.save_file_name, config.prev_epochs + ep) #+25 
                config.write_output('Saved Network\n')
            except:
                print('Failed to save network!!!')
            
            
            # validates the network
            data_gen = TestDataGen(config.wait_for_data, frame_skip=1)
            margin_loss, seg_loss, accuracy, _ = capsnet.eval(sess, data_gen, validation=True)
            config.write_output('Validation\tCL: %.4f. SL: %.4f. Acc: %.4f.\n' %
                                    (margin_loss, seg_loss, accuracy))
            
            '''
            # only validates after a certain number of epochs and when the training accuracy is greater than a threshold
            # this is mainly used to save time, since validation takes about 10 minutes
            if (acc >= config.acc_for_eval or n_eps_after_acc >= 0) and ep >= config.n_eps_until_eval:
                n_eps_after_acc += 1

            # validates the network
            if (acc >= config.acc_for_eval and n_eps_after_acc % config.n_eps_for_eval == 0) or ep == config.n_epochs:
                data_gen = TestDataGen(config.wait_for_data, frame_skip=config.frame_skip)
                # data_gen = TestDataGen(config.wait_for_data, frame_skip=1)
                margin_loss, seg_loss, accuracy, _ = capsnet.eval(sess, data_gen, validation=True)

                config.write_output('Validation\tCL: %.4f. SL: %.4f. Acc: %.4f.\n' %
                                    (margin_loss, seg_loss, accuracy))

                # saves the network when validation loss in minimized
                t_loss = margin_loss + seg_loss
                if t_loss < best_loss:
                    best_loss = t_loss
                    try:
                        capsnet.save(sess, config.save_file_name)
                        config.write_output('Saved Network Finally\n')
                    except:
                        print('Failed to save network!!!')
            '''
    # calculate final test accuracy, f-mAP, and v-mAP
    # iou()


def main():
    # trains the network with the given gpu configuration
    train_network(config.gpu_config)


main()
