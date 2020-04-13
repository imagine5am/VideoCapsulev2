from caps_network import Caps3d
from load_synth_data import SynthTrainDataGenDet as TrainDataGen, SynthTestDataGenDet as TestDataGen
import config2 as config
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm


def output_conf(conf):
    try: 
        output_log = open('split_results.txt', 'w')

        output_log.write('Label\t')
        for i in range(config.n_classes):
            output_log.write('%d\t', i)
        output_log.write('Accuracy\n')

        for i in range(config.n_classes):
            output_log.write('%d\t' % i)
            for j in range(config.n_classes):
                output_log.write('%d\t' % conf[i,j])
            output_log.write('%.2f%%\n' % conf[i,i] * 100 / np.sum(conf[i]))

        output_log.close()
    except:
        print('Unable to save to split_results.txt')


def get_val_conf():
    # Returns confusion matrix for validation data.
    
    conf = np.zeros((config.n_classes, config.n_classes), dtype=np.int)

    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.save_file_name)

        data_gen = TestDataGen(config.wait_for_data, frame_skip=1)
        for _ in tqdm(range(data_gen.n_videos)):
            video, bbox, label = data_gen.get_next_video()

            # gets losses and prediction for a single video
            mloss, sloss, pred = capsnet.eval_on_vid(sess, video, bbox, label, validation=False)
            # print('Pred: %d\t| Label: %d' % pred, label)
            conf[label, pred] += 1
    return conf


if __name__=='__main__':
    output_conf(get_val_conf())

