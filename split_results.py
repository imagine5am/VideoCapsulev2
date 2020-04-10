from caps_network import Caps3d
from load_synth_data import SynthTrainDataGenDet as TrainDataGen, SynthTestDataGenDet as TestDataGen
import config2 as config
import tensorflow as tf
import time
from tqdm import tqdm

'''
Prints accuracy for each labels 
'''

records = {i:{'correct':0, 'incorrect':0} for i in range(config.n_classes)}

capsnet = Caps3d()
with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
    tf.global_variables_initializer().run()
    capsnet.load(sess, config.save_file_name)
    
    data_gen = TestDataGen(config.wait_for_data, frame_skip=1)
    start_time = time.time()
    for i in tqdm(range(data_gen.n_videos)):
        video, bbox, label = data_gen.get_next_video()

        # gets losses and predictionfor a single video
        mloss, sloss, pred = capsnet.eval_on_vid(sess, video, bbox, label, validation=False)
        #print('pred:', pred)
        #print('label:', label)

        if pred == label:
            records[label]['correct'] += 1
        else:
            records[label]['incorrect'] += 1

try:
    output_log = open('split_results.txt', 'w')
    output_log.write('Label\tTrue\tFalse\tAccuracy\n')
    for i in range(config.n_classes):
        output_log.write('%d\t%d\t%d\t%.4f%%\n' % (i,records[i]['correct'],records[i]['incorrect'],
                        (records[i]['correct'] * 100)/(records[i]['correct']+records[i]['incorrect'])))
    output_log.close()
except:
    print('Unable to save to pred_recs.txt')

