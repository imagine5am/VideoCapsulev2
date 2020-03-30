from load_synth_data import get_det_annotations
import config2 as config

'''
Prints the count of samples according per trasformation labels
'''

data = get_det_annotations(split='train')

num_samples = {i:0 for i in range(config.n_classes)}
for sample in data:
    label = sample[1]['label']
    num_samples[label] += 1
    
print('Label\tCount\n')
for k,v in data.items():
    print('%d\t%d\n' % (config.labels[k], v))
