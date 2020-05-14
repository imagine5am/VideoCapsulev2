import matplotlib.pyplot as plt
import re

def plot(ylabel, x1, x2=None):
    plt.clf()
    # fig = plt.figure()
    # ax = plt.axes()
    # plt.style.use('seaborn-whitegrid')
    x_vals = range(1, len(x1)+1)
    if x2 == None:
        plt.plot(x_vals, x1, color='blue', linestyle='solid')
    else:
        plt.plot(x_vals, x1, color='blue', linestyle='solid', label='train')
        plt.plot(x_vals, x2, color='red', linestyle='solid', label='validation')
        plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.xticks(range(1, len(x1)+2, 5)) 
    plt.savefig(ylabel+'.png')


def parse():
    values = {'train': {'CL':[], 'SL':[], 'ACC':[]},
                'val': {'CL':[], 'SL':[], 'ACC':[]},
             }
    with open('output7.txt', 'r') as file:
        num_training_samples = 10092
        batch_size = 8
        training_batches = num_training_samples // batch_size + min(num_training_samples % batch_size, 1)
        for line in file:
            line = line.strip()
            if line.startswith('Training'):
                cl, sl, acc = map(float, re.findall(r"\d+\.\d+", line))
                values['train']['CL'].append(cl)
                values['train']['SL'].append(sl)
                values['train']['ACC'].append(acc * 100)
            elif line.startswith('Validation'):
                cl, sl, acc = map(float, re.findall(r"\d+\.\d+", line))
                values['val']['CL'].append(cl)
                values['val']['SL'].append(sl)
                values['val']['ACC'].append(acc * 100)
    return values
    

def moving_average(nums, n):
    return [sum(nums[idx:idx+n])/n for idx in range(0, len(nums)-n+1)]

def min_max_norm(nums):
    _min = min(nums)
    _max = max(nums)
    dm = _max - _min
    return [(num-_min)/dm for num in nums]
    

if __name__=='__main__':
    values = parse()
    print('Training')
    print(values['train'])
    print('Validation')
    print(values['val'])
    plot('Training Segmentation Loss', values['train']['SL'])
    # plot('Training Classification Loss', values['train']['CL'])
    plot('Validation Segmentation Loss', values['val']['SL'])
    # plot('Validation Classification Loss', values['val']['CL'])
    # plot('Accuracy', values['train']['ACC'], values['val']['ACC'])
    
    # divergence = [values['val']['SL'][idx]-values['train']['SL'][idx] for idx in range(len(values['train']['SL']))]
    train_sl_norm = min_max_norm(values['train']['SL'])
    val_sl_norm = min_max_norm(values['val']['SL'])
    divergence = [val1-val2 for val1, val2 in zip(val_sl_norm, train_sl_norm)]
    ma_n = 8
    ma_divergence = moving_average(divergence, ma_n)
    plot('Segmentation Loss Divergence (ma=%d)' % ma_n, ma_divergence)
    