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
    divergence = [values['train']['SL'][idx]-values['val']['SL'][idx] for idx in range(len(values['train']['SL']))]
    plot('Segmentation Loss Divergence', divergence)
    