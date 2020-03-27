import matplotlib.pyplot as plt
import re

def plot(ylabel, train, val):
    # fig = plt.figure()
    # ax = plt.axes()
    # plt.style.use('seaborn-whitegrid')
    x_vals = range(1, len(train)+1)
    plt.plot(x_vals, train, color='blue', linestyle='solid', label='train')
    plt.plot(x_vals, val, color='red', linestyle='solid', label='validation')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(x_vals) 
    plt.savefig(ylabel+'.png')


def parse():
    values = {'train': {'CL':[], 'SL':[], 'ACC':[]},
                'val': {'CL':[], 'SL':[], 'ACC':[]},
             }
    with open('output2.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Training\t'):
                cl, sl, acc = map(float, re.findall(r"\d+\.\d+", line))
                values['train']['CL'].append(cl)
                values['train']['SL'].append(sl)
                values['train']['ACC'].append(acc)
            elif line.startswith('Validation\t'):
                cl, sl, acc = map(float, re.findall(r"\d+\.\d+", line))
                values['val']['CL'].append(cl)
                values['val']['SL'].append(sl)
                values['val']['ACC'].append(acc)
    return values
    
    
if __name__=='__main__':
    values = parse()
    plot('Segmentation Loss', train=values['train']['SL'], val=values['val']['SL'])
    plot('Classification Loss', train=values['train']['CL'], val=values['val']['CL'])
    plot('Segmentation Loss', train=values['train']['ACC'], val=values['val']['ACC'])
# x2 = [10,20,30]
# y2 = [40,10,30]
# plot('acc', x2, y2)
    
    