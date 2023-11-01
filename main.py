import logging
import torch
import numpy as np
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import FlyingSquid
from wrench.endmodel import EndClassifierModel
from utils.RefiningModule import get_graph, get_removing_list
from utils.Utils import init_random, remove_LF
from scipy.stats import sem
import os
import torch
import argparse
import time
parser = argparse.ArgumentParser()
    # required arguments
parser.add_argument('--dataset_path', default = '')
parser.add_argument('--data', default = 'youtube', type =str)
parser.add_argument('--total_lf', default = 10, type = int)
parser.add_argument('--batch_size', default = 8, type = int)
parser.add_argument('--test_batch_size', default = 512, type = int)
parser.add_argument('--optimizer_lr', default = 5e-5, type = float)
parser.add_argument('--metric', default = 'f1_binary', type = str)
parser.add_argument('--patience', default = 50, type = int)
parser.add_argument('--random_seed', nargs='+', type = int)
# parser.add_argument('--random_seed', default = [1,2])
parser.add_argument('--threshold_structure', default = 0.2, type = float)
parser.add_argument('--threshold_removing', default = 1, type =float)
parser.add_argument('--save_dir', default = './results/', type = str)
parser.add_argument('--endModel_weight_decay', default = 0.0, type = float)
args = parser.parse_args()
args = vars(args)
dataset_path = args['dataset_path']
data = args['data']
save_file = os.path.join(args['save_dir'], (args['dataset_path'].split('/')[2]+'_'+args['data']+'.txt'))
with open(save_file, 'a') as file:
    file.write( 'batch size: '+ str(args['batch_size']) +  ', test batch size: '+ str(args['test_batch_size'])+ ', learning rate: '+ str(args['optimizer_lr'])+ ', weight decay: '+ str(args['endModel_weight_decay']) +', threshold_structure: '+ str(args['threshold_structure']) +', threshold_removing: '+ str(args['threshold_removing'])+'\n')

logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

result_dict = {}
device = torch.device('cuda')
result_List = []

for pos, seed in enumerate(args['random_seed']):
    init_random(seed)
    logging.info('random seed used: %d', seed)
    logging.info('loading data...')
    result_dict['seed_{}'.format(seed)] ={}
    result_dict['seed_{}'.format(seed)]['metric'] = []
    label_model = FlyingSquid()
    LFs_removed = get_removing_list(args)
    with open(save_file, 'a') as file:
        file.write(f'Seed: {seed} \n')
    with open(save_file, 'a') as file:
        file.write('LF removed: ')
        file.write('[ ')
        for i in LFs_removed:
            file.write( str(i) +', ')
        file.write(' ]')
        file.write('\n')

    train_data, valid_data, test_data = load_dataset(
            dataset_path,
            data,
            extract_feature=False,
            extract_fn='bert',
            model_name='bert-base-cased',
            cache_name='bert'
        )
    train_data, valid_data, test_data = remove_LF(train_data, LFs_removed), remove_LF(valid_data, LFs_removed), remove_LF(test_data, LFs_removed)

    dependency_graph = get_graph(args, LFs_removed)
        

            
    with open(save_file, 'a') as file:
        file.write('dependencies: ')
        file.write('[ ')
        for i in dependency_graph:
            file.write( '( '+ str(i[0]) + ', '+str(i[1])+' )' +', ')
        file.write(' ]')
        file.write('\n')
        





    model_start = time.time()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data,
        dependency_graph = dependency_graph
    )
    with open(save_file, 'a') as file:
        file.write(f'Fit label model time: {time.time()-model_start}\n')

   

 
    total_data_num = len(train_data)
    train_data = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data)
    


    model = EndClassifierModel(
        batch_size=args['batch_size'],
        real_batch_size=args['batch_size'],  # for accumulative gradient update
        test_batch_size=args['test_batch_size'],
        n_steps=1000,
        backbone='BERT',
        backbone_model_name='roberta-base',
        backbone_max_tokens=128,
        backbone_fine_tune_layers=-1, # fine  tune all
        optimizer='AdamW',
        optimizer_lr=args['optimizer_lr'], 
        optimizer_weight_decay=0.0,
    )
    
    model.fit(
        dataset_train=train_data,
        y_train=aggregated_soft_labels,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric=args['metric'],
        patience=args['patience'],
        device=device
    )
    endmodel_test = model.test(test_data, args['metric'])
    

    




    with open(save_file, 'a') as file:
        file.write('End model' + args['metric'] +': '+ str(endmodel_test) +'\n\n')
    
    acc = model.test(test_data, args['metric'])
    logging.info('end model (Roberta) test {}: {}'.format(args['metric'], acc))
    result_dict['seed_{}'.format(seed)]['metric'].append(float(acc))
    del model
    result_List.append(result_dict['seed_{}'.format(seed)]['metric'])

with open(save_file, 'a') as file:
    file.write('Means: ')
    for i in np.mean(result_List,0):
        file.write( str(i) + ', ')
    file.write('\n')
    file.write('STD:')
    for i in sem(result_List,0):
        file.write( str(i) + ', ')
    file.write('\n\n')

print('Result', result_dict)
