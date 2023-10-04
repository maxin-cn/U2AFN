import argparse, random
from tqdm import tqdm
import numpy as np
import random
import cv2

import torch

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)

    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    print("===> Random Seed: [%d]"%seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)

    # create train and val dataloader
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val_set.name(), len(val_set)))

        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)

    solver = create_solver(opt)

    model_name = opt['networks']['which_model'].upper()

    print('===> Start Train')
    print("==================================================")

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver.current_epoch

    print("Method: %s || Epoch Range: (%d ~ %d)"%(model_name, start_epoch, NUM_EPOCH))

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> Training Epoch: [%d/%d]...'%(epoch, NUM_EPOCH))

        solver.current_epoch = epoch

        # Train model
        train_loss_list = []
        with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1, ncols=120) as t:
            for iter, batch in enumerate(train_loader):
                solver.feed_data(batch)
                iter_loss = solver.train_step()
                
                train_loss_list.append(iter_loss*opt['datasets']['train']['batch_size'])
                t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
                t.update()

                solver.save()

        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f' % (epoch,
                                                    NUM_EPOCH,
                                                    sum(train_loss_list)/len(train_set)))
        
        solver.save(enforce=True)
        
    print('===> Finished !')


if __name__ == '__main__':
    main()