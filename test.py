import os
import cv2
import random
import argparse

import torch
import torch.nn as nn

import options.options as option
from data import create_dataloader
from data import create_dataset
from networks import create_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, default='scripts/train.json', help='Path to options JSON file.')
    parser.add_argument('-mask', type=str, required=True, help='Path to options JSON file.')
    parser.add_argument('-checkpoint', type=str, required=True, help='Path to options JSON file.')
    parser.add_argument('-dataRoot', type=str, required=True, help='Path to options JSON file.')
    parser.add_argument('-output', type=str, default='./results', help='Path to options JSON file.')
    parser.add_argument('-test_mode', type=str, default='eswa', help='eswa?')
    command_args = parser.parse_args()
    opt = option.parse(command_args.opt)
    opt = option.dict_to_nonedict(opt)
    opt['mask'] = command_args.mask # not necessary
    opt['checkpoint'] = command_args.checkpoint
    opt['datasets']["test"]['dataRoot'] = command_args.dataRoot
    opt['output'] = command_args.output
    opt['test_mode'] = command_args.test_mode

    return opt

def test_single_mask(args, mask_path, output_folder):
    args['datasets']["test"]['mask_file'] = mask_path
    dataset_opt = args['datasets']["test"]

    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    
    # model = create_model(args).cuda()
    model = create_model(args).to(device)
    model.load_state_dict(torch.load(args['checkpoint']))

    img_names = [_.split('/')[-1].split('\n')[0] for _ in open(args['datasets']["test"]['dataRoot'], 'r').readlines()]

    
    for iter, batch in enumerate(test_loader):
        # masked_imgs, gt_imgs, masks = [_.cuda() for _ in batch]
        masked_imgs, gt_imgs, masks = [_.to(device) for _ in batch]

        masked_imgs = masked_imgs * 2 - 1 # [0, 1] to [-1, 1]
        gt_imgs = gt_imgs * 2 - 1

        input_tensor = torch.cat((masked_imgs, masks), 1)
        with torch.no_grad():
            # pred_imgs, log_uncertainty_maps = model(input_tensor)
            pred_imgs, uncertainty_maps, mutual_infos = model(input_tensor, masks)
        
        comp_imgs = [pred_imgs[i] * masks + gt_imgs * (1 - masks) for i in range(len(pred_imgs))]
        
        final_img = comp_imgs[-1]
        final_img = ((final_img + 1) * 127.5)[0].permute(1,2,0).detach().cpu().numpy()
        cv2.imwrite(os.path.join(output_folder, img_names[iter]), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        # import ipdb; ipdb.set_trace()
        if iter%1000 == 0:
            print("Processing No. ", iter)



if __name__ == "__main__":
    args = parse_args()
    if args['test_mode'] in ['eswa', 'ESWA']:
        print('TEST MODE IS ESWA')
        if not os.path.isdir(args['mask']):
            masks = [args['mask']]
            mask_names = [args['mask'].split('/')[-1]]
        else:
            mask_names = os.listdir(args['mask'])
            masks = [os.path.join(args['mask'], _) for _ in mask_names]
        
        for i in range(len(masks)):
            mask = masks[i]
            # import ipdb; ipdb.set_trace()
            # output_folder = os.path.join(args['output'], mask_names[i].split('.')[0])
            output_folder = os.path.join(args['output'], '.'.join(mask_names[i].split('.')[:-1]))
            os.makedirs(output_folder, exist_ok=True)
            test_single_mask(args, mask, output_folder)
    elif args['test_mode'] in ['pr', 'PR']: # facial mask
        output_folder = os.path.join(args['output'])
        os.makedirs(output_folder, exist_ok=True)
        test_single_mask(args, args['mask'], output_folder)
