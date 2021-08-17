import argparse
import os
import numpy as np

class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='DualSR')

        # Directories
        self.parser.add_argument('--in_dir', '-i', type=str, default='../datasets/DIV2KRK/lr_x4', help='path to input images directory')
        self.parser.add_argument('--out_dir', '-o', type=str, default='../results/DIV2KRK/x4' , help='path to output images directory')
        self.parser.add_argument('--gt_dir', '-g', type=str, default='', help='path to grand-truth images directory')
        
        self.parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='path to the pretrained models')
        
        # DualSR fine-tuning parameters
        self.parser.add_argument('--input_crop_size', type=int, default=128, help='crop size for finetuning')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
        self.parser.add_argument('--scale', type=int, default=4, help='the upscaling scale factor')
        
        self.parser.add_argument('--lambda_cycle', type=int, default=5, help='lambda parameter for cycle consistency losses')
        self.parser.add_argument('--lambda_interp', type=int, default=1, help='lambda parameter for masked interpolation loss')
        self.parser.add_argument('--lambda_regularization', type=int, default=2, help='lambda parameter for downsampler regularization term')
        
        self.parser.add_argument('--max_lr', type=float, default=0.00001, help='maximum learning rate in OneCycle learning rate policy')
        self.parser.add_argument('--div_factor', type=float, default=10, help='divide factor in OneCycle learning rate policy')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')
        
        self.parser.add_argument('--finetune_iters', type=int, default=1000, help='number of fine-tuning iterations')
        
        self.parser.add_argument('--real', action='store_true', help='configuration is for real images, we add some noise to the downscaled image')
        self.parser.add_argument('--noise_std', type=float, default=0.04, help='standard deviation of noise for real images')
        
        self.conf = self.parser.parse_args()
        
        if not os.path.exists(self.conf.out_dir):
            os.makedirs(self.conf.out_dir)
            

        
        
    def get_config(self, img_name):
        self.conf.abs_img_name = os.path.splitext(img_name)[0]
        self.conf.input_image_path = os.path.join(self.conf.in_dir, img_name)
        self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) if self.conf.gt_dir != '' else None
        
        if self.conf.scale == 2:
            self.conf.lambda_set = [0.85, 1.47, 2.27, 3.23, 4.37]
            self.conf.ker_size = 13
        else:
            self.conf.lambda_set = [4.25, 7.35, 11.35, 16.15, 21.85]
            self.conf.ker_size = 33
            
        self.conf.theta_set = [np.pi/16, 3*np.pi/16, 5*np.pi/16, 7*np.pi/16, 9*np.pi/16, 11*np.pi/16, 13*np.pi/16, 15*np.pi/16]
        
        self.conf.sr_networks_path = os.path.join(self.conf.checkpoints_dir, 'x%d'%(self.conf.scale), 'SR_networks_ensemble.pt')
        self.conf.ise_path = os.path.join(self.conf.checkpoints_dir, 'x%d'%(self.conf.scale), 'ISE.pt')
        self.conf.ken_path = os.path.join(self.conf.checkpoints_dir, 'x%d'%(self.conf.scale), 'KEN.pt')
        
        print('*' * 60 + '\nRunning MoESR ...')
        print('input image: \'%s\'' %self.conf.input_image_path)
        print('grand-truth image: \'%s\'' %self.conf.gt_path)
        print('real image: \'%r\'' %self.conf.real)
        return self.conf
    
