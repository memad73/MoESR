import networks
import util
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch_sobel import Sobel
import itertools
from data import create_dataset
import tqdm



class EnsSR:
            
    def __init__(self, conf):
        self.conf = conf
        
        # Read input and ground-truth image
        self.in_img = util.read_image(conf.input_image_path)
        self.gt_img = util.read_image(conf.gt_path) if conf.gt_path is not None else None
        
        self.in_img_t= util.im2tensor(self.in_img)
        
        # Load models in the ensemble
        self.ensemble_models = torch.load(self.conf.sr_networks_path)
    
    
    def estimate_kernel(self):       
        # Define and load models
        U1 = networks.SR_Net().cuda()
        if self.conf.scale == 4:
            U2 = networks.SR_Net().cuda()
        
        ISE = networks.ISE_Net().cuda()
        ISE.load_state_dict(torch.load(self.conf.ise_path))
        
        KEN = networks.KEN_Net().cuda()
        KEN.load_state_dict(torch.load(self.conf.ken_path))
        
        # Generate bicubic upsampled image
        bq_img_t = F.interpolate(self.in_img_t, scale_factor=self.conf.scale, mode='bicubic')
          
        errs = []
        
        # Loop over all SR networks in the ensemble
        for lambda_1 in self.conf.lambda_set:
            for lambda_2 in self.conf.lambda_set:
                for theta in self.conf.theta_set:
                    # Lambda_1 is the biger eigenvalue
                    if lambda_2 > lambda_1:
                        continue
                    # For isotropic kernels, theta is euqal to 0
                    if lambda_2 == lambda_1:
                        if theta != self.conf.theta_set[0]:
                            continue
                        theta = 0
                    
                    # For theta > pi/4, we need to rotate/flip input image
                    if theta < np.pi/4:
                        model_theta = theta
                        I = self.in_img_t
                    elif theta > np.pi/4 and theta < np.pi/2:
                        model_theta = np.pi/2 - theta
                        I = self.in_img_t.flip(3).rot90(dims=[2, 3])
                    elif theta > np.pi/2 and theta < 3*np.pi/4:
                        model_theta = theta - np.pi/2
                        I = self.in_img_t.rot90(dims=[2, 3])
                    else:
                        model_theta = np.pi - theta
                        I = self.in_img_t.flip(3)
                    
                    # Load weights for SR network
                    model_id = 'U_%.2f_%.2f_%.2f'%(lambda_1, lambda_2, model_theta)
                    U1.load_state_dict(self.ensemble_models[model_id][0])
                    if self.conf.scale == 4:
                        U2.load_state_dict(self.ensemble_models[model_id][1])
                    
                    # Run SR network
                    with torch.no_grad():
                        O = U1(I)
                        if self.conf.scale == 4:
                            O = U2(O)
                    
                    # Rotate/flip back the upsampled image to the original position
                    if theta < np.pi/4:
                        sr_img_t = O
                    elif theta > np.pi/4 and theta < np.pi/2:
                        sr_img_t = O.flip(3).rot90(dims=[2, 3])
                    elif theta > np.pi/2 and theta < 3*np.pi/4:
                        sr_img_t = O.rot90(-1, dims=[2, 3])
                    else:
                        sr_img_t = O.flip(3)
                        
                    # Run ISE on the upsampled image
                    with torch.no_grad():
                        delta = ISE(sr_img_t, bq_img_t)
                    
                    # Store mean square of ISE outputs
                    errs += [(delta*delta).mean()]
            
        errs_t = torch.stack(errs).unsqueeze(0)
        
        # Run KEN on ISE errors
        with torch.no_grad():
            rho_t = KEN(errs_t)
        self.ker_params = rho_t.squeeze().cpu().numpy()
        print('Estimated kernel parameters: [%.2f %.2f %.2f]'%(self.ker_params[0], self.ker_params[1], self.ker_params[2]))
        
        
        
    def finetune_network(self):              
        # Define the networks
        self.U1 = networks.SR_Net().cuda()
        if self.conf.scale == 4:
            self.U2 = networks.SR_Net().cuda()

        # Losses
        self.loss = torch.nn.L1Loss()

        # Optimizers
        if self.conf.scale == 4:
            U_optimizer = torch.optim.Adam(itertools.chain(self.U1.parameters(), self.U2.parameters()), betas=(self.conf.beta1, 0.999))
        else:
            U_optimizer = torch.optim.Adam(self.U1.parameters(), betas=(self.conf.beta1, 0.999))
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(U_optimizer, max_lr=self.conf.max_lr, total_steps=self.conf.finetune_iters, div_factor=self.conf.div_factor)
         
        self.blur_kernel = util.gen_kernel(self.ker_params, k_size=self.conf.ker_size, scale_factor=self.conf.scale)
        self.blur_kernel_t = torch.FloatTensor(self.blur_kernel).cuda()
        
        if self.conf.scale == 4:
            self.ker_params_2x = np.array([self.ker_params[0]/5, self.ker_params[1]/5, self.ker_params[2]])
            self.blur_kernel_2x = util.gen_kernel(self.ker_params_2x, k_size=13, scale_factor=2)
            self.blur_kernel_2x_t = torch.FloatTensor(self.blur_kernel_2x).cuda()

        lambda_1 = min(self.conf.lambda_set, key=lambda x:abs(x-self.ker_params[0]))
        lambda_2 = min(self.conf.lambda_set, key=lambda x:abs(x-self.ker_params[1]))
        theta = min(self.conf.theta_set, key=lambda x:abs(x-self.ker_params[2]))
        
        if theta < np.pi/4:
            model_theta = theta
        elif theta > np.pi/4 and theta < np.pi/2:
            model_theta = np.pi/2 - theta
        elif theta > np.pi/2 and theta < 3*np.pi/4:
            model_theta = theta - np.pi/2
        else:
            model_theta = np.pi - theta
        
        if lambda_2 > lambda_1:
            lambda_1 = lambda_2
        if lambda_1 == lambda_2:
            model_theta = 0
            
        # Load weights for SR network
        model_id = 'U_%.2f_%.2f_%.2f'%(lambda_1, lambda_2, model_theta)
        self.U1.load_state_dict(self.ensemble_models[model_id][0])
        if self.conf.scale == 4:
            self.U2.load_state_dict(self.ensemble_models[model_id][1])
        
        # Instead of rotate/flip the LR image, we rotate/flip the network weights
        if theta < np.pi/4:
            pass
        elif theta > np.pi/4 and theta < np.pi/2:
            self.U1.apply(networks.weights_flip_rotate)
            if self.conf.scale == 4:
                self.U2.apply(networks.weights_flip_rotate)
        elif theta > np.pi/2 and theta < 3*np.pi/4:
            self.U1.apply(networks.weights_rotate)
            if self.conf.scale == 4:
                self.U2.apply(networks.weights_rotate)
        else:
            self.U1.apply(networks.weights_flip)
            if self.conf.scale == 4:
                self.U2.apply(networks.weights_flip)
                
        dataloader = create_dataset(self.conf, self.in_img)
        
        for data in tqdm.tqdm(dataloader):
            U_optimizer.zero_grad()
            self.forward_and_backward(data)
            U_optimizer.step()
            lr_scheduler.step()
    
    
    def forward_and_backward(self, data):
        # Set input data
        real_HR = data['big_img']
        real_LR = data['sml_img']
        real_LR_bicubic = F.interpolate(real_LR, scale_factor=self.conf.scale, mode='bicubic')
        
        # DualSR forward cycle
        fake_HR = self.U1(real_LR)
        if self.conf.scale == 4:
            fake_HR = self.U2(fake_HR)
        rec_LR = util.downscale_with_kernel(fake_HR, self.blur_kernel_t, stride=self.conf.scale)
        
        # DualSR backward cycle
        fake_LR = util.downscale_with_kernel(real_HR, self.blur_kernel_t, stride=self.conf.scale)
        if self.conf.real:
            # If the input is real, we add a small noise to the downsampled version
            fake_LR += torch.randn(fake_LR.shape, device='cuda') * self.conf.noise_std
        fake_LR = util.quantize_image(fake_LR)
        rec_HR = self.U1(fake_LR.detach())
        if self.conf.scale == 4:
            rec_HR_i = rec_HR
            rec_HR = self.U2(rec_HR)
        
        # Losses
        loss_cycle_forward = self.loss(rec_LR, util.shave_a2b(real_LR, rec_LR)) * self.conf.lambda_cycle
        loss_cycle_backward = self.loss(rec_HR, util.shave_a2b(real_HR, rec_HR)) * self.conf.lambda_cycle
        if self.conf.scale == 4:
            fake_LR_i = util.downscale_with_kernel(real_HR, self.blur_kernel_2x_t, stride=2)
            loss_cycle_backward += self.loss(rec_HR_i, util.shave_a2b(fake_LR_i, rec_HR_i)) * self.conf.lambda_cycle
        
        sobel_A = Sobel()(real_LR_bicubic.detach())
        loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
        loss_interp = self.loss(fake_HR * loss_map_A.detach(), real_LR_bicubic * loss_map_A.detach()) * self.conf.lambda_interp
        
        total_loss = loss_cycle_backward + loss_cycle_forward + loss_interp
        total_loss.backward()

        
        
    def eval(self):
        with torch.no_grad():
            sr_img_t = self.U1(self.in_img_t)
            if self.conf.scale == 4:
                sr_img_t = self.U2(sr_img_t)
        
        sr_img = util.tensor2im(sr_img_t)
        
        test_psnr = None
        if self.gt_img is not None:
            test_psnr = util.cal_y_psnr(sr_img, self.gt_img, border=self.conf.scale)
            print('PSNR = ', test_psnr)
        
        util.save_image(os.path.join(self.conf.out_dir, '%s_x%d.png' %(self.conf.abs_img_name, self.conf.scale)), sr_img)
        print('*' * 60 + '\nOutput is saved in \'%s\' folder\n' % self.conf.out_dir)
        return test_psnr
    