import os
from options import options
from MoESR import MoESR
import numpy as np



def main():
    opt = options()
    
    # Write PSNR values in a log file
    log_f = open(os.path.join(opt.conf.out_dir, 'log.txt'),'w')
    psnrs = []
    
    # Run MoESR on all images in the input directory
    for img_name in sorted(os.listdir(opt.conf.in_dir)):   #['img_%03d.png'%(i) for i in range(1, 11)]:
        if img_name.endswith(".png"):
            conf = opt.get_config(img_name)
            
            model = MoESR(conf)
            model.estimate_kernel()
            model.finetune_network()
            psnr = model.eval()
            
            if conf.gt_path is not None:
                log_f.write('%s:   psnr=%6f\n' %(conf.abs_img_name, psnr))
                psnrs.append(psnr)
       
    if len(psnrs) > 0:
        average_psnr = np.array(psnrs).mean()
        print('Average PSNR: sr=%6f' %(average_psnr))
        log_f.write('Average PSNR: sr=%6f\n' %(average_psnr))
    log_f.close()


if __name__ == '__main__':
    main()
