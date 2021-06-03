import numpy as np
from torch.utils.data import Dataset, DataLoader
import util


def create_dataset(conf, in_img):
    dataset = DataGenerator(conf, in_img)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)
    return dataloader



class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """
    
    def __init__(self, conf, in_img):
        np.random.seed(100)
        self.conf = conf
            
        # Default shapes
        self.big_input_shape = conf.input_crop_size
        self.sml_input_shape = conf.input_crop_size // conf.scale
        
        # Prepare input image
        self.input_image = in_img / 255.
        self.shave_edges(scale_factor=conf.scale, real_image=conf.real)

        self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # Create prob map for choosing the crop
        self.crop_indices_big, self.crop_indices_sml = self.make_list_of_crop_indices(conf=conf)

    def __len__(self):
        return self.conf.finetune_iters * self.conf.batch_size

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        big_img = self.next_crop(for_big=True, idx=idx)
        sml_img = self.next_crop(for_big=False, idx=idx)
        
        # Because kernels are symmetric with respect to the Origin, we can rotate the input image by 180 degree
        if np.random.rand() < 0.4:
            big_img = np.rot90(big_img, k=2).copy()           
        if np.random.rand() < 0.4:
            sml_img = np.rot90(sml_img, k=2).copy()

        
        return {'big_img':util.im2tensor(big_img).squeeze(), 'sml_img':util.im2tensor(sml_img).squeeze()}

    def next_crop(self, for_big, idx):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        size = self.big_input_shape if for_big else self.sml_input_shape
        top, left = self.get_top_left(size, for_big, idx)
        crop_im = self.input_image[top:top + size, left:left + size, :]
        return crop_im

    def make_list_of_crop_indices(self, conf):
        iterations = conf.finetune_iters * conf.batch_size
        prob_map = self.create_prob_maps()
        crop_indices_big = np.random.choice(a=len(prob_map), size=iterations, p=prob_map)
        crop_indices_sml = np.random.choice(a=len(prob_map), size=iterations, p=prob_map)
        return crop_indices_big, crop_indices_sml

    def create_prob_maps(self):
        # Create loss maps for input image and downscaled one
        loss_map = util.create_gradient_map(self.input_image)

        # Create corresponding probability maps
        prob_map = util.create_probability_map(loss_map, self.sml_input_shape)
        return prob_map

    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            if self.input_image.shape[0] > 160 and self.input_image.shape[1] > 160:
                self.input_image = self.input_image[10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = scale_factor
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image

    def get_top_left(self, size, for_big, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_big[idx] if for_big else self.crop_indices_sml[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        # Choose even indices (to avoid misalignment with the loss map for_big)
        return top - top % 2, left - left % 2
        