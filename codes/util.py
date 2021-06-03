import os
import torch
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation



def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()
    

def tensor2im(im_t, normalize_en = False, uint16_en = False):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.transpose(move2cpu(im_t[0]), (1, 2, 0))  
    if normalize_en:
        im_np = (im_np + 1.0) / 2.0
    
    if uint16_en:
        im_np = np.clip(np.round(im_np * 65535.0), 0, 65535).astype(np.uint16)
    else:
        im_np = np.clip(np.round(im_np * 255.0), 0, 255).astype(np.uint8)
    
    return im_np


def im2tensor(im_np, normalize_en = False, cuda_en=True):
    """Copy the image to the gpu & converts to range [0,1]"""
    if im_np.dtype == 'uint8':
        im_np = im_np / 255.0
    elif im_np.dtype == 'uint16':
        im_np = im_np / 65535.0 
    
    if normalize_en:
        im_np = im_np * 2.0 - 1.0
    out = torch.FloatTensor(np.transpose(im_np, (2, 0, 1))).unsqueeze(0)
    if cuda_en:
        return out.cuda()
    else:
        return out


def quantize_image(img):
    return torch.clamp(torch.round(img * 255.0), 0, 255) / 255.0

  
def read_image(path):
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im
    
def save_image(path, img):
    assert(img.dtype == np.uint8)
    Image.fromarray(img).save(os.path.join(path))


def rgb2gray(im):
    """Convert and RGB image to gray-scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def make_1ch(im):
    s = im.shape
    assert s[1] == 3
    return im.reshape(s[0] * 3, 1, s[2], s[3])


def make_3ch(im):
    s = im.shape
    assert s[1] == 1
    return im.reshape(s[0] // 3, 3, s[2], s[3])


def shave_a2b(a, b):
    """Given a big image or tensor 'a', shave it symmetrically into b's shape"""
    # If dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = (type(a) == torch.Tensor)
    r = 2 if is_tensor else 0
    c = 3 if is_tensor else 1
    
    assert (a.shape[r] >= b.shape[r]) and (a.shape[c] >= b.shape[c])
    assert ((a.shape[r] - b.shape[r]) % 2 == 0) and ((a.shape[c] - b.shape[c]) % 2 == 0)
    # Calculate the shaving of each dimension
    shave_r, shave_c = max(0, a.shape[r] - b.shape[r]), max(0, a.shape[c] - b.shape[c])
    return a[:, :, shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2] if is_tensor \
        else a[shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2]


def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / prob_map.flatten().shape[0]
    return prob_vec


def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min
    

def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def nn_interpolation(im, sf):
    """Nearest neighbour interpolation"""
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST), dtype=im.dtype)


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    #kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel

      
def downscale_with_kernel(hr_img, kernel, stride=2, padding_en = False):
    hr_img = make_1ch(hr_img)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    if padding_en:
        pad = (kernel.shape[-1] - 1) // 2
        hr_img = F.pad(hr_img, (pad, pad, pad, pad), mode='reflect')
        
    lr_img = F.conv2d(hr_img, kernel, stride=stride)
    lr_img = make_3ch(lr_img)
    return lr_img


def cal_y_psnr(A, B, border):
    A = A.astype('float64')
    B = B.astype('float64')
    
    if len(A.shape) == 3:
        # calculate Y channel like matlab 'rgb2ycbcr' function
        Y_A = np.dot(A / 255., [65.481, 128.553, 24.966]) + 16
        Y_B = np.dot(B / 255., [65.481, 128.553, 24.966]) + 16
    else:
        Y_A = A
        Y_B = B
    
    Y_A = Y_A[border:-border,border:-border]
    Y_B = Y_B[border:-border,border:-border]
    
    e=Y_A-Y_B;
    mse=np.mean(e**2);
    psnr_cur=10*np.log10(255*255/mse);
    
    return psnr_cur
    
def gen_kernel(k_params, k_size=13, scale_factor=2, noise_level=0):
    
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = k_params[0]
    lambda_2 = k_params[1]
    theta = k_params[2]
    
    k_size=np.array([k_size, k_size])
    scale_factor=np.array([scale_factor, scale_factor])
    
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2
    
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
    
    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2  + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]
    
    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]
    
    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    
    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
    raw_kernel_centered = raw_kernel
    
    # Normalize the kernel and return
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    return kernel
