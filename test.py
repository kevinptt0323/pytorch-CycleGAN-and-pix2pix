"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_data images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import numpy as np
import random
import csv
import ntpath
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util.util import reconstruct_tomo, tensor2im
from util import html
from skimage.measure import compare_psnr as psnr, compare_ssim as ssim, compare_nrmse as nrmse
from dominate.tags import p, br

def reconstruct_result(model):
    shift_x = 1
    for name in ['real_A', 'fake_B', 'real_B']:
        image = getattr(model, name)
        if name == 'real_A':
            mask = getattr(model, name + '_mask')
            start = -1
            interval = -1
            for i, m in enumerate(mask[0,0]):
                if m[0] > 0:
                    if start == -1:
                        start = i
                    else:
                        interval = i - start 
                        break
            theta = np.linspace(0., 180., mask.shape[2], endpoint=False)[start::interval]
            image = image[:,:,start::interval]
            tomo, sino = reconstruct_tomo(image, shift_x=shift_x, theta=theta)
        else:
            tomo, sino = reconstruct_tomo(image, shift_x=shift_x)
        tomo3 = np.stack((tomo,) * 3, axis=-1)
        setattr(model, name + '_tomo', tomo3)

def tomo_desc(webpage, model):
    with webpage.doc:
        _p = p()
        d = -1
        for i, mask in enumerate(model.real_A_mask[0,0]):
            if mask[0] > 0:
                if d == -1:
                    d = i
                else:
                    d = i - d
                    break
        _p.add('d: 1/{}'.format(d))
        _p.add(br())
        real_B = tensor2im(model.real_B_tomo)[:,:,0]
        for name in ['real_A', 'fake_B']:
            image = tensor2im(getattr(model, name + '_tomo'))[:,:,0]
            PSNR, SSIM, NRMSE = psnr(real_B, image), ssim(real_B, image), nrmse(real_B, image)
            _p.add('{} : PSNR: {:.4}, SSIM: {:.4}, NRMSE: {:.4}'.format(name + '_tomo', PSNR, SSIM, NRMSE))
            _p.add(br())
    return [d, PSNR, SSIM, NRMSE]

if __name__ == '__main__':
    torch.manual_seed(7122)
    torch.cuda.manual_seed_all(7122)
    np.random.seed(7122)
    random.seed(7122)
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    model.visual_names += [name + '_tomo' for name in ['real_A', 'fake_B', 'real_B']]
    if opt.alpha_min == opt.alpha_max:
        csvName = 'result_{}.csv'.format(opt.alpha_min)
    else:
        csvName = 'result_{}_{}.csv'.format(opt.alpha_min, opt.alpha_max)
    with open(os.path.join(web_dir, csvName), 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([opt.name, 'd', 'PSNR', 'SSIM', 'NRMSE'])
        PSNR, SSIM, NRMSE = [], [], []
        for i, data in enumerate(dataset):
            if i >= opt.num_data:  # only apply our model to opt.num_data images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            reconstruct_result(model)
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            csvData = tomo_desc(webpage, model) # d, PSNR, SSIM, NRMSE
            PSNR.append(csvData[1])
            SSIM.append(csvData[2])
            NRMSE.append(csvData[3])

            short_path = ntpath.basename(img_path[0])
            name = os.path.splitext(short_path)[0]
            writer.writerow([name] + csvData)

        avg_d = csvData[0] if opt.alpha_min == opt.alpha_max else ''
        writer.writerow(['AVERAGE', avg_d, np.mean(PSNR), np.mean(SSIM), np.mean(NRMSE)])
        writer.writerow(['STD', '', np.std(PSNR), np.std(SSIM), np.std(NRMSE)])
        # writer.writerow(['AVERAGE','','=AVERAGE(C2:C101)','=AVERAGE(D2:D101)','=AVERAGE(E2:E101)'])
    webpage.save()  # save the HTML
