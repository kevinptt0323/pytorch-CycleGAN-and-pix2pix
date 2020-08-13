import random
import torch
from .base_model import BaseModel
from . import networks


class TrapPix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--alpha_min', type=float, default=0.2, help='min proportion of retained data')
        parser.add_argument('--alpha_max', type=float, default=0.2, help='max proportion of retained data')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_TV', type=float, default=10.0, help='weight for TV loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_fake', 'G_L1', 'D_real', 'D_fake', 'G_TV', 'D_gp']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        if opt.netT in ['sentence', 'uniform-sentence']:
            self.loss_names.append('T_fake', 'T_L1')
            self.model_names.append('T')

        self.rand_alpha = lambda: 1 / random.randint(int(1 / opt.alpha_max + 0.5), int(1 / opt.alpha_min + 0.5))

        # define networks (both generator, discriminator, and trapper)
        mask_nc = 1 if opt.mask else 0
        self.netG = networks.define_G(opt.input_nc + mask_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netT = networks.define_T(opt.input_nc, opt.crop_size, opt.netT, opt.ntf, opt.stf, self.rand_alpha,
                                      opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc + mask_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, 'none', opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss functions
            # self.criterionGAN_G = self.criterionGAN_D = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGAN_G = networks.GANLossV2(opt.gan_mode, 'G', 'S').to(self.device)
            self.criterionGAN_D = networks.GANLossV2(opt.gan_mode, 'D', 'S').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if 'T' in self.model_names:
                self.optimizer_T = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
                self.optimizers.append(self.optimizer_T)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device) if 'B' in input else None
        if not AtoB:
            self.real_A, self.real_B = self.real_B, self.real_A

        self.image_paths = input['A_paths']
        if not AtoB and 'B_paths' in input:
            self.image_paths = input['B_paths']

    def forward(self, detach_A=True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_A, self.real_A_mask = self.netT(self.real_B)
        if self.opt.mask:
            self.real_A_input = torch.cat([self.real_A, self.real_A_mask], dim=1)
        else:
            self.real_A_input = self.real_A
        if detach_A:
            self.real_A_input.detach_()
        self.fake_B = self.netG(self.real_A_input)  # G(A)
        # self.fake_B = torch.where(self.real_A_mask > 0, self.real_A, self.fake_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A_input, self.fake_B), 1)
        real_AB = torch.cat((self.real_A_input, self.real_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        pred_real = self.netD(real_AB)

        self.loss_D_fake, self.loss_D_real = self.criterionGAN_D(pred_fake, pred_real)

        # WGAN-GP
        if self.opt.gradient_penalty:
            gradient_penalty, gradients = networks.cal_gradient_penalty(self.netD, real_AB, fake_AB.detach(), self.device)
            self.loss_D_gp = gradient_penalty
        else:
            self.loss_D_gp = 0.

        # combine loss and calculate gradients
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_gp
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A_input, self.fake_B), 1)
        real_AB = torch.cat((self.real_A_input, self.real_B), 1)
        pred_fake = self.netD(fake_AB)
        pred_real = self.netD(real_AB)

        self.loss_G_fake, _ = self.criterionGAN_G(pred_fake, pred_real)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G_TV = self.criterionTV(self.fake_B) * self.opt.lambda_TV

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_fake + self.loss_G_L1 + self.loss_G_TV
        self.loss_G.backward()

    def backward_T(self):
        fake_AB = torch.cat((self.real_A_input, self.fake_B), 1)
        real_AB = torch.cat((self.real_A_input, self.real_B), 1)
        pred_fake = self.netD(fake_AB)
        pred_real = self.netD(real_AB)

        self.loss_T_fake = -self.criterionGAN_G(pred_fake, pred_real)[0]

        self.loss_T_L1 = -1 * self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_T = self.loss_T_fake + self.loss_T_L1
        self.loss_T.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        for iter_d in range(self.opt.update_d_num):
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        if 'T' in self.model_names:
            # update T
            self.forward(detach_A=False)
            self.optimizer_T.zero_grad()
            self.backward_T()
            self.optimizer_T.step()
