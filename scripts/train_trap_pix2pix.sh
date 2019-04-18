set -ex
python train.py \
--dataroot ../sparse-view/data/trap-pix2pix/train \
--name fly_korea_trap_pix2pix \
--model trap_pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode single --norm batch --pool_size 0 \
--lr 0.0002 --niter 200 --niter_decay 200 \
--input_nc 1 \
--output_nc 1 \
--load_size 256 \
--crop_size 256 \
--num_threads 8 \
--gpu_ids 0,1
# --epoch 100 --epoch_count 101 --continue_train \
