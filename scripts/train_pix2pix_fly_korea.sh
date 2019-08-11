set -ex
python train.py \
--dataroot ../sparse-view/data/pix2pix-sparse-32 \
--name fly_korea_pix2pix_sparse_test \
--model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 \
--lr 0.0002 --niter 200 --niter_decay 200 \
--input_nc 1 \
--output_nc 1 \
--load_size 256 \
--crop_size 256 \
--gpu_ids 0,1
# --epoch 100 --epoch_count 101 --continue_train \
