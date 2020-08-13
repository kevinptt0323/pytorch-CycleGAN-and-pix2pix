DATA_ROOT=../sparse-view/data/PLS6C/train
MODEL_NAME=u-nn-20-256x2560

set -ex
python train.py \
--dataroot ${DATA_ROOT} \
--name ${MODEL_NAME} \
--model trap_pix2pix \
--netT uniform \
--netG unet_256 --direction BtoA \
--lambda_L1 100 --lambda_TV 0 --gan_mode vanilla \
--dataset_mode single --norm batch --pool_size 0 --num_data 10000 \
--lr 0.0002 --niter 100 --niter_decay 100 \
--input_nc 1 \
--output_nc 1 \
--load_size 256,2560 \
--crop_size 256,2560 \
--gpu_ids 0 \
--alpha_min 0.08 --alpha_max 0.08 \
--batch_size 8 \
--display_port 8097 \
--display_ncols 1 \
--num_threads 8
# --epoch 400 --epoch_count 401 --continue_train
