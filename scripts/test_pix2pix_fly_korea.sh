set -ex
python test.py \
--dataroot ../sparse-view/data/pix2pix-sparse-32 \
--name fly_korea_pix2pix_sparse_32 \
--model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch \
--num_test 1000 \
--input_nc 1 \
--output_nc 1 \
--load_size 256 \
--crop_size 256 \
--gpu_ids 1
