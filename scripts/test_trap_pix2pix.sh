# DATA_ROOT=../sparse-view/data/sino/train/20151110_Fly_Korea
DATA_ROOT=../sparse-view/data/PLS6C/test200
TEST_PHASE=all
MODEL_NAME=${1:-uniformsentence}
ALPHA_MIN=${2:-0.05}
ALPHA_MAX=${2:-0.2}

set -ex
python test.py \
--dataroot ${DATA_ROOT} \
--name ${MODEL_NAME} \
--checkpoints_dir ./checkpoints/ \
--results_dir ./results/${TEST_PHASE}/ \
--model trap_pix2pix --eval \
--netT uniform --netG unet_256 --direction BtoA --dataset_mode single --norm batch \
--serial_batches \
--num_data 200 \
--input_nc 1 \
--output_nc 1 \
--load_size 256,2560 \
--crop_size 256,2560 \
--gpu_ids 1 \
--alpha_min ${ALPHA_MIN} --alpha_max ${ALPHA_MAX}

rm -rf results/${TEST_PHASE}/${MODEL_NAME}/${ALPHA_MIN}
mv results/${TEST_PHASE}/${MODEL_NAME}/test_latest results/${TEST_PHASE}/${MODEL_NAME}/${ALPHA_MIN}
