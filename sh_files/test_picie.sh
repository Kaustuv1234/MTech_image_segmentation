# EVAL_PATH='../../picie.pkl' # Your checkpoint directory. 

K_train=6
K_test=6
EVAL_PATH='results/picie/train/1/augmented/res1=320_res2=640/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale=0.5/K_train=6_cosine/checkpoint.pth.tar'
mkdir -p results/picie/test/

python train_picie.py \
--data_root content/drive/MyDrive/datasets/coco/ \
--eval_only \
--save_root results/picie/test/ \
--K_train ${K_train} --K_test ${K_test} \
--eval_path ${EVAL_PATH} \
--res 320 
