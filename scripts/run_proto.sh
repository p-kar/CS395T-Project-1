source /scratch/cluster/pkar/pytorch-gpu-py3/bin/activate
code_root=__CODE_ROOT__

python -u $code_root/driver.py \
	--mode __MODE__ \
	--data_dir __DATA_DIR__ \
	--nworkers __NWORKERS__ \
	--bsize __BSIZE__ \
	--shuffle __SHUFFLE__ \
	--nclasses __NCLASSES__ \
	--resize __RESIZE__ \
	--img_size __IMG_SIZE__ \
	--arch __ARCH__ \
	--target_type __TARGET_TYPE__ \
	--pretrained __PRETRAINED__ \
	--optim __OPTIM__ \
	--lr __LR__ \
	--wd __WD__ \
	--momentum __MOMENTUM__ \
	--epochs __EPOCHS__ \
	--max_norm __MAX_NORM__ \
	--lr_decay_step __LR_DECAY_STEP__ \
	--lr_decay_gamma __LR_DECAY_GAMMA__ \
	--start_epoch __START_EPOCH__ \
	--save_path __SAVE_PATH__ \
	--log_dir __LOG_DIR__ \
	--log_iter __LOG_ITER__ \
	--resume __RESUME__ \
	--seed __SEED__
