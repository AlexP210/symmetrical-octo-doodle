python ${PROJECT_ROOT}/tools/preprocess_data_mmap.py \
	--checkpoint=/path/to/pretrained_checkpoints/dino/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
	--fraction=0.7 \
	--batch_size=16 \
	--save-images \
	$1