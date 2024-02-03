CUDA_VISIBLE_DEVICES=3 python -m geometryhints.test \
--config_file configs/models/hero_model.yaml \
--load_weights_from_checkpoint /mnt/nas3/personal/mohameds/hero_model_sr_int_flip_removed_opts.ckpt \
--data_config configs/data/scannet_default_test.yaml  \
--num_workers 10 --batch_size 8 \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet --run_fusion; 

CUDA_VISIBLE_DEVICES=1 python -m geometryhints.test \
--config_file configs/models/hero_model.yaml \
--load_weights_from_checkpoint /mnt/nas3/personal/mohameds/hero_model_sr_int_flip_removed_opts.ckpt \
--data_config configs/data/scannet_default_train_test.yaml  \
--num_workers 8 --cache_depths --batch_size 4 \
--run_fusion --output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet; 

CUDA_VISIBLE_DEVICES=1 python -m geometryhints.test \
--config_file configs/models/hero_model.yaml \
--load_weights_from_checkpoint /mnt/nas3/personal/mohameds/hero_model_sr_int_flip_removed_opts.ckpt \
--data_config configs/data/scannet_default_val_test.yaml  \
--num_workers 8 --cache_depths --batch_size 4 \
--run_fusion --output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet;