# ssd_scratch vsr model
python test.py data.modality=video data/dataset=lrs3 experiment_name=vsr_prelrs3vox2_large_ftlrs3_test model/visual_backbone=resnet_transformer_large model.pretrained_model_path=/ssd_scratch/cvit/akshat/vsr_prelrs3vox2_large_ftlrs3.pth

# home vsr model
python test.py data.modality=video data/dataset=lrs3 experiment_name=vsr_prelrs3vox2_large_ftlrs3_test model/visual_backbone=resnet_transformer_large model.pretrained_model_path=/home2/akshat/vsr_prelrs3vox2_large_ftlrs3.pth
