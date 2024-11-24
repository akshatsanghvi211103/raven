python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/accented_speakers \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/supersymo/val_reduced_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune_deaf.py \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/eleanor/train_reduced_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/eleanor/val_reduced_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/eleanor/test_reduced_labels.txt \