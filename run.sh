python inference.py data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/eleanor/test_reduced_labels.txt

python finetune_deaf.py \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/the_book_leo/train_reduced_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/the_book_leo/val_reduced_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/the_book_leo/test_reduced_labels.txt