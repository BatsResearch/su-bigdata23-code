# Youtube datset
python main.py  --dataset_path './data/Original/' --data 'youtube' --total_lf  10 \
--batch_size  128  --optimizer_lr 2e-05 --metric  'acc' \
--random_seed 1 2 3 4 5 \
--threshold_structure 1 --threshold_removing 1 --endModel_weight_decay 0


python main.py  --dataset_path './data/Aug/' --data 'youtube' --total_lf  20 \
--batch_size  16 \
   --optimizer_lr 2e-05 --metric  'acc' \
   --random_seed 1 2 3 4 5 --threshold_structure 0 \
    --threshold_removing 1 --endModel_weight_decay 1e-4 




# SMS dataset
python main.py  --dataset_path './data/Original/' --data 'sms' --total_lf  73  --batch_size  16 \
   --optimizer_lr 2e-05 \
  --metric  'f1_binary'   --random_seed 1 2 3 4 5  --threshold_structure 0 \
   --threshold_removing 0.1 --endModel_weight_decay 1e-4 


python main.py  --dataset_path './data/Aug/' --data 'sms' --total_lf  146 --batch_size  16 \
   --optimizer_lr 3e-05 --metric  'f1_binary' \
   --random_seed 1 2 3 4 5  --threshold_structure 0 --threshold_removing 10 --endModel_weight_decay 1e-4


# Spouse Dataset
python main.py --dataset_path './data/Original/' --data 'spouse' --total_lf  11 \
--batch_size  32   --optimizer_lr 2e-05 --metric  'f1_binary' \
--random_seed 1 2 3 4 5 \
--threshold_structure 3 --threshold_removing 1 --endModel_weight_decay 1e-4 





python main.py  --dataset_path './data/Aug/' --data 'spouse' --total_lf  22 --batch_size  32 \
   --optimizer_lr 3e-05 --metric  'f1_binary'  --random_seed 1 2 3 4 5 \
  --threshold_structure 0.08 --threshold_removing 0 \
 --endModel_weight_decay 1e-4 