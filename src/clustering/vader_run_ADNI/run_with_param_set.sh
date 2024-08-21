#!/bin/bash

SEED=200
python3 run_vader_val.py --input_data_file=../../data/ADNI_tables --data_reader_script=../../data/ADNI_datareader.py --n_epoch=50 --n_consensus=40 --k=2 --n_hidden 32 1 --learning_rate=0.0001 --batch_size=16 --alpha=1 --output_path=results --val_data_file=../../data/NACC_tables --val_savepath=NACC_consensus_matrix.csv --val_data_reader_script=../../data/NACC_datareader.py --seed $SEED
