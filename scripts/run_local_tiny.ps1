python scripts/make_sample_data.py
python scripts/train_tokenizer.py --input_dir data/raw --out_dir tokenizer --vocab_size 4096
python scripts/prepare_dataset.py --input_dir data/raw --tokenizer tokenizer/tokenizer.json --out_dir data/processed --val_fraction 0.05
python train.py --config configs/local_80m_tiny.yaml --resume auto
