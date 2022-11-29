config = {'split_ratio': 1, 'epochs': 50, 'batch_size': 16, 'task': 'classification', 'split_unlabel': False, 'change_output': False,
 'attention_module': 'MTSC',
 'lr': 0.001, 'l2_reg': 0, 'optimizer': 'RAdam', 'console': False, 'print_interval': 1, 'global_reg': False,
 'd_model': 16, 'dim_feedforward': 64, 'num_heads': 8, 'num_layers': 1, 'dropout': 0.1,
 'pos_encoding': 'pv_1', 'activation': 'gelu', 'freeze': False, 'normalization_layer': 'BatchNorm', 'key_metric': 'accuracy','val_interval': 2,
'save_dir': 'src/experiments/exp_1/checkpoints', 'pred_dir': 'src/experiments/exp_1/predictions',
'load_dir': 'src/experiments/exp_1/checkpoints/model_last.pth'}
