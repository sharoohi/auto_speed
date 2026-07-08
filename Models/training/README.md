# auto_speed

## auto_speed_trainer.py

Main script for training auto_speed neural network

### Example usage

```bash
  python3 auto_speed_trainer.py --dataset /dataset_root_path
```

### Main parameters:

*--dataset* : dataset directory path (required)

*--input-width* : image input width (default: 1024)

*--input-height* : image input height (default: 512)

*--batch-size* : training batch size (default: 32)

*--val-batch-size* : validation batch size (default: 4)

*--local-rank* : local rank for distributed training (default: 0)

*--version* : model version (default: "n")

*--runs-dir* : training runs directory (default: "runs")

*--epochs* : number of training epochs (default: 150)

*--world-size* : number of processes for distributed training (default: 1)

*--checkpoint-path* : pretrained checkpoint path (default: none)

*--config* : config YAML specifying model hyperparameters (default: `../config/auto_speed.yaml`)

*--output-subdir* : exact output subdirectory name within `runs_dir` (default: none; overwrites if it already exists). If omitted, a new unique subdirectory is created automatically (e.g. `run-2026-07-08-0000`)