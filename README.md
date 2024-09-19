# QuEst

### Prerequisites

- **Python 3.6+**  
  Ensure you have Python version 3.6 or higher.

- **GPU Memory >= 6G**  
  A GPU with at least 6GB of memory is recommended for training and testing.

- **Numpy**  
  Install the necessary version of numpy using:
  ```
  pip install numpy
  ```

## Datasets
Market-1501: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html

## Pre-trained weight
Person re-ID: [https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/tutorial/README.md](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master)


### Train & Test
```
# Train
python train.py --path <dataset_path> --batch_size <size> --lr <learning_rate> --workers <num_workers> --epochs <num_epochs> --data <data_file> --input_dim <input_dimension>
# Test
python test.py --path <dataset_path> --data <data_file> --batch_size <size> --lr <learning_rate> --workers <num_workers> --epochs <num_epochs> --input_dim <input_dimension>

```

- `--path`  
  Specifies the path to the dataset.  
  Example: `--path Market-1501/estimator_data`

- `--data`  
  The data file to be used for testing.  
  Example: `--data QRA_metricfgsm.txt`

- `--batch_size`  
  Defines the batch size for testing.  

- `--lr`  
  Sets the initial learning rate for the test phase (using SGD).  

- `--workers`  
  Number of data loading workers (default is 0).  

- `--epochs`  
  The total number of epochs for testing.  

- `--input_dim`  
  Specifies the input dimension size for the model.  
  Example: `--input_dim 57`

#### Example of running the test:
```bash
python test.py --path ./data --data QRA_metricfgsm.txt --batch_size 128 --lr 0.01 --workers 4 --epochs 50 --input_dim 57
