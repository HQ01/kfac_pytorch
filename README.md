# PyTorch Distributed K-FAC UNet example

## Installation

### Setting up virtual environment
Python 3.6 is required to set up the virtualenv needed to run this example. In the root directory of this repository:
```
$ python3.6 -m venv env_unet
$ source env_unet/bin/activate
$ pip install -r requirements.txt
$ pip install torchsummary
```
Then, clone the k-fac in a separate folder and install the experimental branch code:

```
$ git clone https://github.com/gpauloski/kfac_pytorch.git
$ cd kfac_pytorch
$ git checkout experimental  # Note: use experimental branch for most up to date features
$ pip install -e .           # Note: -e installs in development mode
```


## Example
Untar `kaggle.tar` in the repository's root directory, which creates the `kaggle_3m` folder contains all 
data. Then run:
```
python -m torch.distributed.launch --nproc_per_node=4 examples/torch_brain_unet.py
```
4 being the number of GPUs on the current machine. Logs are stored by default in ./logs/torch_kfac_unet.

To run the example without kfac, pass `--kfac-update-freq=0` as an argument when running the example python script.