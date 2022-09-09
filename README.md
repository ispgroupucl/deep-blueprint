# Deep-blueprint

Framework combining PyTorch Lightning and Hydra to make it easy to train a network.

You can use the code as-is, by adding your code to the different folders, but it
is recommended to use it as a library, which is explained in the usage section.


## Use as a library

### Installing

First, you can install the code using pip : 

```
pip install -e git+https://github.com/ispgroupucl/deep-blueprint.git@main#egg=deep_blueprint
```

> **WARNING** : don't install this project in your global environment. First create a
> virtual environment using conda/virtualenv/venv/poetry/pyenv

> **NOTE** : if you need to install a specific version of pytorch, install it before
> installing this project. The default pytorch version installed in this project doesn't
> work with the latest Nvidia gpu's (GeForce 3090, A100)

If you're using poetry, you can add the following line to `pypoetry.toml` :
```toml
deep_blueprint = { git = "https://github.com/ansible/ansible.git", develop = true}
```

Or, for pipenv, in the `Pipfile` file :
```toml
# torch = { version = "==1.11.0+cu113", index = "pytorch" } # If you need to use CUDA v11.3
# torchvision = { version = "==0.12.0+cu113", index = "pytorch" } # If you need to use CUDA v11.3
deep_blueprint = { editable = true, git = "https://github.com/ispgroupucl/deep-blueprint.git" }
```

### Usage

> TODO

## Use directly 
### Installation

Using poetry :

```
poetry install
```

### Running

To run a toy example of training on MNIST :

```
deep_blueprint gpus=1
```