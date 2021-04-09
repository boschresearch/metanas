# Meta-Learning of Neural Architectures for Few-Shot Learning


This is the implmentation for [Meta-Learning of Neural Architectures for Few-Shot Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Elsken_Meta-Learning_of_Neural_Architectures_for_Few-Shot_Learning_CVPR_2020_paper.html).



## Requirements and Setup

### Install requiered packages.
Run

```bash
conda env create -f environment.yml
```
to create a new conda environment named `metanas` with all requiered packages and activate it.

### Download the data

Download the data sets you want to use (Omniglot or miniImagenet). You can also set `download=True` for the data loaders in [`torchmeta_loader.py`](metanas/tasks/torchmeta_loader.py) to use the data download provided by [Torchmeta](https://github.com/tristandeleu/pytorch-meta). 



## How to Use

Please refer to the [`scripts`](scripts/) folder for examples how to use this code. E.g., for experiments on miniImagenet:

- Running meta training for MetaNAS: [`run_in_meta_train.sh`](scripts/run_in_meta_train.sh)
- Running meta testing for a checkpoint from the above meta training experiment: [`run_in_meta_testing.sh`](scripts/run_in_meta_testing.sh)
- Scaling up an optimized architecture from above meta training experiment and retraining it: [`run_in_upscaled.sh`](scripts/run_in_upscaled.sh)


## Purpose of this Project

This software is a research prototype, solely developed for the publication cited above. It will neither be maintained nor monitored in any way.

## License

'Meta-Learning of Neural Architectures for Few-Shot Learning' is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE.txt) file for details.

For a list of other open source components included in 'Meta-Learning of Neural Architectures for Few-Shot Learning', see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

