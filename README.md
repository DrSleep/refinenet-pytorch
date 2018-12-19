# RefineNet (in PyTorch)

This repository provides the ResNet-101-based model trained on PASCAL VOC from the paper `RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation` (the provided weights achieve **80.5**% mean IoU on the validation set in the single scale setting)

```
RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
In CVPR 2017
```

## Getting Started

For flawless reproduction of our results, the Ubuntu OS is recommended. The model have been tested using Python 3.6.

### Dependencies

```
pip3
torch>=0.4.0
```
To install required Python packages, please run `pip3 install -r requirements3.txt` (Python3) - use the flag `-u` for local installation.
The given examples can be run with, or without GPU.

## Running examples

For the ease of reproduction, we have embedded all our examples inside Jupyter notebooks. One can either download them from this repository and proceed working with them on his/her local machine/server, or can resort to online version supported by the Google Colab service.

### Jupyter Notebooks [Local]

If all the installation steps have been smoothly executed, you can proceed with running any of the notebooks provided in the `examples/notebooks` folder.
To start the Jupyter Notebook server, on your local machine run `jupyter notebook`. This will open a web page inside your browser. If it did not open automatically, find the port number from the command's output and paste it into your browser manually.
After that, navigate to the repository folder and choose any of the examples given. 

Inside the notebook, one can try out their own images, write loops to iterate over videos / whole datasets / streams (e.g., from webcam). Feel free to contribute your cool use cases of the notebooks!

### Colab Notebooks [Web]

*Coming soon*

## Training scripts

Please refer to the training scripts for [Light-Weight-RefineNet](https://github.com/DrSleep/light-weight-refinenet)


## More projects to check out

[Light-Weight-RefineNet](https://github.com/DrSleep/light-weight-refinenet) - compact version of RefineNet running in real-time with minimal decrease in accuracy (3x decrease in the number of parameters, 5x decrease in the number of FLOPs)

## License

For academic usage, this project is licensed under the 2-clause BSD License - see the [LICENSE](LICENSE) file for details. For commercial usage, please contact the authors.
