# Hyperspectral Image Classification Based on Expansion Convolution Network

[Cuiping Shi](scp1980@126.com), [Diling Liao](liaodiling2020@163.com), [Tianyu Zhang]( 2019910178@qqhru.edu.cn), [Liguo Wang]( wangliguo@hrbeu.edu.cn)
___________

The code in this toolbox implements the ["Hyperspectral Image Classification Based on Expansion Convolution Network"]. More specifically, it is detailed as follow.

![alt text](./FECNet.PNG)

In order to illustrate the difference between 2-D expansion convolution and standard convolution, taking convolution kernel with size of 3 × 3 as an example, the process of the 2-D expansion convolution is shown in Fig. 2 (where p × p represents the size of space). Similarly, in order to
illustrate the working process of 3-D expansion convolution, take the convolution kernel with the size of 3 × 3 × 3 as an example. The relationship between 3-D expansion convolution and 3-D standard convolution still follows the law of 2-D convolution.

![alt text](./BConv.PNG)

Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

C. Shi, D. Liao, T. Zhang and L. Wang, "Hyperspectral Image Classification Based on Expansion Convolution Network," in IEEE Trans. on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022.
    
System-specific notes
---------------------
The codes of networks were tested using PyTorch 1.10.0 version (CUDA 11.3) in Python 3.8 on Ubuntu system.

How to use it?
---------------------
This toolbox consists of two proposed branchs, i.e., Strategy for Extracting Spectral Features Based on MSSP and Strategy for Extracting Spatial Features, that can be plug-and-played into both pixel-wise and patch-wise hyperspectral image classification. For more details, please refer to the paper.

Here an example experiment is given by using **Indian Pines hyperspectral data**. Directly run **main.py** functions with different network parameter settings and input **dataset** to produce the results. Please note that due to the randomness of the parameter initialization, the experimental results might have slightly different from those reported in the paper.

If you want to run the code in your own data, you can accordingly change the input (e.g., data, labels) and tune the parameters.

If you encounter the bugs while using this code, please do not hesitate to contact us.

If emergency, you can also add my email: liaodiling2020@163.com or QQ: 3097264896.
