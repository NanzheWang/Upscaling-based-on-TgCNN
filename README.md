# Upscaling-based-on-TgCNN
Project: Efficient upscaling of geologic model based on  theory-guided convolutional encoder-decoder<br>
* This is a course project for [Deep Generative Models](https://deep-generative-models.github.io/index2020.html).<br>
* This project trys to use the deep learning models to deal with the engineering problems in geological modeling.<br>
* This project trys to incorporate the physical laws into the training process of encoder-decoder and achieve unsupervised training.<br>





## Framework of the proposed method
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/framework.JPG) 
###
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/training_strategy.JPG) 
###
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/postprossesing.JPG) 

## Results
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/results1.JPG) 
###
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/results2.JPG) 
###
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/results3.JPG) 
###
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/results4.JPG) 
#### Efficiency comparison
![](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/4_figures/results5.JPG) 

## File description for codes
* [1_TgCNN_2D_hete_upscaling_tx.py](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/1_code/1_TgCNN_2D_hete_upscaling_tx.py): mapping construction with theory-guided training.<br>
* [2_up_calculate_stack.py](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/1_code/2_up_calculate_stack.py): the implementation of upscaling with the proposed method.<br>
* [3_up_results_plot.py](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/1_code/3_up_results_plot.py): results visualization.<br>
* [KLE.py](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/1_code/KLE.py): geological model generation tool.<br>
* [MyConvModel.py](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/1_code/MyConvModel.py): the structure of the convolutional network.<br>
* [fun_P5_periodic.py](https://github.com/NanzheWang/Upscaling-based-on-TgCNN/blob/main/1_code/fun_P5_periodic.py): the numerical solution tool for each patch of geological model.<br>


### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Analytical solution for upscaling hydraulic conductivity in anisotropic heterogeneous formations](https://www.sciencedirect.com/science/article/pii/S0309170818310194)
* [3] [Efficient analytical upscaling method for elliptic equations in three-dimensional heterogeneous anisotropic media](https://www.sciencedirect.com/science/article/pii/S0022169420300202)

### Author
- [Nanzhe Wang](https://github.com/NanzheWang)
