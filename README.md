# DRFUZZ

## Description

`DRFuzz` is a novel regression fuzzing framework for deep learning systems. It is designed to generate high-fidelity test inputs that trigger diverse regression faults effectively. To improve the fault-triggering capability of test inputs, `DRFuzz` adopts a Markov Chain Monte Carlo (MCMC) strategy to select mutation rules that are prone to trigger regression faults. Furthermore, to enhance the diversity of generated test inputs, we propose a diversity criterion to guide triggering more faulty behaviors. In addition, `DRFuzz` incorporates a GAN-based fidelity assurance method to guarantee the fidelity of test inputs. We conducted an empirical study to evaluate the effectiveness of `DRFuzz` on four regression scenarios (i.e., supplementary training, adversarial training, model fixing, and model pruning). The experimental results demonstrate the effectiveness of `DRFuzz`.



## The Structure

Here, we briefly introduce the usage/function of each directory: 

- `coverages`: baseline coverages and the input evaluation to calculate the diversity in `DRFuzz`
- `dcgan`: the GAN-based Fidelity Assurance Technique (the DCGAN structure and prediction code)
- `models`: the original models and its regression model. (Since the file size of some models are large, here we provide all the models and regression model on MNIST-LeNet5 for the reproduction)
- `params`: some params of `DRFuzz` and each model/datasets
- `src`: the main algorithm of `DRFuzz` & The experimental script
- `kmnc_profile`: profile for baseline approach DeepHunter, which saves the boundary value of each neuron.（Please note that this file is only for DeepHunter-KMNC following the implementation of its source code to improve the efficiency of KMNC, and it is not in the scope of DRFuzz.）

## Datasets/Models

We use 4 popular DL models based on 4 datasets under five regression scenarios, as the initial seed models in `DRFuzz`, which have been widely used in many existing studies.

| ID   | model    | dataset | M1_acc | M2_acc | Scenario |
| ---- | -------- | ------- | ------ | ------ | -------- |
| 1    | LeNet5   | MNIST   | 85.87% | 97.83% | SUPPLY   |
| 2    | LeNet5   | MNIST   | 98.07% | 97.50% | ADV:BIM  |
| 3    | LeNet5   | MNIST   | 98.07% | 98.30% | ADV:CW   |
| 4    | LeNet5   | MNIST   | 98.07% | 98.12% | FIXING   |
| 5    | LeNet5   | MNIST   | 98.07% | 98.12% | PRUNE    |
| 6    | VGG16    | CIFAR10 | 87.67% | 87.88% | SUPPLY   |
| 7    | VGG16    | CIFAR10 | 87.92% | 87.51% | ADV:BIM  |
| 8    | VGG16    | CIFAR10 | 87.92% | 88.00% | ADV:CW   |
| 9    | VGG16    | CIFAR10 | 87.92% | 88.40% | FIXING   |
| 10   | VGG16    | CIFAR10 | 87.92% | 76.27% | PRUNE    |
| 11   | AlexNet  | FM      | 89.33% | 90.34% | SUPPLY   |
| 12   | AlexNet  | FM      | 91.70% | 90.96% | ADV:BIM  |
| 13   | AlexNet  | FM      | 91.70% | 91.87% | ADV:CW   |
| 14   | AlexNet  | FM      | 91.70% | 92.90% | FIXING   |
| 15   | AlexNet  | FM      | 91.70% | 91.54% | PRUNE    |
| 16   | ResNet18 | SVHN    | 88.85% | 91.93% | SUPPLY   |
| 17   | ResNet18 | SVHN    | 92.05% | 91.90% | ADV:BIM  |
| 18   | ResNet18 | SVHN    | 92.05% | 92.01% | ADV:CW   |
| 19   | ResNet18 | SVHN    | 92.05% | 92.10% | FIXING   |
| 20   | ResNet18 | SVHN    | 92.05% | 91.00% | PRUNE    |


1: We design 4 regression scenarios: supplementary training (denoted as SUPPLY), adversarial training (denoted as ADV), white-box model fixing (denoted as FIXING), and model pruning(denoted as PRUNE).

2: For SUPPY, we select 80% of the training set to train the original model and use the 20% remaining data for supplementary training.

3: For ADV, we provide adversarial training on 2 kinds of adversarial examples. (C&W and BIM)
 
4: The overall result is saved here, named `overall_result.jpg`, due to the limited paper space.

5: Please note that since the code logic of our baseline method (DiffChaser) is so different from our technique and DeepHunter, we directly use the source code released by the authors for the experiment rather than combining them in our framework.

6: If you want to download other models, please check the link below:
- MNIST: https://drive.google.com/file/d/1l_s--uSm5TN5a0xUTb1WobRHLti6o7Zw/view?usp=sharing
- Cifar10:https://drive.google.com/file/d/14ZBdd_AlDfVYcbdV31O0MNFHh0-z87cL/view?usp=sharing
- FM:https://drive.google.com/file/d/1C-gl_HgOOirM4I1mhDvShKFGLPzFlRbn/view?usp=sharing
- SVHN:https://drive.google.com/file/d/1UFZv6WZ0b-W0Qk4o8xfvAhK-mPiJhQGk/view?usp=sharing
After downloading the models, please put these models in `models` and make sure that the address of models are correct in **src/experiment_builder.py**. 

## The Requirements:

- python==3.7  (In fact 3.7.5 and 3.7.16 fits our work)

- keras==2.3.1 

- tensorflow==1.15.0 

- cleverhans==3.0.1  

- h5py==2.10.0

Please note that if you encounter the following error 'TypeError: Descriptors can not be created directly.', you may need to downgrade the protobuf package to 3.20.x or lower. It is so rarely happened. You can use 'pip install protobuf==3.20.1' to avoid this circumstance. Still, if you are confused with the environment setting, we also provided a file named `requirement.txt` to facilitate the installation process. You can directly use the script below. You can choose to use _pip_ or _conda_ in the script. 

~~~
pip install -r requirement.txt
~~~

Please note that if your are alerted that you are missing corresponding `.so` file in your environment, it indicates that you should install the corresponding library `libglib2.0-dev`. You may use the script below.

~~~
apt-get install libglib2.0-dev
~~~

We strongly suggest you run this project on **Linux**. We implemented the entire project on **Linux version 4.18.0-15-generic**. We will provide the configuration on windows in this project description in the future. We also provide you with the docker image of DRFuzz on https://drive.google.com/file/d/1rIpzyY_jWFPp-ZuvKl_lZNH1y4HvRzHz/view?usp=sharing. The code has been downloaded in the directory `/home/share/DRFuzz-main`, you can use the conda environment `drfuzz` by the script below to using the artifact.

~~~
source activate drfuzz
~~~

## Reproducibility

### Environment

We conducted 20 experiments in `DRFuzz`. The basic running steps are presented as follows:

**Step 0:** Please install the above runtime environment.

**Step 1:** Clone this repository. Download the dataset and models from our Google-Drive. 
Save the code and unzip datasets and models to `/your/local/path/`, e.g., `/your/local/path/models` and `/your/local/path/dataset`. 
(`/your/local/path/` should be the absolute path on your server, e.g., `/home/user_xxx/`) 

**Step 2:** Train yourself DCGAN models and save them to `/your/local/path/dcgan`. (Or you can use the one provided by us for reproductivity.)

**Step 3:** Edit configuration files `/your/local/path/src/experiment_builder.py` and `/your/local/path/dcgan/DCGAN_utils.py` in order to set the dataset, model, and DCGAN model path into `DRFuzz`

### Running DRFuzz

The `DRFuzz` artifacts are well organized, and researchers can simply run `DRFuzz` with the following command.

~~~
python main.py --params_set mnist LeNet5 change drfuzz --dataset MNIST --model LeNet5 --coverage change --time 1440
~~~

main.py contains all the configurations for our experimentation.

`--params_set` refers to the configuration and parameters in each dataset/model. Please select according to your requirements based on files in `params` directory. 
            If you want to adopt your own parameters, please go to `your/local/path/params` to change the setting params.

`--dataset` refers to the dataset you acquired. There are totally 4 choices ["MNIST", "CIFAR10", "FM", "SVHN"]

`--models` refers to the model you acquired. We totally designed 20 models according to 'Datasets/Models'. The settings are presented in the main.py. 
            Please select according to Datasets/Models and your experimental settings.

`--coverage` refers to the coverage used guide the fuzzing process; please set it to 'change' if you want to use `DRFuzz`; other choices are for compared approaches such as DeepHunter.

`--time` refers to the time of running our experiments. We set it to 1440 minutes (24 hours) for our experiment. For quick installation, you can set it to 5 minutes as a try.

## Reusability

For users who want to evaluate if `DRFuzz` is effective on their own regression models, we also prepared an approach to reuse `DRFuzz` on new regression models and new datasets.

Firstly, the users need to update the addresses of their own datasets and regression models under test in function _\_get\_dataset_ and function _\_get\_models_ in the **src/experiment\_builder.py** file for `DRFuzz` to load. Please note that if the dataset requires further preprocessing, the users should also update the corresponding preprocessing method _picture\_preprocess_ in the **src/utility.py** file. 

Secondly, the users are required to train a simple Discriminator of GAN (e.g., in `dcgan`) to guarantee the fidelity of generated test inputs. From that, `DRFuzz` can be adapted to new regression models under test; it will conduct the following fuzzing process and finish the entire job. 

Please do not forget to name your own regression scenarios and regression datasets, setting the corresponding parameters (or configurations) in `params` so that you can load the parameters through experimental scripts.

## Extra Files

For users who want to fetch the kmnc_profile file, please see the link below:

Link: https://pan.baidu.com/s/1apWPrYnvrEutz7VA8gZZNg?pwd=z3an 

Pin-code: z3an


## Additional Results
### RQ1 
The overall result is saved here named `overall_result.jpg` due to the limited paper space.

You can also refer to the table below for detailed information.

| ID   | Dataset-Model | Regression | Approach   | #RF     | #URF   | #Seed  | #GF     |
| ---- | ------------- | ---------- | ---------- | ------- | ------ | ------ | ------- |
|      |               |            | DiffChaser | 27,065  | 1,185  | 1,138  | 27,400  |
| 1    | MNIST-LeNet5  | SUPPLY     | DeepHunter | 4,837   | 2,701  | 2,127  | 49,003  |
|      |               |            | DRFuzz     | 36,293  | 8,767  | 4,345  | 68,451  |
|      |               |            | DiffChaser | 14,994  | 303    | 277    | 15,602  |
| 2    | MNIST-LeNet5  | ADV:BIM    | DeepHunter | 19,588  | 8,670  | 4,760  | 61,871  |
|      |               |            | DRFuzz     | 41,376  | 13,342 | 5,948  | 82,377  |
|      |               |            | DiffChaser | 531     | 81     | 73     | 1,601   |
| 3    | MNIST-LeNet5  | ADV:CW     | DeepHunter | 3,799   | 2,408  | 1,791  | 53,178  |
|      |               |            | DRFuzz     | 25,727  | 7,996  | 3,972  | 117,329 |
|      |               |            | DiffChaser | 4,799   | 582    | 513    | 6,927   |
| 4    | MNIST-LeNet5  | FIXING     | DeepHunter | 2,760   | 2,279  | 1,906  | 35,139  |
|      |               |            | DRFuzz     | 32,858  | 11,075 | 4,990  | 116,021 |
|      |               |            | DiffChaser | 19,047  | 2,832  | 2,426  | 33,838  |
| 5    | MNIST-LeNet5  | PRUNE      | DeepHunter | 6,763   | 3,689  | 2,547  | 68,281  |
|      |               |            | DRFuzz     | 26,470  | 9,071  | 4,571  | 100,967 |
|      |               |            | DiffChaser | 8,356   | 2,204  | 1,739  | 23,790  |
| 6    | CIFAR10-VGG16 | SUPPLY     | DeepHunter | 983     | 630    | 516    | 8,089   |
|      |               |            | DRFuzz     | 41,422  | 16,105 | 6,505  | 331,701 |
|      |               |            | DiffChaser | 8,976   | 1,562  | 1,259  | 17,888  |
| 7    | CIFAR10-VGG16 | ADV:BIM    | DeepHunter | 857     | 664    | 519    | 7,272   |
|      |               |            | DRFuzz     | 58,192  | 17,222 | 6,925  | 321,619 |
|      |               |            | DiffChaser | 3,230   | 780    | 628    | 13,489  |
| 8    | CIFAR10-VGG16 | ADV:CW     | DeepHunter | 247     | 185    | 163    | 4,228   |
|      |               |            | DRFuzz     | 33,412  | 11,471 | 5,004  | 374,531 |
|      |               |            | DiffChaser | 18,220  | 2,614  | 1,877  | 27,192  |
| 9    | CIFAR10-VGG16 | FIXING     | DeepHunter | 2,126   | 1,202  | 854    | 7,360   |
|      |               |            | DRFuzz     | 74,644  | 22,576 | 7,338  | 249,626 |
|      |               |            | DiffChaser | 152,702 | 6,989  | 4,037  | 169,086 |
| 10   | CIFAR10-VGG16 | PRUNE      | DeepHunter | 6,885   | 2,051  | 1,212  | 11,394  |
|      |               |            | DRFuzz     | 115,099 | 22,333 | 7,883  | 228,750 |
|      |               |            | DiffChaser | 13,315  | 325    | 285    | 21,326  |
| 11   | FM-AlexNet    | SUPPLY     | DeepHunter | 6,248   | 2,872  | 2,088  | 39,535  |
|      |               |            | DRFuzz     | 63,981  | 12,711 | 5,743  | 260,899 |
|      |               |            | DiffChaser | 26,157  | 750    | 557    | 45,619  |
| 12   | FM-AlexNet    | ADV:BIM    | DeepHunter | 6,130   | 2,886  | 1,875  | 26,690  |
|      |               |            | DRFuzz     | 114,729 | 17,886 | 6,806  | 382,470 |
|      |               |            | DiffChaser | 4,995   | 491    | 407    | 26,169  |
| 13   | FM-AlexNet    | ADV:CW     | DeepHunter | 2,178   | 1,473  | 1,191  | 33,401  |
|      |               |            | DRFuzz     | 50,801  | 13,803 | 5,759  | 389,195 |
|      |               |            | DiffChaser | 32,580  | 1,292  | 861    | 44,825  |
| 14   | FM-AlexNet    | FIXING     | DeepHunter | 9,574   | 5,225  | 3,044  | 25,493  |
|      |               |            | DRFuzz     | 176,104 | 27,352 | 7,999  | 377,966 |
|      |               |            | DiffChaser | 52,384  | 1,483  | 1,063  | 66,501  |
| 15   | FM-AlexNet    | PRUNE      | DeepHunter | 17,302  | 8,150  | 3,963  | 35,696  |
|      |               |            | DRFuzz     | 168,029 | 26,229 | 7,885  | 306,389 |
|      |               |            | DiffChaser | 1,220   | 250    | 221    | 1,599   |
| 16   | SVHN-ResNet18 | SUPPLY     | DeepHunter | 1,731   | 1,126  | 878    | 10,790  |
|      |               |            | DRFuzz     | 31,364  | 15,980 | 8,493  | 170,618 |
|      |               |            | DiffChaser | 1,088   | 83     | 71     | 1,487   |
| 17   | SVHN-ResNet18 | ADV:BIM    | DeepHunter | 1,492   | 1,057  | 864    | 9,908   |
|      |               |            | DRFuzz     | 29,779  | 18,131 | 9,558  | 169,835 |
|      |               |            | DiffChaser | 370     | 64     | 60     | 1,074   |
| 18   | SVHN-ResNet18 | ADV:CW     | DeepHunter | 264     | 225    | 213    | 5,773   |
|      |               |            | DRFuzz     | 10,943  | 8,509  | 5,610  | 178,922 |
|      |               |            | DiffChaser | 663     | 199    | 184    | 1,198   |
| 19   | SVHN-ResNet18 | FIXING     | DeepHunter | 941     | 742    | 627    | 8,815   |
|      |               |            | DRFuzz     | 22,612  | 16,431 | 8,741  | 168,544 |
|      |               |            | DiffChaser | 712     | 626    | 532    | 1,200   |
| 20   | SVHN-ResNet18 | PRUNE      | DeepHunter | 1,888   | 1,118  | 887    | 5,429   |
|      |               |            | DRFuzz     | 34,561  | 18,266 | 10,420 | 105,750 |


### RQ3 and its accuracy against test set


| model    | scenario | train on\test on | DRFuzz | DeepHunter   | DiffChaser | Acc Change |
| -------- | -------- | ---------------- | ------ | ------------ | ----------- | ---------- |
| LeNet5   | SUPPLY   | DRFuzz           | 84.58% | 59.42%       | 83.36%      | 0.78%      |
| LeNet5   | SUPPLY   | DeepHunter       | 58.99% | 80.71%       | 71.29%      | 0.90%      |
| LeNet5   | SUPPLY   | DiffChaser       | 53.42% | 34.27%       | 72.92%      | 0.96%      |
| AlexNet  | SUPPLY   | DRFuzz           | 84.68% | 67.56%       | 77.56%      | 0.79%      |
| AlexNet  | SUPPLY   | DeepHunter       | 51.85% | 73.56%       | 42.57%      | 1.18%      |
| AlexNet  | SUPPLY   | DiffChaser       | 41.25% | 37.18%       | 62.28%      | -1.95%     |
| VGG16    | SUPPLY   | DRFuzz           | 90.70% | 90.54%       | 88.84%      | -0.33%     |
| VGG16    | SUPPLY   | DeepHunter       | 68.65% | 71.20%       | 65.75%      | -0.12%     |
| VGG16    | SUPPLY   | DiffChaser       | 72.27% | 75.77%       | 71.24%      | -0.22%     |
| ResNet18 | SUPPLY   | DRFuzz           | 79.97% | 78.84%       | 43.25%      | 0.12%      |
| ResNet18 | SUPPLY   | DeepHunter       | 61.01% | 65.85%       | 68.25%      | -2.22%     |
| ResNet18 | SUPPLY   | DiffChaser       | 46.44% | 51.25%       | 62.01%      | -2.69%     |
| LeNet5   | ADV:BIM  | DRFuzz           | 87.69% | 73.38%       | 75.94%      | 0.84%      |
| LeNet5   | ADV:BIM  | DeepHunter       | 45.47% | 79.42%       | 58.76%      | 0.56%      |
| LeNet5   | ADV:BIM  | DiffChaser       | 49.58% | 57.97%       | 61.75%      | 0.42%      |
| AlexNet  | ADV:BIM  | DRFuzz           | 85.68% | 74.40%       | 76.94%      | 1.30%      |
| AlexNet  | ADV:BIM  | DeepHunter       | 67.44% | 76.04%       | 70.93%      | 1.08%      |
| AlexNet  | ADV:BIM  | DiffChaser       | 49.29% | 51.99%       | 72.66%      | 0.09%      |
| VGG16    | ADV:BIM  | DRFuzz           | 88.76% | 85.09%       | 78.09%      | 0.34%      |
| VGG16    | ADV:BIM  | DeepHunter       | 73.74% | 71.53%       | 76.35%      | 0.65%      |
| VGG16    | ADV:BIM  | DiffChaser       | 71.11% | 62.68%       | 76.64%      | 0.61%      |
| ResNet18 | ADV:BIM  | DRFuzz           | 85.97% | 86.63%       | 91.74%      | 0.75%      |
| ResNet18 | ADV:BIM  | DeepHunter       | 69.83% | 73.99%       | 81.31%      | 0.35%      |
| ResNet18 | ADV:BIM  | DiffChaser       | 65.38% | 68.90%       | 80.79%      | 0.43%      |
| LeNet5   | ADV:CW   | DRFuzz           | 75.76% | 35.00%       | 65.87%      | 0.00%      |
| LeNet5   | ADV:CW   | DeepHunter       | 42.12% | 69.01%       | 26.59%      | -0.08%     |
| LeNet5   | ADV:CW   | DiffChaser       | 52.55% | 30.65%       | 78.77%      | -0.63%     |
| AlexNet  | ADV:CW   | DRFuzz           | 86.72% | 69.36%       | 63.21%      | 0.04%      |
| AlexNet  | ADV:CW   | DeepHunter       | 53.82% | 62.61%       | 59.50%      | -0.15%     |
| AlexNet  | ADV:CW   | DiffChaser       | 46.21% | 42.98%       | 65.77%      | 0.42%      |
| VGG16    | ADV:CW   | DRFuzz           | 88.98% | 82.49%       | 85.92%      | -0.63%     |
| VGG16    | ADV:CW   | DeepHunter       | 77.69% | 73.74%       | 74.59%      | -0.01%     |
| VGG16    | ADV:CW   | DiffChaser       | 66.04% | 64.65%       | 69.49%      | 0.02%      |
| ResNet18 | ADV:CW   | DRFuzz           | 81.44% | 83.71%       | 92.49%      | 0.34%      |
| ResNet18 | ADV:CW   | DeepHunter       | 60.36% | 68.37%       | 95.83%      | 0.41%      |
| ResNet18 | ADV:CW   | DiffChaser       | 58.00% | 63.26%       | 95.83%      | -0.80%     |
| LeNet5   | FIXING  | DRFuzz           | 85.41% | 66.48%       | 65.64%      | 0.01%      |
| LeNet5   | FIXING  | DeepHunter       | 54.25% | 70.46%       | 62.47%      | -0.01%     |
| LeNet5   | FIXING  | DiffChaser       | 50.83% | 49.23%       | 64.78%      | -0.12%     |
| AlexNet  | FIXING  | DRFuzz           | 76.08% | 56.88%       | 39.46%      | -0.20%     |
| AlexNet  | FIXING  | DeepHunter       | 41.96% | 62.41%       | 39.28%      | -1.14%     |
| AlexNet  | FIXING  | DiffChaser       | 37.86% | 35.28%       | 60.27%      | -4.65%     |
| VGG16    | FIXING  | DRFuzz           | 71.98% | 66.42%       | 45.52%      | -1.21%     |
| VGG16    | FIXING  | DeepHunter       | 54.42% | 56.56%       | 58.67%      | -2.64%     |
| VGG16    | FIXING  | DiffChaser       | 55.56% | 59.13%       | 44.45%      | -4.38%     |
| ResNet18 | FIXING  | DRFuzz           | 71.56% | 67.55%       | 58.11%      | 0.40%      |
| ResNet18 | FIXING  | DeepHunter       | 64.19% | 68.09%       | 74.32%      | 0.00%      |
| ResNet18 | FIXING  | DiffChaser       | 50.98% | 61.73%       | 73.27%      | -1.78%     |
| LeNet5   | PRUNE    | DRFuzz           | 75.21% | 39.57%       | 70.27%      | 0.09%      |
| LeNet5   | PRUNE    | DeepHunter       | 38.77% | 72.11%       | 44.41%      | -0.06%     |
| LeNet5   | PRUNE    | DiffChaser       | 34.54% | 30.69%       | 76.54%      | -0.03%     |
| AlexNet  | PRUNE    | DRFuzz           | 83.85% | 71.00%       | 63.38%      | 0.78%      |
| AlexNet  | PRUNE    | DeepHunter       | 53.46% | 71.88%       | 54.12%      | 0.68%      |
| AlexNet  | PRUNE    | DiffChaser       | 35.12% | 33.04%       | 64.31%      | -1.51%     |
| VGG16    | PRUNE    | DRFuzz           | 86.53% | 87.51%       | 80.22%      | 11.75%     |
| VGG16    | PRUNE    | DeepHunter       | 74.90% | 83.67%       | 79.62%      | 12.00%     |
| VGG16    | PRUNE    | DiffChaser       | 72.49% | 78.22%       | 79.65%      | 12.13%     |
| ResNet18 | PRUNE    | DRFuzz           | 80.54% | 83.41%       | 83.53%      | 3.54%      |
| ResNet18 | PRUNE    | DeepHunter       | 71.82% | 76.74%       | 77.22%      | 3.17%      |
| ResNet18 | PRUNE    | DiffChaser       | 71.68% | 80.26%       | 81.94%      | 4.06%      |

