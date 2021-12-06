## Semi-Supervised Medical Image Segmentation through Dual-Task Consistency

This is an implementation of the dual-task network used for medical image segmentation in TensorFlow. The original paper was published at [AAAI 2021](https://arxiv.org/abs/2009.04448v2?fbclid=IwAR3UK-rt7H81ePiVMHJEODAUsHomGYgslHt6RB6XBS54m8ZRg4eoE5lUygM), and the original source code (using PyTorch) can be found [here](https://github.com/HiLab-git/DTC). This code was written as part of the [CISC 867 Deep Learning](https://www.queensu.ca/academic-calendar/search/?P=CISC%20867) course at Queen's University in Kingston, following the [ML Reproducibility Challenge 2021](https://paperswithcode.com/rc2021).

## Usage
1. Clone the repo:
```
git clone https://github.com/chriscyyeung/DTC-TensorFlow.git
```
2. Install required packages:
```
pip install -r requirements.txt
```
3. Move to code directory:
```
cd code
```
4. Train the model:
```
python main.py -p train
```
5. Test the model:
```
python main.py -p test
```
