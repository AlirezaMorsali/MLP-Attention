# MLP-Attention
This is the PyTorch implementation of our paper __MLP-Attention: Improving Transformer Architecture with MLP Attention Weights__, submitted to ICLR Tiny paper 2023.

<div align=center>
<img width=95% src="https://github.com/AlirezaMorsali/MLP-Attention/blob/main/Architecture.png"/>
</div>
The Transformer architecture has revolutionized natural language processing (NLP) and has achieved state-of-the-art results in various tasks. The attention mechanism is one of the key components of the Transformer architecture, which allows the model to focus on relevant parts of the input. In the standard Transformer, the attention weights are computed by the dot product of query and key vectors followed by a softmax function. However, in this paper, we propose to replace the dot product of query and key vectors with a multi-layer perceptron (MLP) to compute attention weights directly from the embeddings.  The proposed modification is simple and can be easily implemented in existing Transformer-based models to improve their performance as shown in this paper for an NLP task.

## Results 
<div align=center>
<img width=95% src="https://github.com/AlirezaMorsali/MLP-Attention/blob/main/Loss.png"/>
</div>

# Run

## 1. Clone Repository
```bash
$ git clone https://github.com/AlirezaMorsali/MLP-Attention.git
$ cd MLP-Attention/
```
## 2. Requirements
```bash
$ pip install -r requirements.txt
```

## 3. Run experiments:
Use the following codes to run the experiments.

```bash
python main.py
```
The results will be saved in the `Results` directory.

## 4. Visualize Results
Use the following codes to plot the results.

```bash
python plot_results.py
```
