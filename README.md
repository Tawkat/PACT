# PACT
Pretraining with Adversarial CL for Text Classification

![plot](./Images/PACT.png)

## 1. Requirements:
```diff
python version: 3.8
pip3 install -r requirements.txt
```

## 2. Prepare Pretraining Data:

### Collect Wikipedia Dataset
```diff
pip install datasets 
```
```diff
cd pretraining_data
chmod +x ./download_raw_data.sh
./download_raw_data.sh 
```

### Tokenize Dataset

```diff
chmod +x ./tokenize_bert_uncased_data.sh
./tokenize_bert_uncased_data.sh
```

## Pretraining PACT

```diff
cd pretraining
chmod +x ./train_pact.sh
./train_pact.sh
```

## Finetuning

All the experiments on English benchmarks are conducted using huggingface transformers library. You can find the instruction to finetune our model [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)

### Hyperparameter values for Validation Set

| Dataset  | BS | LR |
| ------------- | ------------- | ------------- |
| CoLA  | 128  | 1e-5  |
| SST-2  | 128  | 2e-5  |
| MRPC  | 128  | 5e-5  |
| QQP  | 128  | 3e-5  |
| MNLI  | 128  | 3e-5  |
| QNLI  | 128  | 3e-5  |
| RTE  | 128  | 3e-5  |
| NER  | 128  | 2e-5  |
| POS  | 128  | 3e-5  |
| NC  | 256  | 3e-5  |
| QAM  | 256  | 5e-6  |
| QADSM  | 256  | 3e-5  |
| PAWSX  | 256  | 3e-5  |

## Acknowledgements

Parts of the code are modified from [BERT](https://github.com/jcyk/BERT) and [TaCL](https://github.com/yxuansu/TaCL). We appreciate the authors for making it open-sourced.
