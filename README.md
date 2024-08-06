# Does DETECTGPT Fully Utilize Perturbation? Bridging Selective Perturbation to Fine-tuned Contrastive Learning Detector would be Better

This repository contains code for *Does DETECTGPT Fully Utilize Perturbation? Bridging Selective Perturbation to Fine-tuned Contrastive Learning Detector would be Better* (https://arxiv.org/pdf/2402.00263, 2024., ACL 2024) by Shengchao Liu, Xiaoming Liu*, Yichen Wang, Zehua Cheng, Chengzhengxu Li, Yu Lan, Chao Shen. In this codebase we provide Pecola, a novel fine-tuned detector bridging metric-based and fine-tuned methods by contrastive learning on selective perturbation.Selective strategy retains important tokens during perturbation and weights for multi-pair contrastive learning.  And we further analyze the effectiveness, robustness, and generalization of the method.


To demonstrate the effectiveness of PECOLA, we conduct extensive experiments on four open-source datasets.

This dataset implements the MGT detection algorithms developed by us (Pecola,ACL 2024) and baselines:

| Dataset |  Paper | 
| --------- | ------ | 
| Grover | [Defending Against Neural Fake News](https://proceedings.neurips.cc/paper/2019/file/3e9f0fc9b2f89e043bc6233994dfcf76-Paper.pdf) (NeurIPS 2019) |
| GPT2 | [Gpt-2 output dataset. Website](https://github.com/openai/gpt-2-output-dataset) |
| GPT3.5 | [COCO: Coherence-Enhanced Machine-Generated Text Detection Under LowResource With Contrastive Learning](https://aclanthology.org/2023.emnlp-main.1005.pdf) (EMNLP 2023) |
| HC3 |  [HowClose is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/pdf/2301.07597) |

 


### Requirements
Python: 3.7.13

To install the dependencies, run
<pre/>pip install -r requirements.txt</pre> 


##### Select Strategy

```
python select_strategy.py --model_name t5-large --input_file <YOUR_DIR> --output_file <YOUR_DIR> --dataset_name grover
```


### Pecola Training

```
python train.py --model roberta-base --output_dir <YOUR_DIR> --seed 41 --dataset grover --log_file <YOUR_LOG_FILE> --lr 1e-5 --epochs 30 --batch_size 16   --loss_type margin_weight
```



## Citation

If you find our work helpful, please cite us with the following BibTex entry:
<pre>
@article{liu2024does,
  title={Does DETECTGPT Fully Utilize Perturbation? Bridging Selective Perturbation to Fine-tuned Contrastive Learning Detector would be Better
},
  author={Liu, Shengchao and Liu, Xiaoming and Wang, Yichen and Cheng, Zehua and Li, Chengzhengxu and Zhang, Zhaohan and Lan, Yu and Shen, Chao},
  journal={arXiv preprint arXiv:2402.00263},
  year={2024}
}
</pre>


