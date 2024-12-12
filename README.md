# Toward Understanding Hallucinations through Latent Knowledge

[Paper Link](https://www.overleaf.com/project/675795a7465199b27c6dc18d)

This is a Repository of all the Experiements conducted to gather more deep understanding into Extraction of Latent Knowledge by Using a particular technique of Contrast Consistent Search

The Study can be Segmented into the following Categories
- Reimplementation of the Paper that isnpired this project. [Burns et al](https://arxiv.org/pdf/2212.03827)
- Pilot Study on Vision Language models and Analysis of POPE Dataset (VLM_Pilot_Study.ipynb)
- Extension of CCS in Vision-Language Models (VLM_CCS Experiment)
- Contrastive Study design between Performance of Base model Vs CCS on CLIP and LLaVa 1.5B (CCL_CLIP_LLaVa.ipynb)

### Reimplementation of the paper
In the LLM reimplementation part, we tested CCS on every layer of LLama-3.2 (instruct and base version) Qwen-2.5 (instruct and base version) using different databases. 
Compared to the original code, main modifications are made to `LLM_reimplementation/utils.get_individual_hidden_states`,
`LLM_reimplementation/utils.get_dataloader`, `LLM_reimplementation/utils.ContrastDataset`

#### Args used
Model: ['Qwen/Qwen2.5-3B-Instruct','meta-llama/Llama-3.2-3B-Instruct'] 

Dataset: ['imdb','amazon_polarity','ag_news','dbpedia_14','multi_nli',["glue","qnli"],"gimmaru/story_cloze-2016","piqa"]

Split: ['test','test','test','test','test','validation_mismatched','test','train']

Prompt: ['imdb','amazon_polarity','ag_news','dbpedia_14','multi_nli',["glue","qnli"],"story_cloze/2016","piqa"]

#### Running the Code
1. To generate the hidden layers embedding on one model and one dataset, please run:

`python LLM_reimplementation/generate.py --model_name [] --dataset_name [] --split [] --prompt_name [] --all_layers`

2. Given a list of models and a list of datasets and splits, using the hidden embedding already gotten to get CCS performance for each model, each layer, each dataset.

`python LLM_reimplementation/evaluation.py --models [] --datasets [] --splits [] --save_dir [] --all_layers`

3. code for ploting is in LLM_reimplementation/plot.py

Running environment is in `requirements.txt`



### Pilot Study on Vison Language models and Analysis of POPE Dataset + Contrastive Study design between Performance of Base model Vs CCS on CLIP and LLaVa 1.5B
#### Running the Code
All the `.ipynb` files can be run as is, it includes all the dependencies that need to be installed and are plug and play.






### Extension of CCS in Vision-Language Models 

In the VLM_hallucination part, we studied the hallucination problem using LLaVa-1.5-7b-hf model, and POPE dataset.

#### Running the Code
1. We use VLM_hallucination/VLM_hallucination_answer_generate.py to generate model answers to POPE dataset, as well as the logits value of answers. Please run:

`python VLM_hallucination/VLM_hallucination_answer_generate.py --dataset_name lmms-lab/POPE default --save_logits_dir []`

2. After we get the hallucination dataset, we generate the hidden embeddings of hallucination contrast pairs. Please run:

`python VLM_hallucination/VLM_hallucination_pair_generate.py --dataset_name lmms-lab/POPE default --save_logits_dir [] --save_hidden_dir []`

3. After we get the hidden embedding, we test CCS's performance as a hallucination classifier. Please run:

`python VLM_hallucination/evaluate.py --dataset_name lmms-lab/POPE default --save_hidden_dir []`

Running environment is in `requirements.txt`
