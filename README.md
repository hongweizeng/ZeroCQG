# SPARTA for ZeroCQG
This repository contains code and models for the paper: [Synthesize, Prompt and Transfer: Zero-shot Conversational Question Generation with Pre-trained Language Model](https://aclanthology.org/2023.acl-long.500/).

# Requirements
```shell
transformers==4.23.1
datasets==2.3.2
allennlp==2.9.3
allennlp-models==2.9.3
torch==1.12.1
nlg-eval==2.3
```

# Running
```shell
CUDA_VISIBLE_DEVICES=0 bash scripts/search-t5.sh
```

# Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{zeng-etal-2023-synthesize,
    title = "Synthesize, Prompt and Transfer: Zero-shot Conversational Question Generation with Pre-trained Language Model",
    author = "Zeng, Hongwei  and Wei, Bifan  and Liu, Jun  and Fu, Weiping",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    year = "2023",
}
```
Please contact us or post an issue if you have any questions.

