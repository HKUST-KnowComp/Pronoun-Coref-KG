# Knowledge-aware Pronoun coreference resolution

This is the source code for ACL 2019 paper "Knowledge-aware Pronoun Coreference Resolution".

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:
```
@inproceedings{zhang2019pronounkg,
  author    = {Hongming Zhang and
               Yan Song and
               Yangqiu Song
               and Dong Yu},
  title     = {Knowledge-aware Pronoun Coreference Resolution},
  booktitle = {Proceedings of ACL},
  year      = {2019}
}
```

#Usage

Before repeating our experiment or train your own model, please setup the environment as follows:

1. Download python 3.6 or above and setup the anaconda environment by: conda env create -f environment.yml
2. Download the conll data with ./setup_conll_data.sh (Please replace the location of Ontonotes in the script with the correct one)
3. Download the medical data from the i2b2 website and put it under folder 'medical_data' then prepare the medical data with ./setup_medical_data.sh
4. Setup the pretrain ELMo module by: python cache_elmo train.jsonlines dev.jsonlines test.jsonlines
5. Download and process the word embeddings: ./setup_embedding.sh
6. Setup all other components for training with ./setup_training.sh

You can train our model on two datasets with python Train_Pronoun.py conll/medical

If you want to try other experiment settings, simply modify the experiments.conf file to add your setting and run: python Train_Pronoun.py YourSettingName

You can test the performance of the trained model by: python Evaluate_Pronoun.py conll/medical/YourSettingName

PS: The provided final_kg.json is collected via the method described in the paper. You can also any knowledge you are interested in into the final_kg.json file.
# Acknowledgment
We built the training framework based on the original [End-to-end Coreference Resolution](https://github.com/kentonl/e2e-coref).

# Others
If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.