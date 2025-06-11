## SAVED

**SAVED** (**S**emantic m**A**tching-based **V**aluable us**E**r i**D**entification) is a novel approach  leveraged app user reviews and update logs to identify valuable users whose feedback has influenced app updates and may contribute continuously and positively to long-term app improvement.

SAVED operates in three key stages. First, a review classifier is designed to extract informative reviews in the app review repository. Second, the approach incorporates temporal constraints and leverages large language models to establish high-precision semantic matching between user reviews and app update logs, thereby detecting instances where user feedback has been explicitly or implicitly addressed by developers. Finally, the model integrates the historical behavior and other multidimensional characteristics of these responded users to complete the identification of valuable users synthetically.



### Usage

#### 1.Download the model and data

pre-trained RoBERTa model: [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main)

classification model of informative reviews [review classification](https://huggingface.co/XQ112/SAVED/resolve/main/chinese_review_classifier.pth?download=true)

user review and update log: [data](https://github.com/qcy321/SAVED/releases/download/Data/data.zip)

Put the `chinese-roberta-wwm-ext-large`  and the  `data` into the `codes` directory, the `review classification` put into `01_preprocess&classification/model_output` directory.

#### 2.Data Preprocessing

src: 01_preprocess&classification

`python rev_preprocess.py`

#### 3.Informative Reviews Extraction

src:  01_preprocess&classification

`python predict.py`

#### 4.Semantic Response-Based Matching

src:  02_semantic_matching

```python
python temporal_constraint.py	# prepare candidate review-log pair
python deepseek.py	# expert_ds
python qwenplus.py	# expert_qw
python consensus_mechanism.py	# dual-expert consensus mechanism 
```

#### 5.Valuable Users Identification 

src:  03_valuable_user_identification

```python
python value_assess.py	# compute user score
python prop_sensitivity.py	# different proportion user
python general_use.py	# extract general user
python active_user.py	# extract active user
python feature_ablation.py	# different feature
```

The above example uses the Tencent Meeting app. If run other app data, modify the file path in the corresponding code.

Reviews of other 215 apps, please  contact the author.

