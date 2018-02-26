# 11-747 Neural Network for NLP
# Second Language Acquisition Modeling

## Dataset

You might want to load raw data/ directory (gitignored, requires local downloads).

	|-data/
   		|-data_en_es/
   		|-data_es_en/
   		|-data_fr_en
   		|-Assistant/

We use a deduplicated version of `skill_builder_data_corrected.csv` from Skill-builder data 2009-2010 https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010


## Command

### On Assistment09 benchmark:

    python ./cli.py ncf assistments data/Assistant/skill_builder_data.csv --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col problem_id --num-iters 50 --first-learning-rate 0.001 --embedding_dim 200 --hidden-dim 200

A .csv dump of cross-validated prediction is located at `data/Assistant/prediction/skill_builder_data.csv`
 
### On Duolingo dataset:

To train a Logistic Regression baseline and predict:
	
	python baseline.py --train ../data/data_es_en/es_en.slam.20171218.train --test ../data/data_es_en/es_en.slam.20171218.dev --pred prediction_es_en/es_en.slam.20171218.pred 

To evaluate:
	
	python eval.py --pred prediction_es_en/es_en.slam.20171218.pred --key ../data/data_es_en/es_en.slam.20171218.dev.key

Output:
	
	Metrics:	accuracy=0.848	avglogloss=0.378	auroc=0.752	F1=0.176
 
