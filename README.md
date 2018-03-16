# 11-747 Neural Network for NLP
# Second Language Acquisition Modeling

## Dataset

You need to download raw data and organize them in this hierarchy. (currently partially gitignored).

	|-data/
   		|-data_en_es/
   		|-data_es_en/
   		|-data_fr_en/
   		|-Assistant/
			|-skill_builder_data_corrected.csv
			|-skill_builder_data.csv

`data_en_es/` and 2 others can be download from [http://sharedtask.duolingo.com](http://sharedtask.duolingo.com)

We use a deduplicated version of `skill_builder_data.csv`, named as `skill_builder_data_corrected.csv` from [Skill-builder data 2009-2010](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010)


## Command

### On Assistment09 benchmark:

To run a neural collaborative filtering model (based on MLP):

    python ./cli.py ncf assistments data/Assistant/skill_builder_data.csv --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col problem_id --num-iters 50 --first-learning-rate 0.001 --embedding_dim 200 --hidden-dim 200

A `.csv` dump of cross-validated prediction is located at `data/Assistant/prediction/skill_builder_data.csv`

You could also run a naive MLP with pre-defined features (e.g. ms_response, attempt, opportunity, etc. ) from `.csv`

    python ./cli.py mlp assistments data/Assistant/skill_builder_data.csv --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col problem_id --num-iters 50 --dropout-prob 0.25 --first-learning-rate 0.01 --hidden-dim 100
     
### On Duolingo dataset:

To train a (official release baseline) Logistic Regression and predict:

	cd starter_code/
	
	python baseline.py --train ../data/data_es_en/es_en.slam.20171218.train --test ../data/data_es_en/es_en.slam.20171218.dev --pred prediction_es_en/es_en.slam.20171218.pred 

The output will show the total training and dev instances for English -> Spanish are,

    loading 1973558 instances across 731896 exercises.
    loading 288864 instances across 96003 exercises.
    
Then to evaluate:
	
	python eval.py --pred prediction_es_en/es_en.slam.20171218.pred --key ../data/data_es_en/es_en.slam.20171218.dev.key

You will see an output:
	
	Metrics:	accuracy=0.848	avglogloss=0.378	auroc=0.752	F1=0.176


### Under the duolingo_submission_0319 branch:
Our submission attempt to the shared task 

TODO:
1. organize data in csv format, just as assistments.csv 
2. 
