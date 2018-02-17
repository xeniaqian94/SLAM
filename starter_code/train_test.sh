python baseline.py --train ../data/data_es_en/es_en.slam.20171218.train --test ../data/data_es_en/es_en.slam.20171218.dev --pred prediction_es_en/es_en.slam.20171218.pred

read -p "Training finished, starts prediction? "

python eval.py --pred prediction_es_en/es_en.slam.20171218.pred --key ../data/data_es_en/es_en.slam.20171218.dev.key

