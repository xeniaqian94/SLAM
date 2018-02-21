from __future__ import division

import click
import numpy as np

from cliutils import (CommonOptionGroup, ensure_directory_callback, logging_callback,
                       valid_which_fold, require_value_callback)
# from IRT import run_rnn, rnn
from IRT.constants import USER_IDX_KEY, SINGLE
from IRT.splitting_utils import split_data
from IRT.wrapper import load_data, DataOpts


SKILL_ID_KEY = 'skill_id'
PROBLEM_ID_KEY = 'problem_id'
TEMPLATE_ID_KEY = 'template_id'
USER_ID_KEY = 'user_id'



def rnn(common, source, data_file, compress_dim, hidden_dim, output_compress_dim, test_spacing,
        recurrent, use_correct, num_iters, dropout_prob, use_hints, first_learning_rate,
        decay_rate):
    """ RNN based proficiency estimation.
    SOURCE specifies the student data source, and should be 'assistments' or 'kddcup'.
    DATA_FILE is the filename for the interactions data.
    """
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=None, template_id_col=None, use_correct=use_correct,
                         remove_skill_nans=common.remove_skill_nans, seed=common.seed,
                         use_hints=use_hints,
                         drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained)

    data_opts = DataOpts(num_folds=5, item_id_col="problem_id",
                         concept_id_col=None, template_id_col=None, use_correct=True,
                         remove_skill_nans=False, seed=12345,
                         use_hints=False,
                         drop_duplicates=True,
                         max_interactions_per_user=100,
                         min_interactions_per_user=0,
                         proportion_students_retained=1.0)



    data, _, item_ids, _, _ = load_data(data_file, source, data_opts)

    print(type(data))
    print(data.shape)
    num_questions = len(item_ids)
    data_folds = split_data(data, num_folds=common.num_folds, seed=common.seed)


def main():
    rnn()

if __name__ == '__main__':
    main()



