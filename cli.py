from __future__ import division

import click
import numpy as np
import os
from cliutils import (CommonOptionGroup, ensure_directory_callback, logging_callback,
                      valid_which_fold, require_value_callback)
from IRT import run_mlp
from IRT.constants import USER_IDX_KEY, SINGLE
# from IRT.split_data import split_data
from IRT.splitting_utils import split_data
from IRT.wrapper import load_data, DataOpts

SKILL_ID_KEY = 'skill_id'
PROBLEM_ID_KEY = 'problem_id'
TEMPLATE_ID_KEY = 'template_id'
USER_ID_KEY = 'user_id'

# Usage: python ./cli.py mlp assistments data/Assistant/skill_builder_data.csv --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col problem_id
# --num-iters 50 --dropout-prob 0.25 --first-learning-rate 5.0 --compress-dim 50 --hidden-dim 100

#        python ./cli.py ncf assistments data/Assistant/skill_builder_data.csv --no-remove-skill-nans --drop-duplicates --num-folds 5
#  --item-id-col problem_id --num-iters 50 --first-learning-rate 5.0  --embedding_dim 210 --hidden-dim 256


# nohup python ./cli.py ncf assistments data/Assistant/skill_builder_data.csv --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col problem_id --num-iters 50 --dropout-prob 0.25 --first-learning-rate 5.0  --embedding_dim 50 --hidden-dim 200 > emb_50_hidden_200.out &


# Visualize call graph: pycallgraph graphviz -- ./cli.py rnn assistments skill_builder_data_corrected_big.txt ...


# Setup common options
common_options = CommonOptionGroup()

# System options
common_options.add('--log-level', '-l', type=click.Choice(['warn', 'info', 'debug']),
                   default='info', help="Set the logging level", extra_callback=logging_callback)
common_options.add('--seed', '-r', type=int, default=0,
                   help="Random number seed for data splitting and model initialization")

# Data options
common_options.add('--remove-skill-nans/--no-remove-skill-nans', is_flag=True, default=False,
                   help="Remove interactions from the data set whose skill_id column is NaN. "
                        "This will occur whether or not the item_id_col is skill_id")
common_options.add('--item-id-col', type=str, nargs=1,
                   help="(Required) Which column should be used for identifying items from the "
                        "dataset. Depends on source as to which names are valid.",
                   extra_callback=require_value_callback((SKILL_ID_KEY, PROBLEM_ID_KEY,
                                                          TEMPLATE_ID_KEY, SINGLE)))
common_options.add('--drop-duplicates/--no-drop-duplicates', default=True,
                   help="Remove duplicate interactions: only the first row is retained for "
                        "duplicate row indices in Assistments")
common_options.add('--max-inter', '-m', type=int, default=0, help="Maximum interactions per user",
                   extra_callback=lambda ctx, param, value: value or None)
common_options.add('--min-inter', type=int, default=2,
                   help="Minimum number of interactions required after filtering to retain a user",
                   extra_callback=lambda ctx, param, value: value or None)
common_options.add('--proportion-students-retained', type=float, default=1.0,
                   help="Proportion of user ids to retain in data set (for testing sensitivity "
                        "to number of data points). Default is 1.0, i.e., all data retained.")

# Learning options
common_options.add('--num-folds', '-f', type=int, nargs=1, default=5,
                   help="Number of folds for testing.", is_eager=True)
common_options.add('--which-fold', type=int, nargs=1, default=None, extra_callback=valid_which_fold,
                   help="If you want to parallelize folds, run several processes with this "
                        "option set to run a single fold. Folds are numbered 1 to --num-folds.")

# Reporting options
common_options.add('--output', '-o', default='rnn_result',
                   help="Where to store the pickled output of training",
                   extra_callback=ensure_directory_callback)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    """ Collection of scripts for evaluating RNN proficiency models """
    pass


@cli.command('rnn')
@click.argument('source')
@click.argument('data_file')
@click.option('--compress-dim', '-d', type=int, nargs=1, default=100,
              help="The dimension to which to compress the input. If -1, will do no compression")
@click.option('--hidden-dim', '-h', type=int, nargs=1, default=100,
              help="The number of hidden units in the RNN.")
@click.option('--output-compress-dim', '-od', type=int, nargs=1, default=None,
              help="The dimension to which we should compress the output vector. "
                   "If not passed, no compression will occur.")
@click.option('--test-spacing', '-t', type=int, nargs=1, default=10,
              help="How many iterations before running the tests?")
@click.option('--recurrent/--no-recurrent', default=True,
              help="Whether to use a recurrent architecture")
@click.option('--use-correct/--no-use-correct', default=True,
              help="If True, record correct and incorrect responses as different input dimensions")
@click.option('--num-iters', '-n', type=int, default=50,
              help="How many iterations of training to perform on the RNN")
@click.option('--dropout-prob', '-p', type=float, default=0.0,
              help="The probability of a node being dropped during training. Default is 0.0 "
                   "(i.e., no dropout)")
@click.option('--use-hints/--no-use-hints', default=False,
              help="Should we add a one-hot dimension to represent whether a student used a hint?")
@click.option('--first-learning-rate', nargs=1, default=30.0, type=float,
              help="The initial learning rate. Will decay at rate `decay_rate`. Default is 30.0.")
@click.option('--decay-rate', nargs=1, default=0.99, type=float,
              help="The rate at which the learning rate decays. Default is 0.99.")
@common_options
def rnn(common, source, data_file, compress_dim, hidden_dim, output_compress_dim, test_spacing,
        recurrent, use_correct, num_iters, dropout_prob, use_hints, first_learning_rate,
        decay_rate):
    """ RNN based proficiency estimation.
    SOURCE specifies the student data source, and should be 'assistments' or 'kddcup'.
    DATA_FILE is the filename for the interactions data.
    """
    print("common.max_inter" + str(common.max_inter))
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=None, template_id_col=None, use_correct=use_correct,
                         remove_skill_nans=common.remove_skill_nans, seed=common.seed,
                         use_hints=use_hints,
                         drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained, meta=False)

    data, _, item_ids, _, _ = load_data(data_file, source, data_opts)

    num_questions = len(item_ids)
    data_folds = split_data(data, num_folds=common.num_folds, seed=common.seed)

    # run_rnn.run(data_folds, common.num_folds, num_questions, num_iters, output=common.output,
    #             compress_dim=compress_dim, hidden_dim=hidden_dim, test_spacing=test_spacing,
    #             recurrent=recurrent, data_opts=data_opts, dropout_prob=dropout_prob,
    #             output_compress_dim=output_compress_dim,
    #             first_learning_rate=first_learning_rate, decay_rate=decay_rate,
    #             which_fold=common.which_fold)


@cli.command('ncf')
@click.argument('source')
@click.argument('data_file')
@click.option('--hidden-dim', '-h', type=int, nargs=1, default=100,
              help="The number of hidden units in the RNN.")
@click.option('--embedding_dim', '-h', type=int, nargs=1, default=200,
              help="Dimension of embeddings units in the MLP.")
@click.option('--test-spacing', '-t', type=int, nargs=1, default=10,
              help="How many iterations before running the tests?")
@click.option('--use-correct/--no-use-correct', default=True,
              help="If True, record correct and incorrect responses as different input dimensions")
@click.option('--num-iters', '-n', type=int, default=50,
              help="How many iterations of training to perform on the RNN")
# @click.option('--dropout-prob', '-p', type=float, default=0.0,
#               help="The probability of a node being dropped during training. Default is 0.0 "
#                    "(i.e., no dropout)")
@click.option('--use-hints/--no-use-hints', default=False,
              help="Should we add a one-hot dimension to represent whether a student used a hint?")
@click.option('--first-learning-rate', nargs=1, default=0.001, type=float,
              help="The initial learning rate. Will decay at rate `decay_rate`. Default is 30.0.")
@click.option('--prediction_output', default="data/Assistant/prediction/", type=str,
              help="where output csv goes")
@common_options
def ncf(common, source, data_file, hidden_dim, embedding_dim, test_spacing, use_correct, num_iters, use_hints,
        first_learning_rate, prediction_output):
    """
    MLP based correctness prediction
    """
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=None, template_id_col=None, use_correct=use_correct,
                         remove_skill_nans=common.remove_skill_nans, seed=common.seed,
                         use_hints=use_hints,
                         drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained, meta=False,
                         prediction_output=prediction_output + data_file.split("/")[-1])

    if os.path.exists(data_opts.prediction_output):
        os.remove(data_opts.prediction_output)
    # input(data_opts)
    data, user_ids, item_ids, _, _ = load_data(data_file, source, data_opts)

    num_questions = len(item_ids)
    data_folds = split_data(data, num_folds=common.num_folds, seed=common.seed, by_interaction=True)

    run_mlp.run(data_folds, common.num_folds, num_questions, num_iters,
                hidden_dim=hidden_dim, test_spacing=test_spacing,
                data_opts=data_opts,
                first_learning_rate=first_learning_rate,
                which_fold=common.which_fold, num_users=len(user_ids), user_ids=user_ids, item_ids=item_ids,
                embedding_dim=embedding_dim)


@cli.command('mlp')
@click.argument('source')
@click.argument('data_file')
@click.option('--hidden-dim', '-h', type=int, nargs=1, default=100,
              help="The number of hidden units in the MLP.")
@click.option('--test-spacing', '-t', type=int, nargs=1, default=10,
              help="How many iterations before running the tests?")
@click.option('--use-correct/--no-use-correct', default=True,
              help="If True, record correct and incorrect responses as different input dimensions")
@click.option('--num-iters', '-n', type=int, default=50,
              help="How many iterations of training to perform on the RNN")
@click.option('--dropout-prob', '-p', type=float, default=0.0,
              help="The probability of a node being dropped during training. Default is 0.0 "
                   "(i.e., no dropout)")
@click.option('--use-hints/--no-use-hints', default=False,
              help="Should we add a one-hot dimension to represent whether a student used a hint?")
@click.option('--first-learning-rate', nargs=1, default=0.001, type=float,
              help="The initial learning rate. Will decay at rate `decay_rate`. Default is 30.0.")
@click.option('--prediction_output', default="data/Assistant/prediction/", type=str,
              help="where output csv goes")
@common_options
def mlp(common, source, data_file, hidden_dim, test_spacing, use_correct, num_iters, dropout_prob, use_hints,
        first_learning_rate, prediction_output):
    """
    MLP based correctness prediction
    """
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=None, template_id_col=None, use_correct=use_correct,
                         remove_skill_nans=common.remove_skill_nans, seed=common.seed,
                         use_hints=use_hints,
                         drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained, meta=True,
                         prediction_output=prediction_output)

    # You might probably want to include more metadata features

    data, user_ids, item_ids, template_ids, concept_ids = load_data(data_file, source, data_opts)

    # normalize_by_columns

    num_questions = len(item_ids)
    data_folds = split_data(data, num_folds=common.num_folds, seed=common.seed)

    run_mlp.run(data_folds, common.num_folds, num_questions, num_iters,
                hidden_dim=hidden_dim, test_spacing=test_spacing,
                data_opts=data_opts,
                first_learning_rate=first_learning_rate,
                which_fold=common.which_fold)


# @cli.command('irt')
# @click.argument('source')
# @click.argument('data_file')
# @click.option('--twopo/--onepo', default=False, help="Use a 2PO model (default is False)")
# @click.option('--concept-id-col', type=str, nargs=1,
#               help="(Required) Which column should be used for identifying "
#                    "concepts from the dataset. Depends on source as to which names are valid. "
#                    "If ``single``, use single dummy concept.",
#               callback=require_value_callback((SKILL_ID_KEY, PROBLEM_ID_KEY, SINGLE)))
# @click.option('--template-id-col', type=str, default=None, nargs=1,
#               help="If using templates, this option is used to specify the column in the dataset "
#                    "you are using to represent the template id")
# @click.option('--template-precision', default=None, type=float, nargs=1,
#               help="Use template_id in IRT learning. Item means will be distributed around a "
#                    "template mean. The precision of that distribution is the argument of this "
#                    "parameter.")
# @click.option('--item-precision', default=None, type=float, nargs=1,
#               help="If using a non-templated model, this is the precision of the Gaussian "
#                    "prior around item difficulties. If using a templated model, it is the "
#                    "precision of the template hyperprior's mean. Default is 1.0.")


# python cli.py naive assistments data/Assistant/skill_builder_data.csv --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col problem_id

@cli.command('naive')
@click.argument('source')
@click.argument('data_file')
@common_options
def naive(common, source, data_file):
    """ Just report the percent correct across all events.
    SOURCE specifies the student data source, and should be 'assistments' or 'kddcup'.
    DATA_FILE is the filename for the interactions data.
    What naive classifier does is flipping the inferior group's wrong answer to be correct.
    Percentage correct here is a micro-average
    """
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=None, template_id_col=None, use_correct=True,
                         remove_skill_nans=common.remove_skill_nans, seed=common.seed,
                         use_hints=True,
                         drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained)
    data, _, _, _, _ = load_data(data_file, source, data_opts)
    print("Percentage correct in data set is {}".format(data.correct.mean()))

    agged = data.groupby(USER_IDX_KEY).correct.agg([np.sum, len]).reset_index()
    mask = agged['sum'] <= agged['len'] // 2
    agged.loc[mask, 'sum'] = agged.loc[mask, 'len'] - agged.loc[
        mask, 'sum']  # agged.loc['len', mask] - agged.loc['sum', mask]
    print("Percent correct for pseudo naive classifier is {}".format(agged['sum'].sum() /
                                                                     agged['len'].sum()))


def main():
    cli()


if __name__ == '__main__':
    main()
