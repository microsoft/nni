import tensorflow as tf
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
flags = tf.app.flags
FLAGS = flags.FLAGS

def macro_init():
    DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
    DEFINE_string("output_dir", "", "")
    DEFINE_string("search_for", "macro", "Must be [macro|micro]")

    DEFINE_integer("batch_size", 128, "")
    DEFINE_integer("num_epochs", 310, "")
    DEFINE_integer("train_data_size", 45000, "")

    DEFINE_integer("child_num_layers", 12, "")
    DEFINE_integer("child_num_branches", 6, "")
    DEFINE_integer("child_out_filters", 36, "")
    DEFINE_integer("child_num_cells", 5, "")

    DEFINE_float("controller_lr", 0.001, "")
    DEFINE_float("controller_lr_dec_rate", 1.0, "")
    DEFINE_float("controller_keep_prob", 0.5, "")
    DEFINE_float("controller_l2_reg", 0.0, "")
    DEFINE_float("controller_bl_dec", 0.99, "")
    DEFINE_float("controller_tanh_constant", 1.5, "")
    DEFINE_float("controller_op_tanh_reduce", 2.5, "")
    DEFINE_float("controller_entropy_weight", 0.0001, "")
    DEFINE_float("controller_skip_target", 0.4, "")
    DEFINE_float("controller_skip_weight", 0.8, "")
    DEFINE_float("controller_temperature", None, "")
    DEFINE_integer("controller_num_aggregate", 20, "")
    DEFINE_integer("controller_num_replicas", 1, "")
    DEFINE_integer("controller_train_steps", 50, "")
    DEFINE_integer("controller_forwards_limit", 2, "")
    DEFINE_integer("controller_train_every", 1,
                   "train the controller after this number of epochs")
    DEFINE_boolean("controller_search_whole_channels", True, "")
    DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
    DEFINE_boolean("controller_training", True, "")
    DEFINE_boolean("controller_use_critic", False, "")

    DEFINE_integer("log_every", 50, "How many steps to log")
    DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

    DEFINE_string("tuner_class_name", "", "")
    DEFINE_string("tuner_class_filename", "", "")
    DEFINE_string("tuner_args", "", "")
    DEFINE_string("tuner_directory", "", "")
    DEFINE_string("assessor_class_name", "", "")
    DEFINE_string("assessor_args", "", "")
    DEFINE_string("assessor_directory", "", "")
    DEFINE_string("assessor_class_filename", "", "")
    DEFINE_boolean("multi_phase", True, "")
    DEFINE_boolean("multi_thread", True, "")


def micro_init():

    DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
    DEFINE_string("output_dir", "", "")
    DEFINE_string("search_for", "micro", "Must be [macro|micro]")

    DEFINE_integer("batch_size", 160, "")
    DEFINE_integer("num_epochs", 150, "")
    DEFINE_integer("train_data_size", 45000, "")

    DEFINE_integer("child_num_layers", 6, "")
    DEFINE_integer("child_num_branches", 5, "")
    DEFINE_integer("child_out_filters", 20, "")
    DEFINE_integer("child_num_cells", 5, "")

    DEFINE_float("controller_lr", 0.0035, "")
    DEFINE_float("controller_lr_dec_rate", 1.0, "")
    DEFINE_float("controller_keep_prob", 0.5, "")
    DEFINE_float("controller_l2_reg", 0.0, "")
    DEFINE_float("controller_bl_dec", 0.99, "")
    DEFINE_float("controller_tanh_constant", 1.10, "")
    DEFINE_float("controller_op_tanh_reduce", 2.5, "")
    DEFINE_float("controller_entropy_weight", 0.0001, "")
    DEFINE_float("controller_skip_target", 0.4, "")
    DEFINE_float("controller_skip_weight", 0.8, "")
    DEFINE_float("controller_temperature", None, "")
    DEFINE_integer("controller_num_aggregate", 10, "")
    DEFINE_integer("controller_num_replicas", 1, "")
    DEFINE_integer("controller_train_steps", 30, "")
    DEFINE_integer("controller_forwards_limit", 2, "")
    DEFINE_integer("controller_train_every", 1,
                   "train the controller after this number of epochs")
    DEFINE_boolean("controller_search_whole_channels", True, "")
    DEFINE_boolean("controller_sync_replicas", True, "To sync or not to sync.")
    DEFINE_boolean("controller_training", True, "")
    DEFINE_boolean("controller_use_critic", False, "")

    DEFINE_integer("log_every", 50, "How many steps to log")
    DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

    DEFINE_string("tuner_class_name", "", "")
    DEFINE_string("tuner_class_filename", "", "")
    DEFINE_string("tuner_args", "", "")
    DEFINE_string("tuner_directory", "", "")
    DEFINE_string("assessor_class_name", "", "")
    DEFINE_string("assessor_args", "", "")
    DEFINE_string("assessor_directory", "", "")
    DEFINE_string("assessor_class_filename", "", "")
    DEFINE_boolean("multi_phase", True, "")
    DEFINE_boolean("multi_thread", True, "")