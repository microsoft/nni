import tensorflow as tf
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
flags = tf.app.flags
FLAGS = flags.FLAGS


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "data/ptb/ptb.pkl", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("search_for", "enas", "[rhn|base|enas]")

DEFINE_string("child_fixed_arc", None, "")
DEFINE_integer("batch_size", 20, "")
DEFINE_integer("child_base_number", 100, "")
DEFINE_integer("child_num_layers", 1, "")
DEFINE_integer("child_bptt_steps", 35, "")
DEFINE_integer("child_lstm_hidden_size", 720, "")
DEFINE_float("child_lstm_e_keep", 0.75, "")
DEFINE_float("child_lstm_x_keep", 0.25, "")
DEFINE_float("child_lstm_h_keep", 0.75, "")
DEFINE_float("child_lstm_o_keep", 0.25, "")
DEFINE_boolean("child_lstm_l_skip", False, "")
DEFINE_float("child_lr", 0.25, "")
DEFINE_float("child_lr_dec_rate", 0.95, "")
DEFINE_float("child_grad_bound", 10.0, "")
DEFINE_float("child_temperature", None, "")
DEFINE_float("child_l2_reg", 1e-7, "")
DEFINE_float("child_lr_dec_min", 0.0005, "")
DEFINE_float("child_optim_moving_average", None,
             "Use the moving average of Variables")
DEFINE_float("child_rnn_l2_reg", None, "")
DEFINE_float("child_rnn_slowness_reg", None, "")
DEFINE_float("child_lr_warmup_val", None, "")
DEFINE_float("child_reset_train_states", None, "")
DEFINE_integer("child_lr_dec_start", 12, "")
DEFINE_integer("child_lr_dec_every", 1, "")
DEFINE_integer("child_avg_pool_size", 1, "")
DEFINE_integer("child_block_size", 1, "")
DEFINE_integer("child_rhn_depth", 12, "")
DEFINE_integer("child_lr_warmup_steps", None, "")
DEFINE_string("child_optim_algo", "sgd", "")

DEFINE_boolean("child_sync_replicas", False, "")
DEFINE_integer("child_num_aggregate", 1, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_steps", 1327, "")

DEFINE_float("controller_lr", 0.001, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", 2.5, "")
DEFINE_float("controller_temperature", 5.0, "")
DEFINE_float("controller_entropy_weight", 0.001, "")
DEFINE_float("controller_skip_target", None, "")
DEFINE_float("controller_skip_rate", None, "")

DEFINE_integer("controller_num_aggregate", 10, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 100, "")
DEFINE_integer("controller_train_every", 1,
               "train the controller after how many this number of epochs")
DEFINE_boolean("controller_sync_replicas", True, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("num_epochs", 100, "")

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