import tensorflow as tf
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("search_for", None, "[rhn|base|enas]")

DEFINE_string("child_fixed_arc", None, "")
DEFINE_integer("batch_size", 25, "")
DEFINE_integer("child_base_number", 4, "")
DEFINE_integer("child_num_layers", 2, "")
DEFINE_integer("child_bptt_steps", 20, "")
DEFINE_integer("child_lstm_hidden_size", 200, "")
DEFINE_float("child_lstm_e_keep", 1.0, "")
DEFINE_float("child_lstm_x_keep", 1.0, "")
DEFINE_float("child_lstm_h_keep", 1.0, "")
DEFINE_float("child_lstm_o_keep", 1.0, "")
DEFINE_boolean("child_lstm_l_skip", False, "")
DEFINE_float("child_lr", 1.0, "")
DEFINE_float("child_lr_dec_rate", 0.5, "")
DEFINE_float("child_grad_bound", 5.0, "")
DEFINE_float("child_temperature", None, "")
DEFINE_float("child_l2_reg", None, "")
DEFINE_float("child_lr_dec_min", None, "")
DEFINE_float("child_optim_moving_average", None,
             "Use the moving average of Variables")
DEFINE_float("child_rnn_l2_reg", None, "")
DEFINE_float("child_rnn_slowness_reg", None, "")
DEFINE_float("child_lr_warmup_val", None, "")
DEFINE_float("child_reset_train_states", None, "")
DEFINE_integer("child_lr_dec_start", 4, "")
DEFINE_integer("child_lr_dec_every", 1, "")
DEFINE_integer("child_avg_pool_size", 1, "")
DEFINE_integer("child_block_size", 1, "")
DEFINE_integer("child_rhn_depth", 4, "")
DEFINE_integer("child_lr_warmup_steps", None, "")
DEFINE_string("child_optim_algo", "sgd", "")

DEFINE_boolean("child_sync_replicas", False, "")
DEFINE_integer("child_num_aggregate", 1, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_steps", 1327, "")

DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after how many this number of epochs")
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("num_epochs", 300, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")