from reinfin.config import Config
import reinfin.constants as const

from datetime import datetime
import os


class DDQNRunnerConfig(Config):

    logging_path: str
    pipeline_id: str

    seed: int

    train_file: str
    eval_file: str

    start_balance: int
    cash_at_risk: float

    # how many time units in the past (including current time unit) to consider
    lookback: int

    gamma: float
    epsilon: float
    batch_size: int
    eps_min: float
    eps_dec: float

    n_features: int
    lr: float
    hid_out_dims: int
    dropout: float

    replace_cnt: int

    load_checkpoint: bool

    n_games: int

    save_model: bool

    def __init__(self, config):
        super().__init__(config)
        self.image_save_directory = getattr(
            self,
            "save_directory",
            f"images/ddqn/{self.pipeline_id}",
        )
        self.model_save_directory = getattr(
            self,
            "save_directory",
            f"model/ddqn/{self.pipeline_id}",
        )
        # If save path doesn't already exist, create the directory (needed to make new pipeline-named directories)
        if not os.path.exists(self.image_save_directory):
            os.makedirs(self.image_save_directory, exist_ok=True)
            os.makedirs(self.model_save_directory, exist_ok=True)

        self.train_scores_plot_filename = f"train_scores_plot_{self.pipeline_id}.png"
        self.train_scores_plot_path = os.path.join(
            self.image_save_directory, self.train_scores_plot_filename
        )
        self.train_net_worths_plot_filename = (
            f"train_net_worths_plot_{self.pipeline_id}.png"
        )
        self.train_net_worths_plot_path = os.path.join(
            self.image_save_directory, self.train_net_worths_plot_filename
        )
        self.train_scores_learning_plot_filename = (
            f"train_scores_learning_plot_{self.pipeline_id}.png"
        )
        self.train_scores_learning_plot_path = os.path.join(
            self.image_save_directory, self.train_scores_learning_plot_filename
        )
        self.train_net_worths_learning_plot_filename = (
            f"train_net_worths_learning_plot_{self.pipeline_id}.png"
        )
        self.train_net_worths_learning_plot_path = os.path.join(
            self.image_save_directory, self.train_net_worths_learning_plot_filename
        )
        self.train_actions_plot_filename = f"train_actions_plot_{self.pipeline_id}.png"
        self.train_actions_plot_path = os.path.join(
            self.image_save_directory, self.train_actions_plot_filename
        )

        self.eval_scores_plot_filename = f"eval_scores_plot_{self.pipeline_id}.png"
        self.eval_scores_plot_path = os.path.join(
            self.image_save_directory, self.eval_scores_plot_filename
        )
        self.eval_net_worths_plot_filename = (
            f"eval_net_worths_plot_{self.pipeline_id}.png"
        )
        self.eval_net_worths_plot_path = os.path.join(
            self.image_save_directory, self.eval_net_worths_plot_filename
        )
        self.eval_actions_plot_filename = f"eval_actions_plot_{self.pipeline_id}.png"
        self.eval_actions_plot_path = os.path.join(
            self.image_save_directory, self.eval_actions_plot_filename
        )

        self.q_eval_model_filename = f"q_eval_model_{self.pipeline_id}.pt"
        self.q_eval_model_path = os.path.join(
            self.model_save_directory, self.q_eval_model_filename
        )
        self.q_next_model_filename = f"q_next_model_{self.pipeline_id}.pt"
        self.q_next_model_path = os.path.join(
            self.model_save_directory, self.q_next_model_filename
        )

    @property
    def required_config(self):
        return {
            "logging_path": str,
            "pipeline_id": str,
            "seed": int,
            "train_file": str,
            "eval_file": str,
            "start_balance": int,
            "cash_at_risk": float,
            "lookback": int,
            "gamma": float,
            "epsilon": float,
            "batch_size": int,
            "eps_min": float,
            "eps_dec": float,
            "n_features": int,
            "lr": float,
            "hid_out_dims": int,
            "dropout": float,
            "replace_cnt": int,
            "load_checkpoint": self.is_a(bool),
            "n_games": self.is_a(int),
            "save_model": self.is_a(bool),
        }
