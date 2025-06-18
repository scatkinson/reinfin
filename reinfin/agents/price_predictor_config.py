from reinfin.config import Config
import reinfin.constants as const

from datetime import datetime
import os


class PricePredictorConfig(Config):

    logging_path: str
    pipeline_id: str

    seed: int

    train_file: str
    eval_file: str

    start_p: int
    d: int
    start_q: int
    max_p: int
    max_d: int
    max_q: int
    start_P: int
    D: int
    start_Q: int
    max_P: int
    max_D: int
    max_Q: int
    m: int
    seasonal: bool
    error_action: str
    trace: bool
    suppress_warnings: bool
    stepwise: bool
    n_fits: int

    load_checkpoint: bool

    save_model: bool

    def __init__(self, config):
        super().__init__(config)
        self.image_save_directory = getattr(
            self,
            "save_directory",
            f"images/price_predictor/{self.pipeline_id}",
        )
        self.model_save_directory = getattr(
            self,
            "save_directory",
            f"model/price_predictor/{self.pipeline_id}",
        )
        # If save path doesn't already exist, create the directory (needed to make new pipeline-named directories)
        if not os.path.exists(self.image_save_directory):
            os.makedirs(self.image_save_directory, exist_ok=True)
            os.makedirs(self.model_save_directory, exist_ok=True)

        self.train_loss_plot_filename = f"train_loss_plot_{self.pipeline_id}.png"
        self.train_loss_plot_path = os.path.join(
            self.image_save_directory, self.train_loss_plot_filename
        )
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

        self.eval_loss_plot_filename = f"eval_loss_plot_{self.pipeline_id}.png"
        self.eval_loss_plot_path = os.path.join(
            self.image_save_directory, self.eval_loss_plot_filename
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

        self.model_filename = f"price_predictor_model_{self.pipeline_id}.csv"
        self.model_path = os.path.join(self.model_save_directory, self.model_filename)

    @property
    def required_config(self):
        return {
            "logging_path": str,
            "pipeline_id": str,
            "seed": self.is_a(int),
            "train_file": str,
            "eval_file": str,
            "start_p": self.is_a(int),
            "d": self.is_a(int),
            "start_q": self.is_a(int),
            "max_p": self.is_a(int),
            "max_d": self.is_a(int),
            "max_q": self.is_a(int),
            "start_P": self.is_a(int),
            "D": self.is_a(int),
            "start_Q": self.is_a(int),
            "max_P": self.is_a(int),
            "max_D": self.is_a(int),
            "max_Q": self.is_a(int),
            "m": self.is_a(int),
            "seasonal": self.is_a(bool),
            "error_action": self.is_a(str),
            "trace": self.is_a(bool),
            "suppress_warnings": self.is_a(bool),
            "stepwise": self.is_a(bool),
            "n_fits": self.is_a(int),
            "load_checkpoint": self.is_a(bool),
            "save_model": self.is_a(bool),
        }
