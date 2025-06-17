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

    hidden_sizes: list
    num_layers: int
    learning_rate: float
    dropout_list: list
    batch_size: int
    epochs: int

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

        self.model_filename = f"price_predictor_model_{self.pipeline_id}.pt"
        self.model_path = os.path.join(self.model_save_directory, self.model_filename)

    @property
    def required_config(self):
        return {
            "logging_path": str,
            "pipeline_id": str,
            "seed": int,
            "train_file": str,
            "eval_file": str,
            "hidden_sizes": list,
            "num_layers": int,
            "learning_rate": float,
            "dropout_list": list,
            "batch_size": int,
            "epochs": int,
            "load_checkpoint": self.is_a(bool),
            "save_model": self.is_a(bool),
        }
