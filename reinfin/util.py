import logging
import os
from datetime import date
import time
import random
import string
import itertools
import copy
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


def get_pipeline_id(user: str = "") -> str:
    if not user:
        user = os.getlogin()
    user.replace(".", "")

    today = date.today()
    today_str = today.strftime("%Y%m%d")

    random.seed(int(time.time() * 1000) % 2**32)
    rand_str = "".join(
        random.choices(population=string.ascii_letters + string.digits, k=4)
    )

    return "_".join([user, today_str, rand_str])


def return_updated_dict(d, u):
    """

    Args:
        d: dict to be updated
        u: single-key dict to be appended to d

    Returns: updated dict

    """
    d.update(u)
    return d


def plot_learning_curve(scores, figure_file):
    plt.clf()
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(running_avg)
    plt.title("Running average of previous 100 episodes")
    plt.savefig(figure_file)


def plot_curve(scores, figure_file):
    plt.clf()
    x = [i + 1 for i in range(len(scores))]
    plt.plot(x, scores)
    logging.info(f"Saving plot at {figure_file}.")
    plt.savefig(figure_file)
