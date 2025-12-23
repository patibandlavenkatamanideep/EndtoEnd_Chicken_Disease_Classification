import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            str(self.config.tensorboard_root_log_dir),
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        # Ensure the checkpoint path ends with .h5
        ckpt_path = str(self.config.checkpoint_model_filepath)
        if not ckpt_path.endswith(".h5"):
            ckpt_path += ".h5"

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True
        )

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]