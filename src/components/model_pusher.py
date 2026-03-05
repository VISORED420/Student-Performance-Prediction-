import os
import sys
import shutil
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelPusherConfig:
    model_push_dir: str = os.path.join("saved_models", "model.pkl")


class ModelPusher:
    def __init__(self):
        self.model_pusher_config = ModelPusherConfig()

    def initiate_model_pusher(self, model_path: str):
        try:
            logging.info("Model pusher started")

            os.makedirs(os.path.dirname(self.model_pusher_config.model_push_dir), exist_ok=True)

            shutil.copy(model_path, self.model_pusher_config.model_push_dir)

            logging.info(f"Model pushed to {self.model_pusher_config.model_push_dir}")

            return self.model_pusher_config.model_push_dir

        except Exception as e:
            raise CustomException(e, sys)
