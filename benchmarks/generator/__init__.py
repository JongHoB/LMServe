from .request import *
from .request_generator import (generate_requests, generate_radom_requests,
                                generate_trace, generate_chat_requests)
from .dataset_config import dataset_configs

supported_dataset_names = [
    name for name, config in dataset_configs.items() if not config.multi_turn
]
supported_multi_turn_dataset_names = [
    name for name, config in dataset_configs.items() if config.multi_turn
]
