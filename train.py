
from lib.helper.logger import logger
from lib.core.base_trainer.net_work import trainner

import setproctitle
setproctitle.setproctitle("imagenet*_*_")


trainner=trainner()

trainner.train()
