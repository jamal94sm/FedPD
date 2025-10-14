
from .Local import run as run_local
from .FedAvg import run as run_fedavg
from .FedMD import run as run_fedmd
from .Zero_Shot import run as run_zero_shot
from .Open_Vocab import run as run_open_vocab
from .FL_Vocab import run as run_fl_vocab
from .SID_CLIP import run as run_sid_clip
from .KOALA import run as run_koala
from .Proposed import run as run_proposed
from .FedMD_mix import run as run_fedmd_mix


# setup_registry = {
#     "fedavg": run_fedavg,
#     "zero_shot": run_zero_shot,
#     "koala": run_koala,
#     "sidclip": run_sidclip,
#     "open_vocab": run_open_vocab,
#     "fl_vocab": run_fl_vocab,
#     "local": run_local,
#     "fedmd": run_fedmd,
# }