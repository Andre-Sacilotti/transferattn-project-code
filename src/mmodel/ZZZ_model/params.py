from ..basic_params import basic_parser as p

# training settings

p.add_argument("--batch_size", type=int)


# network settings
p.add_argument("--bottleneck_dim", type=int)

# trianing
p.add_argument("--adv_coeff", type=float)

p.add_argument("--lr_decay_epoch", type=int)

p.add_argument("--lr_gamma", type=float)

p.add_argument("--lr", type=float)

#  dataset settings

p.add_argument("--dataset", type=str)

p.add_argument("--source", type=str)

p.add_argument("--target", type=str)

p.add_argument("--sampled_frame", type=int)

p.add_argument("--queue_size", type=int)
p.add_argument("--IB_weight", type=float)

p.add_argument("--use_ib_loss", action="store_true")
p.add_argument("--use_adv_loss", action="store_true")

p.add_argument("--use_MDTA", action="store_true")

params = p.parse_args()
