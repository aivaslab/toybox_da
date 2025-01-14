import argparse


def get_default_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--epochs", "-e", default=100, type=int, help="Set the number of epochs of training")
    parser.add_argument("--iters", "-it", default=250, type=int, help="Set the number of iters of training per epoch")
    parser.add_argument("--bsize", "-b", default=128, type=int, help="Set the batch size for experiments")
    parser.add_argument("--workers", "-w", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("--lr", "-lr", default=0.1, type=float, help="Set initial lr for experiments")
    parser.add_argument("-wd", "--wd", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--log", choices=['debug', 'info', 'warning', 'error', 'critical'], default='info',
                        help="Set the log level for the experiment", type=str)
    parser.add_argument("--final", "-f", default=False, action="store_true", help="Use this flag to run the final "
                                                                                  "experiment")
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--model-name", default="final_model.pt", type=str, help="Name of the model to load weights "
                                                                                 "from")
    parser.add_argument("--no-save", action='store_true', default=False, help="Use this option to disable saving")
    parser.add_argument("--save-dir", default="", type=str, help="Directory to save")
    parser.add_argument("--save-freq", "-sf", default=-1, type=int, help="Frequency of saving models")
    return parser


def get_dual_ssl_oriented_v1_parser():
    """Return parser for the experiment"""
    parser = get_default_parser()

    parser.add_argument("--tb-ssl-loss", choices=["simclr", "dcl", "sup_dcl"], default="dcl",
                        help="Use this flag to choose ssl loss for toybox")
    parser.add_argument("--in12-ssl-loss", choices=["simclr", "dcl"], default="dcl",
                        help="Use this flag to choose ssl loss for in12")
    parser.add_argument("--tb-alpha", "-tba", default=1.0, type=float, help="Weight of TB contrastive loss in total "
                                                                            "loss")
    parser.add_argument("--orient-alpha", default=1.0, type=float, help="Weight of orientation loss in total loss")
    parser.add_argument("--in12-alpha", "-in12a", default=1.0, type=float, help="Weight of IN-12 contrastive loss in "
                                                                                "total loss")
    parser.add_argument("--tb-ssl-type", "-tbssl", default="object", choices=['self', 'transform', 'object', 'class'],
                        help="Type of ssl for Toybox")
    parser.add_argument("--in12-ssl-type", "-in12ssl", default="self", choices=['self', 'class'],
                        help="Type of ssl for IN-12")
    parser.add_argument("--use-cosine", default=False, action='store_true', help="Use this flag to use cosine "
                                                                                 "distance for orientation loss")
    parser.add_argument("--ignore-orient-loss", default=False, action='store_true', help="Use this flag to not use "
                                                                                         "orientation loss for "
                                                                                         "training")
    parser.add_argument("--use-v2", default=False, action='store_true', help="Use this flag to use V2 orientation loss")

    parser.add_argument("--show-images", default=False, action='store_true', help="Use this flag to only show images "
                                                                                  "from first training batch")

    return parser


def get_dual_ssl_class_mmd_v1_parser():
    """Return parser for the experiment"""
    parser = get_default_parser()

    parser.add_argument("--tb-ssl-loss", choices=["simclr", "dcl", "sup_dcl"], default="sup_dcl",
                        help="Use this flag to choose ssl loss for toybox")
    parser.add_argument("--in12-ssl-loss", choices=["simclr", "dcl", "nwdcl"], default="dcl",
                        help="Use this flag to choose ssl loss for in12")
    parser.add_argument("--tb-alpha", "-tba", default=0.25, type=float, help="Weight of TB contrastive loss in total "
                                                                            "loss")
    parser.add_argument("--div-alpha", "-da", default=1.0, type=float, help="Weight of orientation loss in total loss")
    parser.add_argument("--in12-alpha", "-in12a", default=1.0, type=float, help="Weight of IN-12 contrastive loss in "
                                                                                "total loss")
    parser.add_argument("--tb-ssl-type", "-tbssl", default="object", choices=['self', 'transform', 'object', 'class'],
                        help="Type of ssl for Toybox")
    parser.add_argument("--in12-ssl-type", "-in12ssl", default="self", choices=['self', 'class'],
                        help="Type of ssl for IN-12")
    parser.add_argument("--ignore-div-loss", default=False, action='store_true', help="Use this flag to not use "
                                                                                      "divergence loss for "
                                                                                      "training")
    parser.add_argument("--asymmetric", "-asym", action='store_true', default=False,
                        help="Use this flag to select asymmetric mmd loss during training")
    parser.add_argument("--use-ot", default=False, action='store_true', help="Use this flag to use OT-based loss "
                                                                             "instead of MMD")
    parser.add_argument("--ind-mmd-loss", default=False, action='store_true', help="Use this flag to match profiles "
                                                                                   "of individual datapoints between "
                                                                                   "source and target")
    parser.add_argument("--div-metric", choices=["euclidean", "cosine", "dot"], default="cosine")
    parser.add_argument("--separate-forward-pass", default=False, action='store_true', help="Use this flag to have "
                                                                                            "separate forward passes "
                                                                                            "for the two datasets")
    parser.add_argument("--ignore-knn-acc", default=False, action='store_true', help="Use this flag to turn off "
                                                                                     "within-batch knn acc tracking")
    parser.add_argument("--queue-factor", "-qf", default=20, type=int, help="Set the size of the knn queue wrt the "
                                                                            "batch size")
    parser.add_argument("--normed-mmd", "-norm-mmd", default=False, action='store_true', help="Use this flag to use "
                                                                                              "normalized version of "
                                                                                              "mmd loss")
    parser.add_argument("--split-div-loss", "-split-div", default=False, action='store_true',
                        help="Use this flag to compute div loss using closest and farthest "
                             "distances instead of all distances")
    parser.add_argument("--split-div-type", "-split-div-type",  choices=["closest", "farthest", "both"], default="both",
                        type=str)
    parser.add_argument("--near-images", "-near", type=int, default=3, help="Number of closest images "
                                                                            "to consider for split div loss")
    parser.add_argument("--far-images", "-far", type=int, default=3, help="Number of farthest images "
                                                                          "to consider for split div loss")
    parser.add_argument("--div-alpha-type", choices=["fixed", "cosine", "linear"], help="Schedule for distribution "
                                                                                        "alpha")
    parser.add_argument("--div-alpha-start", "-div-start", default=50, type=int, help="Schedule for distribution "
                                                                                      "alpha")
    parser.add_argument("--use-ssl-mmd", default=False, action='store_true', help="Use this option to use mmd "
                                                                                  "loss on ssl head feats")
    parser.add_argument("--skip-epochs", "-se", default=0, type=int, help="Set the number of epochs of training to "
                                                                          "skip")
    parser.add_argument("--use-jan", default=False, action='store_true', help="Use this option to apply MMD loss "
                                                                              "directly on features")
    # parser.add_argument("--knn-dist-acc", default=0.05, type=float, )
    return parser
