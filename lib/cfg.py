import argparse


def str2bool(v):
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        return argparse.ArgumentTypeError("Boolean value expected.")


class Parameters:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch MI Net")
        parser.add_argument("--iter-num", type=int, default=10000, help="Number of iteration.")
        parser.add_argument(
            "--lr", type=float, default=0.0001, help="Base learning rate for training with polynomial decay."
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
        parser.add_argument("--weight-decay", type=float, default=0.005, help="Weight decay for SGD")
        parser.add_argument("--seed", type=int, default=219, help="Random seed to have reproducible results.")

        parser.add_argument("--dataset", type=str, default="duts-te", help="Specify the dataset to use.")
        parser.add_argument("--prior", type=str, default="srm", help="Prior method.")
        parser.add_argument("--test-dataset", type=str, default="ecssd", help="Specify the dataset to use.")
        parser.add_argument("--test-prior", type=str, default="ras", help="Prior method.")

        parser.add_argument(
            "--batch-size", type=int, default=4, help="Number of images sent to the network in one step for one GPU."
        )
        parser.add_argument("--num-workers", type=int, default=4, help="Worker for each process.")

        parser.add_argument("--backbone", type=str, default="mobilenetv2", help="Backbone name.")
        parser.add_argument("--sync-bn", type=str2bool, default="True", help="Backbone name.")

        parser.add_argument("--t-afg", type=float, default=0.8, help="Foreground Threshold.")
        parser.add_argument("--t-abg", type=float, default=0.3, help="Background Threshold.")
        parser.add_argument("--t-fg", type=float, default=0.9, help="Strict Foreground Threshold.")

        parser.add_argument("--print-interval", type=int, default=50, help="Print interval.")
        parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval.")
        parser.add_argument("--ckp-interval", type=int, default=500, help="Evaluation interval.")

        parser.add_argument("--ckp", type=str, default="./ckp/model.pth", help="Checkpoint path.")

        parser.add_argument("--distributed", type=str2bool, default="True", help="Use distributed.")
        parser.add_argument("--world-size", type=int, default=1, help="Distributed world size.")
        parser.add_argument("--ngpus-per-node", type=int, default=1, help="Distributed GPU numbers.")
        parser.add_argument("--rank", type=int, default=0, help="Distributed rank.")
        parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:21992", help="Distributed URL.")
        parser.add_argument("--gpu-id", type=int, default=0, help="Distributed GPU ID.")

        parser.add_argument("--test", type=str2bool, default="False", help="If test mode.")
        parser.add_argument("--save-result", type=str2bool, default="False", help="If save result.")

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        with open("./log/log.txt", "w") as writer:
            if not args.test:
                print("========= Configuration ==========")
                for key, val in vars(args).items():
                    line = "{:16} {}".format(key, val)
                    print(line)
                    writer.write(line + "\n")
                print("==================================")
        return args
