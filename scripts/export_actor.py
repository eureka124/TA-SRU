#!/usr/bin/env python3
"""导出纯 PyTorch Actor；不导入 Isaac Lab。"""

import argparse
from ta_sru.export import export_actor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    print(export_actor(args.checkpoint, args.output))
