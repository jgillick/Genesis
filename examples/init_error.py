import argparse
import genesis as gs
from genesis.engine.entities import RigidEntity

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
args = parser.parse_args()

def main():
    # Initialize Genesis
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
    gs.init(logging_level="debug", backend=backend, performance_mode=True, debug=True)

    # Do something here
    while True:
        pass

if __name__ == "__main__":
    main()