import eval.meta_disc as md
import sys

if __name__ == "__main__":
    num_args = len(sys.argv)
    checkpoint = sys.argv[1]
    config = sys.argv[2]
    print(config)
    md.train_md(checkpoint, config, 'cuda')