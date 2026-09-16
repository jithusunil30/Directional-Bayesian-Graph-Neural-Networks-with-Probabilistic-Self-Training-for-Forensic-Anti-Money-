import os
import sys
import time

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    sys.path.insert(0, base_dir)

    print("\n" + "#" * 80)
    print("   ML TRAINING 2: COMPLETE PIPELINE WITH 100% PSEUDO-LABELED GRAPH   ")
    print("#" * 80 + "\n")

    t_start = time.time()

    # Step 1: Label all unlabeled nodes
    print(">>> [PHASE 1/2] PROBABILISTIC PSEUDO-LABELING OF 100% UNLABELED NODES...")
    from label_unlabeled_nodes import label_all_unlabeled_nodes
    label_all_unlabeled_nodes(output_dir=base_dir)

    # Step 2: Train Graphical ML & Bayesian GNNs
    print("\n>>> [PHASE 2/2] TRAINING PURE GRAPHICAL ML & BAYESIAN GNNs...")
    from train_fully_labeled_gnns import main as run_train
    run_train()

    print("\n" + "#" * 80)
    print(f"   ML TRAINING 2 COMPLETED ALL STAGES IN {(time.time()-t_start)/60:.2f} MINUTES!   ")
    print("#" * 80 + "\n")

if __name__ == '__main__':
    main()
