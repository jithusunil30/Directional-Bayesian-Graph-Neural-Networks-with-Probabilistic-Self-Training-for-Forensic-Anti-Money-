import os
import sys
import time

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    sys.path.insert(0, base_dir)

    print("\n" + "#" * 85)
    print("   LAUNCHING ML TRAINING 3: ADVANCED DIRECTIONAL & CONFIDENCE-WEIGHTED GNNs   ")
    print("#" * 85 + "\n")

    t_start = time.time()
    from train_advanced_gnns import main as run_pipeline
    run_pipeline()

    print("\n" + "#" * 85)
    print(f"   ML TRAINING 3 COMPLETED IN {(time.time()-t_start)/60:.2f} MINUTES!   ")
    print("#" * 85 + "\n")

if __name__ == '__main__':
    main()
