import os
import sys
import time

def run_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    sys.path.insert(0, base_dir)

    print("\n" + "#" * 80)
    print("   STARTING COMPLETE END-TO-END ELLIPTIC AML & PGM GNN PIPELINE   ")
    print("#" * 80 + "\n")

    t_start = time.time()

    # Step 1: Preprocessing
    print("\n>>> [STEP 1/5] RUNNING DATA PREPROCESSING & GRAPH CONSTRUCTION...")
    from preprocess import run_preprocessing
    run_preprocessing(output_dir=base_dir)

    # Step 2: Supervised Baseline Benchmarking
    print("\n>>> [STEP 2/5] RUNNING SUPERVISED BASELINE MODEL BENCHMARKING...")
    from baseline_models import run_baseline_benchmark
    run_baseline_benchmark(output_dir=base_dir)

    # Step 3: Semi-Supervised GAT Pseudo-Labeling & Retraining
    print("\n>>> [STEP 3/5] RUNNING SEMI-SUPERVISED GAT PSEUDO-LABELING & RETRAINING...")
    from run_gat_pseudolabeling_evaluation import run_gat_pseudolabeling
    run_gat_pseudolabeling(output_dir=base_dir)

    # Step 4: Unlabeled Risk Profiling & Prediction Export
    print("\n>>> [STEP 4/5] RUNNING UNLABELED NODE INFERENCE & RISK PROFILING...")
    from run_unlabeled_inference import run_unlabeled_inference
    run_unlabeled_inference(output_dir=base_dir)

    # Step 5: Visualizations & Master Report Generation
    print("\n>>> [STEP 5/5] GENERATING PUBLICATION PLOTS & MASTER WORD REPORT...")
    from plot_confusion_metrics import generate_all_plots
    generate_all_plots(output_dir=base_dir)

    from create_master_docx_report import build_master_word_report
    build_master_word_report(output_dir=base_dir)

    t_elapsed = time.time() - t_start
    print("\n" + "#" * 80)
    print(f"   ALL PIPELINE STAGES FINISHED IN {t_elapsed/60:.2f} MINUTES!   ")
    print("#" * 80 + "\n")

if __name__ == '__main__':
    run_all()
