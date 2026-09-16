import os
import sys
import numpy as np
import pandas as pd
import torch
from baseline_models import GATModel

def run_unlabeled_inference(pyg_path=None, model_path=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    if pyg_path is None:
        pyg_path = os.path.join(output_dir, 'elliptic_pyg_data.pt')
    if model_path is None:
        model_path = os.path.join(output_dir, 'gat_pseudolabeled_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(output_dir, 'gat_supervised_model.pt')

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("   UNLABELED NODE INFERENCE & BITCOIN TRANSACTION RISK PROFILING   ")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Graph Data from: {pyg_path}")
    data = torch.load(pyg_path, weights_only=False)

    in_feat = data.x.shape[1]
    print(f"Loading GAT Model weights from: {model_path}")
    gat = GATModel(in_features=in_feat, hidden_dim=64, heads=4, dropout_rate=0.3).to(device)
    gat.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    gat.eval()

    x_dev = data.x.to(device)
    edge_index_dev = data.edge_index.to(device)

    print("Executing full graph forward inference...")
    with torch.no_grad():
        logits = gat(x_dev, edge_index_dev)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Filter for all unlabeled nodes
    unlabeled_mask = data.unlabeled_mask.numpy()
    unlabeled_indices = np.where(unlabeled_mask)[0]

    unlabeled_txId = data.txId.numpy()[unlabeled_indices]
    unlabeled_steps = data.time_step.numpy()[unlabeled_indices]
    unlabeled_probs = probs[unlabeled_indices]

    # Calculate metrics for each unlabeled node
    # Binary prediction at standard 0.5 threshold
    pred_class = (unlabeled_probs >= 0.5).astype(int)
    pred_label_str = np.where(pred_class == 1, 'Illicit', 'Licit')
    
    # Risk Score (0-100)
    risk_score = np.round(unlabeled_probs * 100, 2)
    
    # Risk Tier Categorization
    risk_tier = np.where(unlabeled_probs >= 0.70, 'High Risk (>=70%)',
                np.where(unlabeled_probs >= 0.30, 'Medium Risk (30-70%)', 'Low Risk (<30%)'))

    # Confidence Score: max(P, 1-P)
    confidence = np.maximum(unlabeled_probs, 1.0 - unlabeled_probs)

    # Shannon Entropy Uncertainty
    eps = 1e-9
    p_safe = np.clip(unlabeled_probs, eps, 1 - eps)
    entropy = -(p_safe * np.log2(p_safe) + (1 - p_safe) * np.log2(1 - p_safe))

    df_unlabeled = pd.DataFrame({
        'txId': unlabeled_txId,
        'time_step': unlabeled_steps,
        'predicted_prob_illicit': np.round(unlabeled_probs, 4),
        'predicted_class': pred_class,
        'predicted_label': pred_label_str,
        'risk_score_100': risk_score,
        'risk_tier': risk_tier,
        'confidence_score': np.round(confidence, 4),
        'entropy_uncertainty': np.round(entropy, 4)
    })

    print(f"\nCompleted Risk Profiling for {len(df_unlabeled):,} Unlabeled Bitcoin Transactions:")
    print(f" -> High Risk Transactions (P >= 0.70):   {(unlabeled_probs >= 0.70).sum():,} ({(unlabeled_probs >= 0.70).sum()/len(df_unlabeled)*100:.2f}%)")
    print(f" -> Medium Risk Transactions (0.30-0.70): {((unlabeled_probs >= 0.30) & (unlabeled_probs < 0.70)).sum():,} ({((unlabeled_probs >= 0.30) & (unlabeled_probs < 0.70)).sum()/len(df_unlabeled)*100:.2f}%)")
    print(f" -> Low Risk Transactions (P < 0.30):    {(unlabeled_probs < 0.30).sum():,} ({(unlabeled_probs < 0.30).sum()/len(df_unlabeled)*100:.2f}%)")

    # Temporal breakdown summary
    step_summary = df_unlabeled.groupby('time_step').agg(
        total_unlabeled=('txId', 'count'),
        predicted_illicit=('predicted_class', 'sum'),
        avg_illicit_prob=('predicted_prob_illicit', 'mean'),
        avg_confidence=('confidence_score', 'mean'),
        high_risk_count=('risk_tier', lambda x: (x == 'High Risk (>=70%)').sum())
    ).reset_index()
    step_summary['predicted_illicit_ratio'] = step_summary['predicted_illicit'] / step_summary['total_unlabeled']

    # Export to Excel (Top 50,000 for quick Excel opening + full summary)
    xlsx_v1 = os.path.join(output_dir, 'unlabeled_node_predictions.xlsx')
    xlsx_v2 = os.path.join(output_dir, 'unlabeled_node_predictions_v2.xlsx')

    print(f"\nSaving Excel prediction files...")
    with pd.ExcelWriter(xlsx_v1, engine='openpyxl') as writer:
        df_unlabeled.head(50000).to_excel(writer, sheet_name='Top50k_Predictions', index=False)
        step_summary.to_excel(writer, sheet_name='Temporal_Risk_Summary', index=False)
    print(f" -> Saved: {xlsx_v1}")

    # Top high-risk transactions sorted
    df_high_risk = df_unlabeled.sort_values(by='predicted_prob_illicit', ascending=False)
    with pd.ExcelWriter(xlsx_v2, engine='openpyxl') as writer:
        df_high_risk.head(50000).to_excel(writer, sheet_name='Highest_Risk_Transactions', index=False)
        step_summary.to_excel(writer, sheet_name='Temporal_Risk_Summary', index=False)
    print(f" -> Saved: {xlsx_v2}")

    # Save full prediction CSV for fast data pipelines
    csv_path = os.path.join(output_dir, 'unlabeled_node_predictions_full.csv')
    df_unlabeled.to_csv(csv_path, index=False)
    print(f" -> Saved full CSV ({len(df_unlabeled):,} rows): {csv_path}")

    return df_unlabeled, step_summary

if __name__ == '__main__':
    run_unlabeled_inference()
