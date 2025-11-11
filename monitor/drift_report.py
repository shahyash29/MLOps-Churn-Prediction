import os, json, pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
import webbrowser

load_dotenv()

LOG_PATH = Path(os.getenv("PRED_LOG_DIR","logs/inference"))/"events.jsonl"
REF_PATH = Path(os.getenv("REF_DATA_PATH","data/processed/train.csv"))
OUT_DIR = Path(os.getenv("DRIFT_OUT_DIR","artifacts/monitoring"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load reference data
print("Loading reference data...")
ref = pd.read_csv(REF_PATH)
if "churn" in ref.columns: 
    ref = ref.drop(columns=["churn"])

# Load current data from logs
rows = []
if LOG_PATH.exists():
    with open(LOG_PATH, 'r') as f:
        for line in f:
            try: 
                rows.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass

# Create current data
if not rows:
    print(f"⚠️  No prediction logs found at: {LOG_PATH}")
    print("Creating sample data for demonstration...")
    
    # Create sample current data
    sample_size = min(100, len(ref))
    curr = ref.sample(n=sample_size, replace=True).reset_index(drop=True)
    
    # Add some noise to simulate drift
    import numpy as np
    for col in curr.select_dtypes(include=[np.number]).columns:
        if curr[col].std() > 0:
            curr[col] = curr[col] * np.random.uniform(0.9, 1.1, size=len(curr))
    
    print(f"✓ Created {len(curr)} sample records")
else:
    curr = pd.DataFrame([r.get("features", {}) for r in rows])
    print(f"✓ Loaded {len(curr)} prediction records")

# Align columns
common_cols = list(set(ref.columns) & set(curr.columns))
ref_aligned = ref[common_cols].copy()
curr_aligned = curr[common_cols].copy()

# Convert to numeric
for d in [ref_aligned, curr_aligned]:
    for c in ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]:
        if c in d.columns: 
            d[c] = pd.to_numeric(d[c], errors="coerce")

print(f"\nGenerating drift report...")
print(f"  Reference: {len(ref_aligned)} rows, {len(ref_aligned.columns)} columns")
print(f"  Current: {len(curr_aligned)} rows, {len(curr_aligned.columns)} columns")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_aligned, current_data=curr_aligned)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_html = OUT_DIR / f"drift_report_{ts}.html"

try:
    if hasattr(report, '_build_dashboard_info'):
        dashboard_info = report._build_dashboard_info()
        html_content = dashboard_info.get("dashboard", {}).get("html", "")
    else:
        html_content = report._repr_html_()
    
    if html_content:
        if not html_content.startswith("<!DOCTYPE"):
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Drift Report</title>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
        
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nReport saved: {out_html}")
        print(f"Opening in browser...")
        webbrowser.open(f"file://{out_html.absolute()}")
        
    else:
        raise Exception("No HTML content generated")
        
except Exception as e:
    print(f"Primary method failed: {e}")
    print("Trying alternative approach...")
    
    try:
        from IPython.display import HTML
        from IPython import get_ipython
        
        # Get HTML representation
        html_obj = report._repr_html_()
        
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(str(html_obj))
        
        print(f"Report saved: {out_html}")
        webbrowser.open(f"file://{out_html.absolute()}")
        
    except:
        print("Creating basic HTML report...")
        
        drift_detected = []
        no_drift = []
        
        for col in common_cols:
            if col in ref_aligned.columns and col in curr_aligned.columns:
                ref_mean = ref_aligned[col].mean() if pd.api.types.is_numeric_dtype(ref_aligned[col]) else 0
                curr_mean = curr_aligned[col].mean() if pd.api.types.is_numeric_dtype(curr_aligned[col]) else 0
                
                if abs(ref_mean - curr_mean) > (ref_aligned[col].std() * 0.5 if pd.api.types.is_numeric_dtype(ref_aligned[col]) else 0):
                    drift_detected.append(col)
                else:
                    no_drift.append(col)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Drift Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{ 
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
                .drift {{ 
                    background: #ffe5e5;
                    border-left-color: #e74c3c;
                }}
                .no-drift {{ 
                    background: #e5ffe5;
                    border-left-color: #27ae60;
                }}
                .feature-list {{
                    columns: 2;
                    column-gap: 20px;
                }}
                .feature-item {{
                    break-inside: avoid;
                    padding: 5px;
                    margin: 5px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Data Drift Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <div class="metric-card">
                        <div class="metric-value">{len(ref_aligned):,}</div>
                        <div class="metric-label">Reference Samples</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(curr_aligned):,}</div>
                        <div class="metric-label">Current Samples</div>
                    </div>
                    <div class="metric-card drift">
                        <div class="metric-value">{len(drift_detected)}</div>
                        <div class="metric-label">Features with Drift</div>
                    </div>
                    <div class="metric-card no-drift">
                        <div class="metric-value">{len(no_drift)}</div>
                        <div class="metric-label">Stable Features</div>
                    </div>
                </div>
                
                <h2>⚠️ Features with Potential Drift</h2>
                <div class="feature-list">
                    {"".join([f'<div class="feature-item">• {col}</div>' for col in drift_detected]) if drift_detected else '<p>No significant drift detected</p>'}
                </div>
                
                <h2> Stable Features</h2>
                <div class="feature-list">
                    {"".join([f'<div class="feature-item">• {col}</div>' for col in no_drift])}
                </div>
                
                <hr style="margin: 40px 0;">
                <p><em>Note: This is a simplified report. Install the latest Evidently version for full interactive visualizations.</em></p>
            </div>
        </body>
        </html>
        """
        
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nBasic HTML report created: {out_html}")
        print(f"Opening in browser...")
        webbrowser.open(f"file://{out_html.absolute()}")

print("\n" + "="*60)
print("Report generation complete!")
print("="*60)