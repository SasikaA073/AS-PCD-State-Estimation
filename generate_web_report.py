import os
import glob
import re

def parse_results_file(filepath):
    """Parses the KF_results.txt file to extract metrics."""
    metrics = {
        'meas_rmse': 'N/A',
        'kf_rmse': 'N/A',
        'improvement': 'N/A'
    }
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Simple regex search
            m_meas = re.search(r"Measurement RMSE:\s*([\d\.]+)", content)
            m_kf = re.search(r"KF Estimate RMSE:\s*([\d\.]+)", content)
            m_imp = re.search(r"Improvement:\s*([\-\d\.]+)", content)
            
            if m_meas: metrics['meas_rmse'] = m_meas.group(1)
            if m_kf: metrics['kf_rmse'] = m_kf.group(1)
            if m_imp: metrics['improvement'] = m_imp.group(1)
            
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return metrics

def generate_html_report():
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "output")
    docs_dir = os.path.join(project_root, "docs")
    
    # Ensure docs directory exists
    os.makedirs(docs_dir, exist_ok=True)
    
    # Find all result text files
    result_files = glob.glob(os.path.join(output_dir, "*_KF_results.txt"))
    
    data_items = []
    
    for txt_path in result_files:
        filename = os.path.basename(txt_path)
        # Expected format: {index}_{scene_id}_{sequence_name}_KF_results.txt
        # Helper regex to capture parts
        match = re.match(r"(\d+)_(.+?)_(sequence\d+)_KF_results\.txt", filename)
        
        if not match:
            print(f"Skipping file with unexpected name format: {filename}")
            continue
            
        index = int(match.group(1))
        scene_id = match.group(2)
        sequence_name = match.group(3)
        base_prefix = f"{index}_{scene_id}_{sequence_name}"
        
        # Paths for other assets (relative to docs/index.html)
        # Output dir is "../output"
        gif_path = f"../output/{base_prefix}_preview_top_view.gif"
        plot_path = f"../output/{base_prefix}_KF_tracking.png"
        
        # Parse metrics
        metrics = parse_results_file(txt_path)
        
        data_items.append({
            'index': index,
            'scene': scene_id,
            'sequence': sequence_name,
            'gif': gif_path,
            'plot': plot_path,
            'metrics': metrics
        })
        
    # Sort by index
    data_items.sort(key=lambda x: x['index'])
    
    if not data_items:
        print("No result files found in output directory.")
        return

    # Calculate Summary Stats
    improvements = [float(d['metrics']['improvement']) for d in data_items if d['metrics']['improvement'] != 'N/A']
    avg_imp = sum(improvements) / len(improvements) if improvements else 0.0
    processed_count = len(data_items)

    # HTML Template with Swiss Design
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Systems - State Estimation Filter</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@400;500;700&display=swap" rel="stylesheet">
    
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #ffffff;
            color: #000000;
            line-height: 1.4;
        }}

        header {{
            padding: 60px 40px 40px;
            border-bottom: 2px solid #000000;
            margin-bottom: 0;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }}

        .subtitle {{
            font-size: 0.875rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #666666;
        }}

        .stats {{
            display: flex;
            gap: 60px;
            margin-top: 30px;
            font-size: 0.875rem;
        }}

        .stat-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .stat-label {{
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666666;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 0;
        }}

        .grid-row {{
            display: grid;
            grid-template-columns: 200px 1fr 1fr 300px;
            border-bottom: 1px solid #000000;
            align-items: center;
            min-height: 280px;
        }}

        .grid-row:hover {{
            background-color: #f5f5f5;
        }}

        .cell {{
            padding: 30px;
            border-right: 1px solid #000000;
        }}

        .cell:last-child {{
            border-right: none;
        }}

        .sequence-info {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .index {{
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }}

        .seq-name {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666666;
        }}

        .scene-name {{
            font-size: 0.75rem;
            color: #999999;
        }}

        .media-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }}

        .media-container img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}

        .metrics {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .metric-row {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 20px;
            font-size: 0.875rem;
        }}

        .metric-label {{
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.75rem;
            color: #666666;
        }}

        .metric-value {{
            font-weight: 700;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .improvement {{
            color: #000000;
            font-size: 1rem;
        }}

        @media (max-width: 1400px) {{
            .grid-row {{
                grid-template-columns: 150px 1fr 1fr 250px;
            }}
        }}

        @media (max-width: 1024px) {{
            .grid-row {{
                grid-template-columns: 1fr;
                grid-template-rows: auto auto auto auto;
            }}
            
            .cell {{
                border-right: none;
                border-bottom: 1px solid #cccccc;
            }}
            
            .cell:last-child {{
                border-bottom: none;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="subtitle">Project Report <small> by Sasika Amarasinghe</small></div>
        <h1>Autonomous Systems - State Estimation Filter</h1>
        <div class="stats">
            <div class="stat-item">
                <span class="stat-value">{processed_count}</span>
                <span class="stat-label">Sequences</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{avg_imp:.4f} m</span>
                <span class="stat-label">Avg. Improvement</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">Human Scene Synthetic PCD Dataset</span>
                <span class="stat-label">Dataset</span>
            </div>
        </div>
    </header>
    
    <div class="container">
"""
    
    for item in data_items:
        m = item['metrics']
        index_str = str(item['index']).zfill(2)
        
        html += f"""        <div class="grid-row">
            <div class="cell sequence-info">
                <div class="index">{index_str}</div>
                <div class="seq-name">{item['sequence']}</div>
                <div class="scene-name">{item['scene']}</div>
            </div>
            <div class="cell media-container">
                <img src="{item['gif']}" alt="Animation">
            </div>
            <div class="cell media-container">
                <img src="{item['plot']}" alt="Tracking">
            </div>
            <div class="cell metrics">
                <div class="metric-row">
                    <span class="metric-label">Measurement RMSE</span>
                    <span class="metric-value">{m['meas_rmse']} m</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Kalman Filter RMSE</span>
                    <span class="metric-value">{m['kf_rmse']} m</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Improvement</span>
                    <span class="metric-value improvement">{m['improvement']} m</span>
                </div>
            </div>
        </div>
"""

    html += """    </div>
</body>
</html>
"""
    
    html_path = os.path.join(docs_dir, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
        
    print(f"Report generated successfully: {html_path}")


if __name__ == "__main__":
    generate_html_report()