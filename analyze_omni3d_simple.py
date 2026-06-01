#!/usr/bin/env python3
"""
Omni3D 3D Annotation Analysis - Simplified Version
Analyzes Omni3D_pl-sam3d files which contain actual 3D annotations.
"""

import json
import numpy as np
from pathlib import Path

# Files that actually have annotations
FILES_TO_ANALYZE = [
    ("Omni3D_pl-sam3d/SUNRGBD_train", "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl-sam3d/SUNRGBD_train.json"),
    ("Omni3D_pl-sam3d/SUNRGBD_val", "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl-sam3d/SUNRGBD_val.json"),
]

OUTPUT_FILE = "/data/ZhaoX/OVM3D-Dett/omni3d_3d_analysis_report.txt"

def load_json_simple(path, max_items=None):
    """Load JSON file with optional limit."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def analyze_annotations(annotations, name):
    """Analyze 3D annotations and compute statistics."""
    results = {
        "name": name,
        "total": len(annotations),
        "valid3D": 0,
        "invalid3D": 0,
    }
    
    # Arrays for statistics
    center_x, center_y, center_z = [], [], []
    dim_l, dim_h, dim_w = [], [], []
    scores = []
    
    # Anomalies
    anomalies = []
    
    for i, ann in enumerate(annotations):
        valid = ann.get("valid3D", False)
        
        if valid:
            results["valid3D"] += 1
            
            # Extract center
            if "center_cam" in ann and ann["center_cam"]:
                c = ann["center_cam"]
                if isinstance(c, list) and len(c) == 3 and c[0] >= 0:  # Valid center
                    center_x.append(c[0])
                    center_y.append(c[1])
                    center_z.append(c[2])
                    
                    # Check for anomalies
                    dist = np.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
                    if dist > 15:
                        anomalies.append({
                            "id": ann.get("id"),
                            "cat": ann.get("category_name"),
                            "issue": f"Center distance {dist:.2f}m too far",
                            "center": c
                        })
            
            # Extract dimensions
            if "dimensions" in ann and ann["dimensions"]:
                d = ann["dimensions"]
                if isinstance(d, list) and len(d) == 3:
                    dim_l.append(d[0])
                    dim_h.append(d[1])
                    dim_w.append(d[2])
                    
                    # Check for anomalies
                    if d[0] > 10 or d[1] > 10 or d[2] > 10:
                        anomalies.append({
                            "id": ann.get("id"),
                            "cat": ann.get("category_name"),
                            "issue": f"Dimension exceeds 10m: {d}",
                            "dimensions": d
                        })
                    if d[0] < 0.01 or d[1] < 0.01 or d[2] < 0.01:
                        anomalies.append({
                            "id": ann.get("id"),
                            "cat": ann.get("category_name"),
                            "issue": f"Dimension too small: {d}",
                            "dimensions": d
                        })
        else:
            results["invalid3D"] += 1
            
        if "score" in ann and ann["score"] is not None:
            scores.append(ann["score"])
    
    # Compute statistics
    def stats(arr, name):
        if not arr:
            return {"count": 0}
        arr = np.array(arr)
        return {
            "count": len(arr),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
        }
    
    results["center_x"] = stats(center_x, "x")
    results["center_y"] = stats(center_y, "y")
    results["center_z"] = stats(center_z, "z")
    results["dim_l"] = stats(dim_l, "l")
    results["dim_h"] = stats(dim_h, "h")
    results["dim_w"] = stats(dim_w, "w")
    results["scores"] = stats(scores, "score")
    results["anomalies"] = anomalies[:50]  # Limit to first 50
    
    # Category counts
    cat_counts = {}
    for ann in annotations:
        cat = ann.get("category_name", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    results["categories"] = dict(sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    return results

def main():
    print("=" * 60)
    print("Omni3D 3D Annotation Analysis")
    print("=" * 60)
    
    all_results = {}
    
    for name, path in FILES_TO_ANALYZE:
        print(f"\nAnalyzing: {name}")
        print("-" * 40)
        
        data = load_json_simple(path)
        annotations = data.get("annotations", [])
        
        print(f"  Total annotations: {len(annotations)}")
        
        if annotations:
            results = analyze_annotations(annotations, name)
            all_results[name] = results
            
            print(f"  Valid3D: {results['valid3D']}")
            print(f"  Invalid3D: {results['invalid3D']}")
            print(f"  Anomalies found: {len(results['anomalies'])}")
        else:
            print("  No annotations found in this file!")
            all_results[name] = {"error": "No annotations"}
    
    # Generate report
    print("\n" + "=" * 60)
    print("Generating Report...")
    print("=" * 60)
    
    lines = []
    lines.append("=" * 80)
    lines.append("OMNI3D 3D ANNOTATION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary table
    lines.append("FILE SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{'File':<40} {'Total':<12} {'Valid3D':<12} {'Invalid3D':<12} {'Anomalies':<10}")
    lines.append("-" * 80)
    
    for name, r in all_results.items():
        if "error" in r:
            lines.append(f"{name:<40} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<10}")
        else:
            lines.append(f"{name:<40} {r['total']:<12} {r['valid3D']:<12} "
                        f"{r['invalid3D']:<12} {len(r['anomalies']):<10}")
    
    lines.append("")
    
    # Detailed stats for each file
    for name, r in all_results.items():
        if "error" in r:
            continue
            
        lines.append("=" * 80)
        lines.append(f"DETAILED STATISTICS: {name}")
        lines.append("=" * 80)
        
        # Center stats
        lines.append("\n[Camera Center (center_cam)]")
        for coord, key in [("X", "center_x"), ("Y", "center_y"), ("Z", "center_z")]:
            s = r.get(key, {})
            lines.append(f"  {coord}: count={s.get('count', 0)}, "
                        f"range=[{s.get('min', 'N/A'):.4f}, {s.get('max', 'N/A'):.4f}], "
                        f"mean={s.get('mean', 'N/A'):.4f}, std={s.get('std', 'N/A'):.4f}")
        
        # Dimension stats
        lines.append("\n[Object Dimensions (meters)]")
        for dim, key in [("L (length)", "dim_l"), ("H (height)", "dim_h"), ("W (width)", "dim_w")]:
            s = r.get(key, {})
            lines.append(f"  {dim}: count={s.get('count', 0)}, "
                        f"range=[{s.get('min', 'N/A'):.4f}, {s.get('max', 'N/A'):.4f}]m, "
                        f"mean={s.get('mean', 'N/A'):.4f}m")
        
        # Score stats
        if r.get("scores", {}).get("count", 0) > 0:
            s = r["scores"]
            lines.append(f"\n[Detection Scores]")
            lines.append(f"  Range: [{s.get('min', 'N/A'):.4f}, {s.get('max', 'N/A'):.4f}]")
            lines.append(f"  Mean: {s.get('mean', 'N/A'):.4f}, Median: {s.get('median', 'N/A'):.4f}")
        
        # Categories
        lines.append("\n[Top Categories]")
        for cat, count in list(r.get("categories", {}).items())[:10]:
            pct = count / r["total"] * 100
            lines.append(f"  {cat}: {count} ({pct:.1f}%)")
        
        # Anomalies
        lines.append(f"\n[Anomalies ({len(r['anomalies'])} shown of total found)]")
        for a in r["anomalies"][:10]:
            lines.append(f"  ID={a['id']}, Cat={a['cat']}")
            lines.append(f"    Issue: {a['issue']}")
        
        lines.append("")
    
    # Comparison
    if len(all_results) >= 2:
        lines.append("=" * 80)
        lines.append("COMPARISON: Train vs Val")
        lines.append("=" * 80)
        
        train_r = all_results.get("Omni3D_pl-sam3d/SUNRGBD_train", {})
        val_r = all_results.get("Omni3D_pl-sam3d/SUNRGBD_val", {})
        
        if train_r and val_r and "error" not in train_r and "error" not in val_r:
            lines.append("\n[Metric Comparison]")
            lines.append(f"{'Metric':<30} {'Train':<20} {'Val':<20} {'Diff':<15}")
            lines.append("-" * 80)
            
            # Valid3D ratio
            train_ratio = train_r['valid3D'] / train_r['total'] * 100 if train_r['total'] > 0 else 0
            val_ratio = val_r['valid3D'] / val_r['total'] * 100 if val_r['total'] > 0 else 0
            lines.append(f"{'Valid3D Ratio (%)':<30} {train_ratio:<20.2f} {val_ratio:<20.2f} {val_ratio - train_ratio:<15.2f}")
            
            # Mean dimensions
            train_h = train_r.get('dim_h', {}).get('mean', 0)
            val_h = val_r.get('dim_h', {}).get('mean', 0)
            lines.append(f"{'Mean Height (m)':<30} {train_h:<20.4f} {val_h:<20.4f} {val_h - train_h:<15.4f}")
            
            train_w = train_r.get('dim_w', {}).get('mean', 0)
            val_w = val_r.get('dim_w', {}).get('mean', 0)
            lines.append(f"{'Mean Width (m)':<30} {train_w:<20.4f} {val_w:<20.4f} {val_w - train_w:<15.4f}")
            
            train_l = train_r.get('dim_l', {}).get('mean', 0)
            val_l = val_r.get('dim_l', {}).get('mean', 0)
            lines.append(f"{'Mean Length (m)':<30} {train_l:<20.4f} {val_l:<20.4f} {val_l - train_l:<15.4f}")
            
            # Mean center Z (depth)
            train_z = train_r.get('center_z', {}).get('mean', 0)
            val_z = val_r.get('center_z', {}).get('mean', 0)
            lines.append(f"{'Mean Depth Z (m)':<30} {train_z:<20.4f} {val_z:<20.4f} {val_z - train_z:<15.4f}")
            
            # Mean score
            train_s = train_r.get('scores', {}).get('mean', 0)
            val_s = val_r.get('scores', {}).get('mean', 0)
            lines.append(f"{'Mean Score':<30} {train_s:<20.4f} {val_s:<20.4f} {val_s - train_s:<15.4f}")
            
            # Anomalies
            lines.append(f"{'Anomaly Count':<30} {len(train_r.get('anomalies', [])):<20} "
                        f"{len(val_r.get('anomalies', [])):<20} "
                        f"{len(val_r.get('anomalies', [])) - len(train_r.get('anomalies', [])):<15}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    # Save report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Print summary to console
    print(report)

if __name__ == "__main__":
    main()
