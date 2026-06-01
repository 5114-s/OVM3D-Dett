#!/usr/bin/env python3
"""
分析 Omni3D 伪标签 JSON 文件的 3D 数值结构
"""

import json
import numpy as np
from collections import defaultdict

def analyze_annotations(file_path, max_samples=200):
    """分块读取并分析 annotations"""
    print(f"\n{'='*60}")
    print(f"分析文件: {file_path}")
    print(f"{'='*60}")
    
    center_cam_x, center_cam_y, center_cam_z = [], [], []
    dim_depth, dim_width, dim_height = [], [], []
    categories = defaultdict(int)
    anomalies = []
    valid_count = 0
    
    with open(file_path, 'r') as f:
        # 读取整个文件但只处理 annotations 部分
        data = json.load(f)
        
        annotations = data.get('annotations', [])
        info = data.get('info', {})
        images = data.get('images', [])
        
        print(f"基本信息:")
        print(f"  - info: {list(info.keys()) if info else 'None'}")
        print(f"  - images 数量: {len(images)}")
        print(f"  - annotations 总量: {len(annotations)}")
        print()
        
        # 只分析前 max_samples 个
        sample_annotations = annotations[:max_samples]
        print(f"分析前 {len(sample_annotations)} 个样本:")
        
        for i, ann in enumerate(sample_annotations):
            valid_count += 1
            
            # 提取 center_cam
            cc = ann.get('center_cam', [])
            if len(cc) == 3:
                center_cam_x.append(cc[0])
                center_cam_y.append(cc[1])
                center_cam_z.append(cc[2])
                
                # 检测异常值
                for axis, val in zip(['X', 'Y', 'Z'], cc):
                    if not (-20 <= val <= 20):
                        anomalies.append({
                            'id': ann.get('id'),
                            'field': f'center_cam.{axis}',
                            'value': val
                        })
            
            # 提取 dimensions
            dims = ann.get('dimensions', [])
            if len(dims) == 3:
                dim_depth.append(dims[0])
                dim_width.append(dims[1])
                dim_height.append(dims[2])
                
                # 检测异常值
                for axis, val in zip(['depth', 'width', 'height'], dims):
                    if not (0.01 <= val <= 10):
                        anomalies.append({
                            'id': ann.get('id'),
                            'field': f'dimensions.{axis}',
                            'value': val
                        })
            
            # 统计类别
            cat = ann.get('category_name', 'unknown')
            categories[cat] += 1
        
        # 转换为 numpy 数组
        cc_x = np.array(center_cam_x)
        cc_y = np.array(center_cam_y)
        cc_z = np.array(center_cam_z)
        dd = np.array(dim_depth)
        dw = np.array(dim_width)
        dh = np.array(dim_height)
        
        print(f"\n--- center_cam (3D 中心坐标) ---")
        print(f"  X 范围: [{cc_x.min():.4f}, {cc_x.max():.4f}], 均值: {cc_x.mean():.4f}, 标准差: {cc_x.std():.4f}")
        print(f"  Y 范围: [{cc_y.min():.4f}, {cc_y.max():.4f}], 均值: {cc_y.mean():.4f}, 标准差: {cc_y.std():.4f}")
        print(f"  Z 范围: [{cc_z.min():.4f}, {cc_z.max():.4f}], 均值: {cc_z.mean():.4f}, 标准差: {cc_z.std():.4f}")
        
        print(f"\n--- dimensions (depth/width/height) ---")
        print(f"  Depth  范围: [{dd.min():.4f}, {dd.max():.4f}], 均值: {dd.mean():.4f}, 标准差: {dd.std():.4f}")
        print(f"  Width  范围: [{dw.min():.4f}, {dw.max():.4f}], 均值: {dw.mean():.4f}, 标准差: {dw.std():.4f}")
        print(f"  Height 范围: [{dh.min():.4f}, {dh.max():.4f}], 均值: {dh.mean():.4f}, 标准差: {dh.std():.4f}")
        
        print(f"\n--- 类别分布 (Top 10) ---")
        sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
        for cat, count in sorted_cats[:10]:
            print(f"  {cat}: {count}")
        
        print(f"\n--- 异常值检测 ---")
        print(f"  坐标超出 [-20, 20] 米 或 尺寸超出 [0.01, 10] 米:")
        if anomalies:
            for a in anomalies[:20]:  # 只显示前20个
                print(f"    ID={a['id']}, {a['field']}={a['value']:.4f}")
            if len(anomalies) > 20:
                print(f"    ... 还有 {len(anomalies) - 20} 个异常值")
        else:
            print(f"  无异常值")
        
        return {
            'center_cam': {
                'x': (cc_x.min(), cc_x.max(), cc_x.mean(), cc_x.std()),
                'y': (cc_y.min(), cc_y.max(), cc_y.mean(), cc_y.std()),
                'z': (cc_z.min(), cc_z.max(), cc_z.mean(), cc_z.std()),
            },
            'dimensions': {
                'depth': (dd.min(), dd.max(), dd.mean(), dd.std()),
                'width': (dw.min(), dw.max(), dw.mean(), dw.std()),
                'height': (dh.min(), dh.max(), dh.mean(), dh.std()),
            },
            'anomaly_count': len(anomalies),
            'total_samples': valid_count,
            'categories': dict(sorted_cats)
        }

def compare_two_versions(stats1, stats2, name1, name2):
    """对比两个版本的统计差异"""
    print(f"\n{'='*60}")
    print(f"两个版本对比: {name1} vs {name2}")
    print(f"{'='*60}")
    
    print("\n--- center_cam 均值对比 ---")
    for axis in ['x', 'y', 'z']:
        v1 = stats1['center_cam'][axis][2]  # 均值
        v2 = stats2['center_cam'][axis][2]
        diff = abs(v1 - v2)
        print(f"  {axis.upper()}: {name1}={v1:.4f}, {name2}={v2:.4f}, 差异={diff:.4f}")
    
    print("\n--- dimensions 均值对比 ---")
    for axis in ['depth', 'width', 'height']:
        v1 = stats1['dimensions'][axis][2]
        v2 = stats2['dimensions'][axis][2]
        diff = abs(v1 - v2)
        print(f"  {axis}: {name1}={v1:.4f}, {name2}={v2:.4f}, 差异={diff:.4f}")
    
    print(f"\n--- 异常值统计 ---")
    print(f"  {name1}: {stats1['anomaly_count']} 个异常")
    print(f"  {name2}: {stats2['anomaly_count']} 个异常")
    
    # 类别差异
    print(f"\n--- 类别数量 ---")
    cats1 = set(stats1['categories'].keys())
    cats2 = set(stats2['categories'].keys())
    print(f"  {name1} 独有类别: {cats1 - cats2}")
    print(f"  {name2} 独有类别: {cats2 - cats1}")

def main():
    file1 = "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl/SUNRGBD_train.json"
    file2 = "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl-sam3d/SUNRGBD_train.json"
    output_file = "/data/ZhaoX/OVM3D-Dett/pseudo_label_comparison.txt"
    
    # 分析两个文件
    stats1 = analyze_annotations(file1, max_samples=200)
    stats2 = analyze_annotations(file2, max_samples=200)
    
    # 对比
    compare_two_versions(stats1, stats2, "Omni3D_pl", "Omni3D_pl-sam3d")
    
    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        import sys
        from io import StringIO
        
        # 捕获 print 输出
        output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = output
        
        print("="*60)
        print("Omni3D 伪标签 3D 数值结构分析报告")
        print("="*60)
        
        sys.stdout = old_stdout
        
        output.write(output.getvalue())
        
        # 写入两个文件的基本信息
        output.write(f"\n文件1: {file1}\n")
        output.write(f"文件2: {file2}\n")
        output.write(f"分析样本数: 200\n\n")
        
        for name, stats in [("Omni3D_pl", stats1), ("Omni3D_pl-sam3d", stats2)]:
            output.write(f"\n{'='*50}\n")
            output.write(f"文件: {name}\n")
            output.write(f"{'='*50}\n")
            output.write(f"分析样本数: {stats['total_samples']}\n")
            output.write(f"异常值数量: {stats['anomaly_count']}\n\n")
            
            output.write("center_cam 统计:\n")
            for axis in ['x', 'y', 'z']:
                min_v, max_v, mean_v, std_v = stats['center_cam'][axis]
                output.write(f"  {axis.upper()}: [{min_v:.4f}, {max_v:.4f}], mean={mean_v:.4f}, std={std_v:.4f}\n")
            
            output.write("\ndimensions 统计:\n")
            for axis in ['depth', 'width', 'height']:
                min_v, max_v, mean_v, std_v = stats['dimensions'][axis]
                output.write(f"  {axis}: [{min_v:.4f}, {max_v:.4f}], mean={mean_v:.4f}, std={std_v:.4f}\n")
            
            output.write("\n类别分布:\n")
            for cat, count in stats['categories'].items():
                output.write(f"  {cat}: {count}\n")
        
        # 对比部分
        output.write(f"\n{'='*50}\n")
        output.write("两个版本对比\n")
        output.write(f"{'='*50}\n\n")
        
        output.write("center_cam 均值对比:\n")
        for axis in ['x', 'y', 'z']:
            v1 = stats1['center_cam'][axis][2]
            v2 = stats2['center_cam'][axis][2]
            output.write(f"  {axis.upper()}: Omni3D_pl={v1:.4f}, Omni3D_pl-sam3d={v2:.4f}, diff={abs(v1-v2):.4f}\n")
        
        output.write("\ndimensions 均值对比:\n")
        for axis in ['depth', 'width', 'height']:
            v1 = stats1['dimensions'][axis][2]
            v2 = stats2['dimensions'][axis][2]
            output.write(f"  {axis}: Omni3D_pl={v1:.4f}, Omni3D_pl-sam3d={v2:.4f}, diff={abs(v1-v2):.4f}\n")
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(output.getvalue())
    
    print(f"\n\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
