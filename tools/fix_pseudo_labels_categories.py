#!/usr/bin/env python3
"""
修复伪标签中的类别映射问题

问题: 伪标签生成了复合类别（如 "bottle cup", "desk counter"），但 Omni3D 训练器期望标准类别
解决: 将复合类别映射到最匹配的标准类别
"""

import os
import sys
import torch
import re
from collections import defaultdict

sys.path.insert(0, '/data/ZhaoX/OVM3D-Dett')

from cubercnn.generate_label.llm_generated_prior import SUNRGBD


# 类别映射表：将复合类别映射到标准类别
CATEGORY_MAPPING = {
    # 复合类别
    'bottle cup': 'cup',
    'bottle glass': 'glass',
    'cup glass': 'cup',
    'desk counter': 'counter',
    'desk night stand': 'desk',
    'desk table': 'table',
    'desk counter night stand': 'desk',
    'night stand bin': 'night stand',
    'night bin': 'night stand',
    'cabinet bookcase': 'cabinet',
    'cabinet door': 'door',
    'cabinetcase': 'cabinet',
    'cabinet counter': 'counter',
    'cabinet desk night stand': 'desk',
    'cabinet night stand': 'night stand',
    'cabinet stand': 'cabinet',
    'shelves cabinet': 'shelves',
    'shelves cabinet bookcase': 'shelves',
    'shelves cabinetcase': 'shelves',
    'shelves cabinet door': 'shelves',
    'shelves bookcase': 'shelves',
    'shelves cabinet counter': 'shelves',
    'blinds curtain': 'blinds',
    'blinds window': 'window',
    'blinds door': 'door',
    'blinds lamp': 'lamp',
    'blinds picture': 'picture',
    'window curtain': 'window',
    'window mirror': 'mirror',
    'window picture': 'picture',
    'window television': 'television',
    'window picture television': 'television',
    'window mirror picture': 'mirror',
    'chair sofa': 'chair',
    'chair toilet': 'toilet',
    'bed sofa': 'bed',
    'box bin': 'bin',
    'books box': 'books',
    'books towel': 'books',
    'laptop television': 'television',
    'laptop television oven': 'television',
    'refrigerator oven': 'refrigerator',
    'door refrigerator': 'refrigerator',
    'door oven': 'oven',
    'desk stand': 'desk',
    'desk night': 'desk',
    'towel clothes': 'towel',
    'pillow curtain': 'pillow',
    'curtain towel': 'curtain',
    'blinds door curtain': 'blinds',
    'desk floor mat': 'floor mat',
    'desk counter floor': 'counter',
    'desk picture': 'picture',
    'cup picture': 'picture',
    'sink counter': 'sink',
    'cabinetcase night stand': 'night stand',
    'television oven': 'television',
    'floor': 'floor mat',  # 添加 floor 映射
}


def normalize_category(phrase):
    """
    将类别名称标准化:
    1. 转换为小写
    2. 移除多余空格
    3. 映射到标准类别
    """
    # 转小写
    phrase = phrase.lower().strip()
    
    # 替换下划线为空格
    phrase = phrase.replace('_', ' ')
    
    # 移除多余空格
    phrase = re.sub(r'\s+', ' ', phrase)
    
    return phrase


def get_standard_category(phrase, all_standard_categories):
    """
    获取标准类别:
    1. 先尝试精确映射
    2. 再尝试包含匹配
    3. 最后返回最接近的类别
    """
    phrase = normalize_category(phrase)
    
    # 1. 检查精确映射
    if phrase in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[phrase]
    
    # 2. 检查是否已经是标准类别
    if phrase in all_standard_categories:
        return phrase
    
    # 3. 尝试从复合类别中提取主类别
    words = phrase.split()
    
    # 按优先级匹配
    for word in words:
        if word in all_standard_categories:
            return word
    
    # 4. 尝试反向匹配（标准类别是否包含在短语中）
    for std_cat in all_standard_categories:
        if std_cat in phrase:
            return std_cat
    
    # 5. 返回第一个单词（作为主类别）
    return words[0] if words else phrase


def fix_pseudo_labels(input_path, output_path=None):
    """
    修复伪标签中的类别映射
    
    Args:
        input_path: 原始 info_3d.pth 路径
        output_path: 修复后的保存路径，None 则覆盖原文件
    """
    if output_path is None:
        output_path = input_path
    
    print("=" * 60)
    print("修复伪标签类别映射")
    print("=" * 60)
    
    # 加载数据
    print(f"加载: {input_path}")
    data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # 获取所有标准类别
    all_standard_categories = set(SUNRGBD.keys())
    print(f"标准类别数: {len(all_standard_categories)}")
    
    # 统计原始类别
    original_categories = defaultdict(int)
    fixed_categories = defaultdict(int)
    unmapped_count = 0
    
    # 修复每个图像的结果
    for img_id, result in data.items():
        new_phrases = []
        
        for phrase in result['phrases']:
            original_categories[phrase] += 1
            standard_phrase = get_standard_category(phrase, all_standard_categories)
            new_phrases.append(standard_phrase)
            fixed_categories[standard_phrase] += 1
            
            if standard_phrase != phrase.lower().strip().replace('_', ' '):
                unmapped_count += 1
        
        result['phrases'] = new_phrases
    
    # 保存修复后的数据
    print(f"保存: {output_path}")
    torch.save(data, output_path)
    
    # 打印统计
    print(f"\n" + "=" * 60)
    print("修复统计")
    print("=" * 60)
    print(f"修复的类别数: {unmapped_count}")
    
    print(f"\n原始类别分布 (前15):")
    for cat, count in sorted(original_categories.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cat}: {count}")
    
    print(f"\n修复后类别分布 (前15):")
    for cat, count in sorted(fixed_categories.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cat}: {count}")
    
    # 检查是否在标准类别中
    fixed_cats = set(fixed_categories.keys())
    unknown_cats = fixed_cats - all_standard_categories
    
    if unknown_cats:
        print(f"\n警告: 仍有 {len(unknown_cats)} 个未知类别:")
        for cat in sorted(unknown_cats):
            print(f"  {cat}: {fixed_categories[cat]}")
    else:
        print(f"\n✅ 所有类别都已映射到标准类别!")
    
    return data


if __name__ == '__main__':
    # 修复伪标签
    input_path = '/extra/ZhaoX_pseudo_label_gsam/info_3d.pth'
    # 保存到本地目录
    output_path = '/data/ZhaoX/OVM3D-Dett/pseudo_labels_fixed.pth'
    
    data = fix_pseudo_labels(input_path, output_path)
    
    print(f"\n修复完成!")
    print(f"结果保存到: {output_path}")
    print(f"\n使用方法:")
    print(f"  cp {output_path} /extra/ZhaoX_pseudo_label_gsam/info_3d.pth")
