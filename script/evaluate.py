#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集评估脚本：从jsonl文件中读取测试结果，根据audios路径划分数据集，
评估检测和定位结果，计算三分类各项指标并保存为Excel格式

输出格式：行为数据集，列为评估指标
"""

import json
import re
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def normalize_label(label: Optional[str]) -> str:
    """
    标准化标签：统一小写、空格，并映射常见变体到标准类别
    
    标准类别: 'fully real', 'fully fake', 'partially fake'
    兼容数值标签: 0=fully real, 1=fully fake, 2=partially fake
    """
    if label is None or not isinstance(label, str):
        return 'unknown'

    s = ' '.join(label.strip().lower().split()).strip(' "\'`*:_-')
    m = re.search(r'\b([012])\b', s)
    if m:
        s = m.group(1)

    mapping = {
        '0': 'fully real',
        '1': 'fully fake',
        '2': 'partially fake',
    }
    return mapping.get(s, s)


def parse_dataset_from_audio_path(audio_path: str) -> str:
    """
    从音频路径中解析数据集名称
    
    Args:
        audio_path: 音频文件路径，例如 "datasets/PartialSpoof/ADD2023/..."
    
    Returns:
        数据集名称，例如 "ADD2023"
    
    规则：
        - ASVspoof2019LA 数据集划分到 PartialSpoof
        - 其他数据集从路径中提取对应的名称
    """
    # 将路径标准化
    path = audio_path.strip()
    
    # ASVspoof2019LA 划分到 PartialSpoof
    if 'ASVspoof2019LA' in path:
        return 'PartialSpoof'
    
    # 解析 PartialSpoof 下的子数据集
    # 路径格式：datasets/PartialSpoof/<数据集名>/...
    if '/PartialSpoof/' in path:
        parts = path.split('/PartialSpoof/')
        if len(parts) > 1:
            # 提取子数据集名称（路径的下一级目录）
            sub_path = parts[1].strip('/').split('/')[0] if parts[1] else ''
            if sub_path:
                # 处理一些特殊情况
                if sub_path in ['PartialSpoof', 'partialspoof']:
                    return 'PartialSpoof'
                elif sub_path in ['LAV-DF', 'lav-df']:
                    return 'LAV-DF'
                elif sub_path in ['AV-Deepfake1M-PlusPlus', 'av-deepfake1m-plusplus']:
                    return 'AV-Deepfake1M-PlusPlus'
                elif sub_path in ['Speech-Forensics', 'speech-forensics']:
                    return 'Speech-Forensics'
                elif sub_path in ['SINE_v2', 'SINE-v2', 'sine_v2', 'sine-v2']:
                    return 'SINE_v2'
                elif sub_path in ['PhonemeFakeV2', 'phonemefakev2']:
                    return 'PhonemeFakeV2'
                elif sub_path in ['LlamaPartialSpoof', 'llamapartialspoof']:
                    return 'LlamaPartialSpoof'
                else:
                    return sub_path
    
    # 如果没有匹配到，尝试从路径中提取最后一个有效的目录名
    parts = [p for p in path.split('/') if p]
    if parts:
        # 尝试找到 Datasets 目录后的名称
        try:
            idx = parts.index('Datasets')
            if idx + 1 < len(parts):
                return parts[idx + 1]
        except ValueError:
            pass
        # 如果找不到，返回最后一个非空部分
        return parts[-2] if len(parts) > 1 else parts[-1]
    
    return 'Unknown'


def parse_detection_and_localization(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从文本中解析 Detection Result 和 Localization Result
    
    Args:
        text: 包含检测和定位结果的文本
    
    Returns:
        (detection_result, localization_result) 元组
    """
    if not text or not isinstance(text, str):
        return None, None
    
    # 使用正则表达式匹配 Detection Result 和 Localization Result
    # 兼容可能出现的 markdown 粗体/序号等格式
    detection_match = re.search(r'Detection Result:\s*([^\n]+)', text, re.IGNORECASE)
    localization_match = re.search(r'Localization Result:\s*([^\n]+)', text, re.IGNORECASE)
    
    detection = detection_match.group(1).strip() if detection_match else None
    # 兼容两种格式：
    # 1) 三分类评估格式：包含 "Localization Result: ..."
    # 2) 纯定位任务：文本本身就是 "0.0-1.0, 2.0-3.0" 或 "None"
    localization = localization_match.group(1).strip() if localization_match else text.strip()
    
    return detection, localization


def parse_time_intervals(loc_str: str) -> List[Tuple[float, float]]:
    """
    解析定位结果中的时间区间字符串
    
    Args:
        loc_str: 时间区间字符串，例如 "0.0-0.49, 1.41-1.67" 或 "None"
    
    Returns:
        时间区间列表，例如 [(0.0, 0.49), (1.41, 1.67)]
    """
    if not loc_str or not isinstance(loc_str, str) or loc_str.lower() in ['none', 'null', '']:
        return []
    
    intervals = []
    # 分割多个区间（用逗号分隔）
    parts = [p.strip() for p in loc_str.split(',')]
    for part in parts:
        # 匹配 "start-end" 格式（允许空格）
        match = re.match(r'^([\d.]+)\s*-\s*([\d.]+)$', part)
        if match:
            try:
                start = float(match.group(1))
                end = float(match.group(2))
                if start < end:  # 有效区间
                    intervals.append((start, end))
            except ValueError:
                continue
    
    return intervals


def calculate_tiou(pred_intervals: List[Tuple[float, float]], 
                   gt_intervals: List[Tuple[float, float]]) -> float:
    """
    计算 Temporal IoU (TIoU)
    
    TIoU 用于评估预测时间区间与真实时间区间的重叠程度
    计算公式：TIoU = Intersection / Union
    
    Args:
        pred_intervals: 预测的时间区间列表
        gt_intervals: 真实的时间区间列表
    
    Returns:
        TIoU 值（0-1之间）
    """
    if not pred_intervals and not gt_intervals:
        return 1.0  # 都为空，完全匹配
    if not pred_intervals or not gt_intervals:
        return 0.0  # 一个为空一个不为空，完全不匹配
    
    # 计算交集：找到预测和真实区间之间的重叠部分
    intersection_length = 0.0
    for pred_start, pred_end in pred_intervals:
        for gt_start, gt_end in gt_intervals:
            overlap_start = max(pred_start, gt_start)
            overlap_end = min(pred_end, gt_end)
            if overlap_start < overlap_end:
                intersection_length += overlap_end - overlap_start
    
    # 计算并集：合并所有区间
    all_points = []
    for start, end in pred_intervals + gt_intervals:
        all_points.append((start, 1))  # 区间开始
        all_points.append((end, -1))   # 区间结束
    
    all_points.sort()
    union_length = 0.0
    active_intervals = 0
    last_point = None
    
    for point, delta in all_points:
        if active_intervals > 0 and last_point is not None:
            union_length += point - last_point
        active_intervals += delta
        last_point = point
    
    if union_length == 0:
        return 1.0 if intersection_length == 0 else 0.0
    
    return intersection_length / union_length


def _calculate_binary_metrics(y_true: List[str], y_pred: List[str], positive_class: str) -> Dict:
    """计算二分类指标（指定正类）"""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == positive_class and p == positive_class:
            tp += 1
        elif t != positive_class and p == positive_class:
            fp += 1
        elif t == positive_class and p != positive_class:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _get_empty_metrics() -> Dict:
    """返回空指标字典"""
    return {
        'total_acc': 0.0,
        'class_acc': {},
        'weighted_precision': 0.0,
        'weighted_recall': 0.0,
        'weighted_f1': 0.0,
    }


def evaluate_detection(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    评估三分类检测结果，计算：
      - 每个类别的 Precision/Recall/F1
      - Macro/Weighted 平均指标
      - 二分类指标（partially fake 作为正类）
      - 每类准确率（原逻辑保留）
    """
    # 标准化标签
    y_true_norm = [normalize_label(label) for label in y_true]
    y_pred_norm = [normalize_label(label) for label in y_pred]
    
    # 过滤 unknown 标签
    valid_pairs = [(t, p) for t, p in zip(y_true_norm, y_pred_norm) 
                   if t != 'unknown' and p != 'unknown']
    
    if not valid_pairs:
        return _get_empty_metrics()
    
    y_true_valid = [t for t, _ in valid_pairs]
    y_pred_valid = [p for _, p in valid_pairs]
    
    # 获取有效类别（标准三类 + 可能存在的其他类别）
    STANDARD_CLASSES = ['fully real', 'fully fake', 'partially fake']
    classes = sorted(set(y_true_valid))
    
    # 优先使用标准类别顺序，再添加其他类别
    ordered_classes = [c for c in STANDARD_CLASSES if c in classes]
    ordered_classes.extend([c for c in classes if c not in STANDARD_CLASSES])
    
    if not ordered_classes:
        return _get_empty_metrics()
    
    # 初始化混淆矩阵计数
    tp = {cls: 0 for cls in ordered_classes}
    fp = {cls: 0 for cls in ordered_classes}
    fn = {cls: 0 for cls in ordered_classes}
    
    # 计算 TP/FP/FN
    for t, p in zip(y_true_valid, y_pred_valid):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            if p in fp:  # 防止预测了未知类别
                fp[p] += 1
    
    # 计算每个类别的指标
    class_metrics = {}
    class_counts = {cls: y_true_valid.count(cls) for cls in ordered_classes}
    
    for cls in ordered_classes:
        denom_p = tp[cls] + fp[cls]
        denom_r = tp[cls] + fn[cls]
        
        precision = tp[cls] / denom_p if denom_p > 0 else 0.0
        recall = tp[cls] / denom_r if denom_r > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': class_counts[cls]
        }
        
    # 计算加权平均（按样本数加权）
    total_samples = sum(class_counts.values())
    weighted_precision = sum(
        class_metrics[cls]['precision'] * class_counts[cls] 
        for cls in ordered_classes
    ) / total_samples if total_samples > 0 else 0.0
    
    weighted_recall = sum(
        class_metrics[cls]['recall'] * class_counts[cls] 
        for cls in ordered_classes
    ) / total_samples if total_samples > 0 else 0.0
    
    weighted_f1 = sum(
        class_metrics[cls]['f1'] * class_counts[cls] 
        for cls in ordered_classes
    ) / total_samples if total_samples > 0 else 0.0
    
    # 总体准确率
    total_acc = sum(1 for t, p in zip(y_true_valid, y_pred_valid) if t == p) / len(y_true_valid)
    
    # 每类准确率（原逻辑）
    class_acc = {}
    for cls in ordered_classes:
        cls_samples = [(t, p) for t, p in zip(y_true_valid, y_pred_valid) if t == cls]
        if cls_samples:
            class_acc[cls] = sum(1 for t, p in cls_samples if t == p) / len(cls_samples)
        else:
            class_acc[cls] = 0.0
    
    return {
        'total_acc': total_acc,
        'class_acc': class_acc,
        'ordered_classes': ordered_classes,  # 保留类别顺序
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }


def evaluate_dataset(jsonl_file: str) -> Dict[str, Dict]:
    """
    评估jsonl文件中的所有数据集，按数据集分组
    
    Args:
        jsonl_file: 输入的jsonl文件路径
    
    Returns:
        字典，键为数据集名称，值为评估结果字典
    """
    # 按数据集分组存储数据
    dataset_data = defaultdict(lambda: {
        'y_true_det': [],      # 真实检测标签
        'y_pred_det': [],      # 预测检测标签
        'y_true_loc': [],      # 真实定位标签
        'y_pred_loc': [],      # 预测定位标签
        'tious': []            # 存储所有样本的TIoU值
    })
    
    print(f"\n正在读取文件: {jsonl_file}")
    
    # 读取jsonl文件
    valid_lines = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if not line.strip():
                    continue
                
                data = json.loads(line.strip())
                response = data.get('response', '')
                labels = data.get('labels', '')
                audios = data.get('audios', [])
                
                # 从音频路径解析数据集名称
                if audios and len(audios) > 0:
                    audio_path = audios[0]  # 取第一个音频路径
                    dataset_name = parse_dataset_from_audio_path(audio_path)
                else:
                    dataset_name = 'Unknown'
                
                # 解析检测和定位结果
                pred_det, pred_loc = parse_detection_and_localization(response)
                true_det, true_loc = parse_detection_and_localization(labels)
                
                # 特殊规则：如果检测为 fully real，定位结果应该为 none
                if pred_det and normalize_label(pred_det) == 'fully real':
                    pred_loc = 'None'
                if true_det and normalize_label(true_det) == 'fully real':
                    true_loc = 'None'
                
                # 存储到对应的数据集分组中
                dataset_data[dataset_name]['y_true_det'].append(true_det)
                dataset_data[dataset_name]['y_pred_det'].append(pred_det)
                dataset_data[dataset_name]['y_true_loc'].append(true_loc)
                dataset_data[dataset_name]['y_pred_loc'].append(pred_loc)
                
                # 计算TIoU
                pred_intervals = parse_time_intervals(pred_loc) if pred_loc else []
                true_intervals = parse_time_intervals(true_loc) if true_loc else []
                tiou = calculate_tiou(pred_intervals, true_intervals)
                dataset_data[dataset_name]['tious'].append(tiou)
                
                valid_lines += 1
                
            except Exception as e:
                print(f"  警告：处理第{line_num}行时出错: {e}")
                continue
    
    print(f"  总共读取 {line_num} 行，有效数据 {valid_lines} 行")
    print(f"  识别到 {len(dataset_data)} 个数据集: {sorted(dataset_data.keys())}")
    
    # 对每个数据集进行评估
    results = {}
    for dataset_name, data in dataset_data.items():
        print(f"\n正在评估数据集: {dataset_name} (样本数: {len(data['y_true_det'])})")
        
        y_true_det = data['y_true_det']
        y_pred_det = data['y_pred_det']
        tious = data['tious']
        
        # 1. 评估检测结果
        det_metrics = evaluate_detection(y_true_det, y_pred_det)
        
        # 2. 评估定位结果（只考虑检测正确的样本）
        # 过滤出检测正确的样本索引
        y_true_norm = [normalize_label(label) for label in y_true_det]
        y_pred_norm = [normalize_label(label) for label in y_pred_det]
        correct_det_indices = [i for i, (t, p) in enumerate(zip(y_true_norm, y_pred_norm)) 
                              if t != 'unknown' and p != 'unknown' and t == p]
        correct_tious = [tious[i] for i in correct_det_indices] if correct_det_indices else []
        
        # 计算不同阈值下的AP (Average Precision)
        thresholds = [0.5, 0.75, 0.9, 0.95]
        ap_results = {}
        for threshold in thresholds:
            # AP：在检测正确的样本中，TIoU >= threshold 的比例
            ap = sum(1 for tiou in correct_tious if tiou >= threshold) / len(correct_tious) if correct_tious else 0.0
            ap_results[threshold] = ap
        
        # 计算mAP (mean AP) under TIoU thresholds [0.5 : 0.05 : 0.95]
        tiou_thresholds = np.arange(0.5, 1.0, 0.05)
        ap_scores = []
        for threshold in tiou_thresholds:
            ap = sum(1 for tiou in correct_tious if tiou >= threshold) / len(correct_tious) if correct_tious else 0.0
            ap_scores.append(ap)
        
        mean_ap = np.mean(ap_scores) if ap_scores else 0.0
                
        # 3. 综合评估结果
        results[dataset_name] = {
            'detection': det_metrics,
            'localization': {
                'ap_scores': {f'AP@{k}': v for k, v in ap_results.items()},
                'map': float(mean_ap)
            }
        }
        
        print(f"  Total_ACC: {det_metrics['total_acc']*100:.2f}%")
        print(f"  Weighted_F1: {det_metrics['weighted_f1']*100:.2f}%")
        print(f"  mAP: {mean_ap*100:.2f}%")
    
    return results


def save_to_excel(all_results: Dict[str, Dict], output_file: str):
    """
    将所有数据集的结果保存为Excel格式
    
    格式：行为数据集，列为评估指标
    
    输出指标包括：
      - 每类准确率 (ACC)
      - 每类F1分数
      - Macro/Weighted 平均指标
      - 二分类指标（partially fake 作为正类）
      - 定位指标（TIoU, AP, mAP）
    """
    if not all_results:
        print("警告：没有评估结果，无法保存Excel")
        return
    
    # 获取所有出现的类别（按标准顺序）
    STANDARD_CLASSES = ['fully real', 'fully fake', 'partially fake']
    all_classes = set()
    for results in all_results.values():
        all_classes.update(results['detection']['ordered_classes'])
    
    # 构建有序类别列表
    ordered_classes = [c for c in STANDARD_CLASSES if c in all_classes]
    ordered_classes.extend(sorted([c for c in all_classes if c not in STANDARD_CLASSES]))
    
    # 定义类别名称到Excel列名的映射
    def class_to_column_name(cls: str) -> str:
        """将类别名称转换为Excel列名"""
        clean_cls = cls.replace(' ', '').replace('-', '').replace('_', '')
        # return f"{clean_cls}_ACC", f"{clean_cls}_F1"
        return f"{clean_cls}_ACC"
    
    # 构建指标列名
    metric_names = ['Total_ACC']
    
    # 1. 每类准确率
    for cls in ordered_classes:
        acc_col = class_to_column_name(cls)
        metric_names.append(acc_col)
    
    # 2. 每类F1分数
    # for cls in ordered_classes:
    #     _, f1_col = class_to_column_name(cls)
    #     metric_names.append(f1_col)
    
    # 3. 宏平均指标
    # metric_names.extend(['Macro_Precision', 'Macro_Recall', 'Macro_F1'])
    
    # 4. 加权平均指标（工业场景主指标）
    metric_names.extend(['Weighted_Precision', 'Weighted_Recall', 'Weighted_F1'])
    
    # 5. 二分类指标（辅助参考）
    # metric_names.extend(['Binary_Precision', 'Binary_Recall', 'Binary_F1'])
    
    # 6. 定位指标
    metric_names.extend([
        'AP@0.5', 'AP@0.75', 'AP@0.9', 'AP@0.95', 'mAP'
    ])
    
    # 定义数据集排序顺序
    dataset_order = [
        'PartialSpoof', 'HAD', 'LAV-DF', 'SINE_v2', 
        'LlamaPartialSpoof', 'ArEnAV', 'AV-Deepfake1M-PlusPlus',
        'ADD2023', 'Speech-Forensics', 
        'PartialEdit', 'PhonemeFakeV2'
    ]
    
    # 对数据集进行排序：先按指定顺序，然后按字母顺序添加其他数据集
    sorted_datasets = []
    remaining_datasets = set(all_results.keys())
    
    for ds in dataset_order:
        if ds in remaining_datasets:
            sorted_datasets.append(ds)
            remaining_datasets.remove(ds)
    
    # 添加其他未列出的数据集（按字母顺序）
    sorted_datasets.extend(sorted(remaining_datasets))
    
    # 准备数据行
    rows = []
    for dataset_name in sorted_datasets:
        results = all_results[dataset_name]
        det = results['detection']
        loc = results['localization']
        values = []
        
        # Total_ACC
        values.append(f"{det['total_acc'] * 100:.2f}")
        
        # 每类准确率
        for cls in ordered_classes:
            acc = det['class_acc'].get(cls, 0.0) * 100
            values.append(f"{acc:.2f}")
        
        # Weighted 指标
        values.append(f"{det['weighted_precision'] * 100:.2f}")
        values.append(f"{det['weighted_recall'] * 100:.2f}")
        values.append(f"{det['weighted_f1'] * 100:.2f}")
        
        # 定位指标
        for th in [0.5, 0.75, 0.9, 0.95]:
            ap_val = loc['ap_scores'].get(f'AP@{th}', 0.0) * 100
            values.append(f"{ap_val:.2f}")
        values.append(f"{loc['map'] * 100:.2f}")
        
        rows.append([dataset_name] + values)
    
    # 创建DataFrame
    columns = ['Dataset'] + metric_names
    df = pd.DataFrame(rows, columns=columns)
    
    # 保存为Excel
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # 自动调整列宽
        worksheet = writer.sheets['Results']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 25)
    
    print(f"\n✅ 三分类评估完成！结果已保存至: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate ThinkOmni JSONL predictions.')
    parser.add_argument('input_jsonl', help='Path to an ms-swift JSONL result file.')
    parser.add_argument('--output', default=None, help='Output Excel path (default: beside the input file).')
    args = parser.parse_args()
    input_jsonl_file = args.input_jsonl
    output_excel_file = args.output or input_jsonl_file.replace('.jsonl', '.xlsx')
    
    print("=" * 80)
    print("🔊 三分类音频检测评估脚本 (Fully Real / Fully Fake / Partially Fake)")
    print("=" * 80)
    print(f"输入文件: {input_jsonl_file}")
    print(f"输出文件: {output_excel_file}")
    
    # 评估数据集
    try:
        all_results = evaluate_dataset(input_jsonl_file)
        
        if not all_results:
            print("\n⚠️ 警告：没有找到任何有效数据！")
            return
        
        # 保存到Excel
        save_to_excel(all_results, output_excel_file)
        
        print(f"\n✅ 评估完成！共处理 {len(all_results)} 个数据集")
        
        # 打印汇总统计
        print("\n" + "=" * 80)
        print("📈 评估结果汇总")
        print("=" * 80)
        for ds_name in sorted(all_results.keys()):
            det = all_results[ds_name]['detection']
            loc = all_results[ds_name]['localization']
            print(f"{ds_name:25s} | Weighted_F1: {det['weighted_f1']*100:5.2f}% | "
                  f"mAP: {loc['map']*100:5.2f}%")
        
    except Exception as e:
        print(f"\n❌ 错误：评估过程中出现异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
