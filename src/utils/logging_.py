import numpy as np
import os
import sys
import logging
import csv

def get_parent_curr_dir():
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_file_path))
    sys.path.append(parent_dir)
    current_dir = os.path.dirname(current_file_path)
    return parent_dir, current_dir

def make_dir(experiment): 
    parent_dir, current_dir = get_parent_curr_dir()
    logfile = os.path.join(parent_dir, f'{experiment}_log.txt')
    save_dir = os.path.join(parent_dir, 'checkpoints', experiment)
    result_dir = os.path.join(parent_dir, 'result', experiment)
    cache_dir = os.path.join(parent_dir, 'cache')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    return logfile, save_dir, result_dir, cache_dir 

def get_save_name(args, pretrain_dataset_names, save_dir, result_dir): 
    pretrain_dataset_str = ''
    for strs in pretrain_dataset_names: 
        pretrain_dataset_str += '_'+strs
    # set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
    # set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.ablation_pre}_{args.sample_size}_{args.nb_epochs}_{args.de_loss}_{args.de_weight}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'

    set_name = f'model_node_{args.pretrain_method}_{pretrain_dataset_str}_{args.ablation_pre}_{args.sample_size}_{args.nb_epochs}_{args.if_rand}_{args.w1loss}_{args.de_loss}_{args.de_weight}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'

    save_name = os.path.join(save_dir, f'{set_name}.pkl')
    csv_name = os.path.join(result_dir, f'{set_name}.csv')

    return save_name, csv_name

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False

def write(txt='\n'): 
    print(txt)
    logging.info(txt)

import pandas as pd
def write_rst(accs, shot_num, microf=None, macrof=None, csv_name='', seed=0, task=''):
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
    
    write('-' * 100)
    write(f"[{shot_num}-shot]")
    write(f"Acc: {acc_mean:.2f} ± {acc_std:.2f}")

    if microf is not None and macrof is not None: 
        micro_mean = np.mean(microf)
        macro_mean = np.mean(macrof)
        micro_std = np.std(microf)
        macro_std = np.std(macrof)
        write(f"Macro F1: {macro_mean:.2f} ± {macro_std:.2f}, Micro F1: {micro_mean:.2f} ± {micro_std:.2f}")
    write(f"{'-' * 100}\n")

    accs = np.array(accs)

    # 100회 결과 + 평균 1개까지 → 길이 101
    data = {
        f"Acc_{shot_num}shot({seed})_{task}": np.append(accs, accs.mean().round(3)),
        f"Mic_{shot_num}shot({seed})_{task}": np.append(microf, microf.mean().round(3)),
        f"Mac_{shot_num}shot({seed})_{task}": np.append(macrof, macrof.mean().round(3)),
    }
    df_new = pd.DataFrame(data)

    # 파일이 있으면 기존 df 불러와서 merge
    if os.path.exists(csv_name):
        df_old = pd.read_csv(csv_name, encoding="utf-8-sig")
        # 행 수가 다르면 맞춰줌 (빈칸 채움)
        max_len = max(len(df_old), len(df_new))
        df_old = df_old.reindex(range(max_len))
        df_new = df_new.reindex(range(max_len))
        df_out = pd.concat([df_old, df_new], axis=1)
    else:
        df_out = df_new

    df_out.to_csv(csv_name, index=False, encoding="utf-8-sig")

    return acc_mean, macro_mean, micro_mean 

def write_rst2(accs, iter, microf=None, macrof=None):
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
    
    write(f"[Try {iter}]")
    write(f"Acc: {acc_mean:.2f} ± {acc_std:.2f}")

    if microf is not None and macrof is not None: 
        micro_mean = np.mean(microf)
        macro_mean = np.mean(macrof)
        micro_std = np.std(microf)
        macro_std = np.std(macrof)
        write(f"Macro F1: {macro_mean:.2f} ± {macro_std:.2f}, Micro F1: {micro_mean:.2f} ± {micro_std:.2f}\n")
    

    return acc_mean, macro_mean, micro_mean 

def get_pretrain_dataset_names(data, target_id):
        
    #pretrain_dataset_names = [data[source_id]]
    pretrain_dataset_names = []
    for i in range(len(data)): 
        if i != target_id: 
            pretrain_dataset_names.append(data[i])
    return pretrain_dataset_names

def log_args_table(args, max_per_line: int = 5, col_width: int = 30):
    """
    args: argparse.Namespace
    max_per_line: 한 줄에 몇 개 출력할지
    col_width: 각 열의 고정 폭
    """
    args_dict = vars(args)
    arg_items = [f"{k} = {v}" for k, v in sorted(args_dict.items())]

    # 패딩을 넣어 고정 길이 문자열로 변환
    padded_items = [item.ljust(col_width) for item in arg_items]

    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))
    logging.info("Arguments:")
    
    for i in range(0, len(padded_items), max_per_line):
        row = padded_items[i:i+max_per_line]
        logging.info(" | ".join(row))
    
    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))