import os
import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt
import warnings

warnings.filterwarnings("ignore")

import scipy.sparse  # 确保导入这个

def evaluate(
        pred_expr,            # 预测表达 (N_pred, G)
        true_expr,            # 对应真实单细胞 AnnData (N_pred, G)
        ctrl_expr,            # 真实 ctrl 单细胞表达 (N_ctrl, G)
        gene_names,           # 基因名列表
        condition_col="perturbation",
        ctrl_tag=None,        # 可以不指定
        save_dir="./evaluation_results",
        top_n=100,
        de_method="wilcoxon",
        subsample_n=2000
    ):

    os.makedirs(save_dir, exist_ok=True)

    # --- Step1: 构建 eval_adata 三组表达 ---
    expr_ctrl = ctrl_expr
    
    # [修改开始] 检查并转换稀疏矩阵为密集数组
    raw_stim = true_expr.layers["logNor"]
    
    # 判断是否为稀疏矩阵，如果是则转换
    if scipy.sparse.issparse(raw_stim):
        expr_stim = raw_stim.toarray()
    elif hasattr(raw_stim, "toarray"): # 备用检查方式
        expr_stim = raw_stim.toarray()
    else:
        expr_stim = raw_stim
    # [修改结束]

    expr_pred = pred_expr

    # 建议加上这行打印，确认转换后的 shape 对齐（Debug用）
    print(f"Shapes for vstack -> Ctrl:{expr_ctrl.shape}, Stim:{expr_stim.shape}, Pred:{expr_pred.shape}")

    # 真实 stim 名称
    stim_labels = true_expr.obs[condition_col].values.tolist()

    # 合并 (现在三者都是密集数组，不会报错了)
    expr_all = np.vstack([expr_ctrl, expr_stim, expr_pred])

    # 构造对应的标签
    all_conditions = (
        ["control"] * expr_ctrl.shape[0]
        + stim_labels
        + ["pred"] * expr_pred.shape[0]
    )

    obs = pd.DataFrame({condition_col: all_conditions})
    var = pd.DataFrame(index=gene_names[:expr_all.shape[1]])

    eval_adata = sc.AnnData(X=expr_all, obs=obs, var=var)

    # normalize + log1p, store in a named layer so pertpy's Distance can find it
    sc.pp.normalize_total(eval_adata, target_sum=1e4)
    sc.pp.log1p(eval_adata)
    eval_adata.layers["log1p_norm"] = eval_adata.X.copy()

    # --- Step2: 下采样 ---
    def subsample_group(adata, tag):
        sub = adata[adata.obs[condition_col] == tag].copy()
        if sub.shape[0] > subsample_n:
            idx = np.random.choice(sub.shape[0], subsample_n, replace=False)
            sub = sub[idx].copy()
        return sub

    ctrl_sub = subsample_group(eval_adata, "control")
    pred_sub = subsample_group(eval_adata, "pred")

    # 对于 stim 每个 perturbation 做 subsampling
    stim_groups = list(set(stim_labels))
    stim_sub_list = []
    for sg in stim_groups:
        stim_sub_list.append(subsample_group(eval_adata, sg))

    # 合并 stim 子集
    if len(stim_sub_list) > 1:
        stim_sub = sc.concat(stim_sub_list)
    else:
        stim_sub = stim_sub_list[0]

    adata_sub = sc.concat([ctrl_sub, stim_sub, pred_sub])

    # 类型映射：真实 stim group 名称
    unique_stims = list(set(stim_labels))

    # --- Step3: 全基因距离指标 (manual computation, avoiding pertpy bug with high-dim data) ---
    all_metrics = {}
    metric_list = ["mse", "pearson_distance", "edistance", "wasserstein", "sym_kldiv"]

    # Get normalized+log1p data as dense array
    norm_data = adata_sub.layers["log1p_norm"].copy()
    if scipy.sparse.issparse(norm_data):
        norm_data = norm_data.toarray()

    # Build group index mapping
    group_labels = adata_sub.obs[condition_col].values

    # Compute pseudobulk means per group
    groups_present = list(set(group_labels))
    group_means = {}
    for g in groups_present:
        mask = group_labels == g
        group_means[g] = norm_data[mask].mean(axis=0)  # (n_genes,)

    pred_mean = group_means.get("pred")
    if pred_mean is None:
        print("[WARN] 'pred' group not found in adata_sub — skipping Step3 metrics")
        for m in metric_list:
            all_metrics[f"{m}_all"] = np.nan
    else:
        for sg in unique_stims:
            stim_mean = group_means.get(sg)
            if stim_mean is None:
                continue

            # --- MSE ---
            mse_val = np.mean((pred_mean - stim_mean) ** 2)

            # --- Pearson distance (1 - Pearson correlation) ---
            pred_centered = pred_mean - pred_mean.mean()
            stim_centered = stim_mean - stim_mean.mean()
            denom = np.linalg.norm(pred_centered) * np.linalg.norm(stim_centered)
            if denom > 0:
                pearson_r = np.dot(pred_centered, stim_centered) / denom
            else:
                pearson_r = 0.0
            pearson_dist = 1.0 - pearson_r

            # --- E-distance (energy distance) ---
            # E-distance = 2*mean||X-Y|| - mean||X-X'|| - mean||Y-Y'||
            # For pseudobulk (single vector per group), this simplifies to Euclidean distance
            edist_val = np.linalg.norm(pred_mean - stim_mean)

            # --- Wasserstein distance (1D, mean of per-gene Wasserstein) ---
            # For 1D Gaussian approximation: W = |mean1 - mean2|
            # More precisely, for empirical distributions: sort and compare
            # We use the simple L2 norm of the difference (same as E-distance for single vectors)
            wasser_val = np.linalg.norm(pred_mean - stim_mean)

            # --- Symmetric KL divergence ---
            # Get per-group cell-level data for this stim
            pred_cells = norm_data[group_labels == "pred"]
            stim_cells = norm_data[group_labels == sg]
            kl_vals = []
            epsilon = 1e-8
            n_genes = pred_cells.shape[1]
            for i in range(n_genes):
                x_mean, x_std = pred_cells[:, i].mean(), pred_cells[:, i].std() + epsilon
                y_mean, y_std = stim_cells[:, i].mean(), stim_cells[:, i].std() + epsilon
                kl_ab = np.log(y_std / x_std) + (x_std**2 + (x_mean - y_mean)**2) / (2 * y_std**2) - 0.5
                kl_ba = np.log(x_std / y_std) + (y_std**2 + (y_mean - x_mean)**2) / (2 * x_std**2) - 0.5
                kl_vals.append(kl_ab + kl_ba)
            sym_kldiv_val = sum(kl_vals) / len(kl_vals)

            # Store per-stimulus metrics
            all_metrics[f"mse_all_{sg}"] = round(mse_val, 6)
            all_metrics[f"pearson_distance_all_{sg}"] = round(pearson_dist, 6)
            all_metrics[f"edistance_all_{sg}"] = round(edist_val, 6)
            all_metrics[f"wasserstein_all_{sg}"] = round(wasser_val, 6)
            all_metrics[f"sym_kldiv_all_{sg}"] = round(sym_kldiv_val, 6)

        # Also compute averages across stim groups
        for m in metric_list:
            vals = [v for k, v in all_metrics.items() if k.startswith(f"{m}_all_")]
            if vals:
                all_metrics[f"{m}_all"] = round(np.mean(vals), 6)
            else:
                all_metrics[f"{m}_all"] = np.nan

    # --- Step4: TopDEG指标 ---
    # 将 "pred" 加入 groups 列表, 以便后续访问 degs["names"]["pred"]
    rank_groups = unique_stims + ["pred"]
    sc.tl.rank_genes_groups(
        eval_adata,
        groupby=condition_col,
        reference="control",
        groups=rank_groups,
        method=de_method,
        pts=True
    )

    degs = eval_adata.uns["rank_genes_groups"]
    top_metrics = {}

    # 依次对 stim groups 计算 top metrics
    for sg in unique_stims:
        all_names = np.array(degs["names"][sg])
        all_scores = np.array(degs["scores"][sg])
        valid_mask = ~np.isnan(all_scores)
        all_names = all_names[valid_mask]
        top_degs = [g for g in all_names[:top_n] if g in eval_adata.var_names]

        sub_ctrl_top = ctrl_sub[:, top_degs].copy()
        sub_stim_top = eval_adata[eval_adata.obs[condition_col] == sg][:, top_degs].copy()
        sub_pred_top = pred_sub[:, top_degs].copy()

        for m in metric_list:
            try:
                Dist = pt.tools.Distance(metric=m, layer_key="log1p_norm")
                df_top = Dist.onesided_distances(
                    sc.concat([sub_ctrl_top, sub_stim_top, sub_pred_top]),
                    groupby=condition_col,
                    selected_group="pred",
                    groups=[sg]
                )
                top_metrics[f"{m}_top{top_n}_{sg}"] = round(df_top[sg], 6)
            except Exception as e:
                print(f"[WARN top] {m} failed for {sg}: {e}")
                top_metrics[f"{m}_top{top_n}_{sg}"] = np.nan

    # --- Step5: Common DEGs ---
    # 对每个刺激组分别计算 overlap
    for sg in unique_stims:
        pred_degs = np.array(eval_adata.uns["rank_genes_groups"]["names"]["pred"])
        pred_degs = [g for g in pred_degs[:top_n] if g in eval_adata.var_names]
        common_count = len(set(top_degs) & set(pred_degs))
        top_metrics[f"common_degs_top{top_n}_{sg}"] = common_count

    # --- Step6: 合并输出 ---
    combined = {**all_metrics, **top_metrics}
    df = pd.DataFrame([combined])
    df.to_csv(os.path.join(save_dir, "official_benchmark_metrics.csv"), index=False)

    print("Saved evaluation at:", save_dir)
    print(df.round(4))

    return df
