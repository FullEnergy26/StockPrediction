"""
SC2616025 郑慧敏
股票涨跌二分类：对照实验与改进方法

数据源: 本地 CSV (300750 宁德时代, 2021-01-04 ~ 2025-12-31)
特征: 26 维 (OHLCV + 技术指标 + 市场微观结构特征)
任务: 用当天特征预测下一个交易日涨跌

实验设计:
  对照组:
    1. 逻辑回归 (LR)        线性基线, 判断非线性特征的增益
    2. 随机森林 (RF)        经典 Bagging 集成树
    3. 多层感知机 (MLP)     前馈神经网络
    4. 单向 LSTM            序列模型基准, 取最后一步隐状态

  改进组:
    5. XGBoost              梯度提升树, Boosting 替代 Bagging
    6. AttnLSTM             双向 LSTM + 多头自注意力 + 可学习查询池化
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # 无 GUI 后端, 命令行可用
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import xgboost as xgb


# 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "stock_data.csv"     # 用户下载的本地 CSV 文件
SEQ_LEN = 20                    # 序列模型回看 20 个交易日
BATCH = 32
EPOCHS = 200
LR = 0.001
TRAIN, VAL = 0.70, 0.85         # 训练 70%, 验证 15%, 测试 15%
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"[Config] Device={DEVICE}, SeqLen={SEQ_LEN}, Epochs={EPOCHS}")
print(f"[Config] Train/Val/Test = {TRAIN:.0%}/{VAL-TRAIN:.0%}/{1-VAL:.0%}")


# ------------------- 1. 数据获取 -------------------

def fetch_data(path: str = CSV_PATH) -> pd.DataFrame:
    """从本地 CSV 读取股票日线数据, 中文列名转英文, 成交量字符串转数值"""
    df = pd.read_csv(path)
    # 中文列名映射为英文
    df.rename(columns={
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "高": "High", "低": "Low", "交易量": "Volume",
    }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)              # 日期升序

    def _parse_vol(val):
        """解析带量级后缀的成交量字符串, 如 16.03M → 16030000"""
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().upper()
        for suffix, mul in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
            if s.endswith(suffix):
                return float(s[:-1]) * mul
        return float(s)

    df["Volume"] = df["Volume"].apply(_parse_vol)
    # 只保留五个核心价格字段
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    print(f"[Data] 加载成功: {len(df)} 行, "
          f"{df.index[0].date()} ~ {df.index[-1].date()}")
    return df


# ------------------- 2. 特征工程 -------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV 计算全部 26 维特征和次日涨跌标签"""
    O, H, L, C, V = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # ---- 基础技术指标 (9 个) ----
    df["MA5"]  = C.rolling(5).mean()
    df["MA10"] = C.rolling(10).mean()
    df["MA20"] = C.rolling(20).mean()

    # RSI(14)
    d = C.diff()
    gain, loss = d.clip(0), (-d).clip(0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))

    # MACD
    ema12 = C.ewm(12, adjust=False).mean()
    ema26 = C.ewm(26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["MACD_S"] = df["MACD"].ewm(9, adjust=False).mean()
    df["MACD_H"] = df["MACD"] - df["MACD_S"]

    # 5 日价格斜率 (线性回归), 除以收盘价做标准化
    def _slope(s, w=5):
        x = np.arange(w)
        return s.rolling(w).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
    df["Slope"] = _slope(C) / (C + 1e-8)

    # 20 日波动率和 5 日成交量变化率
    df["Volat"] = C.pct_change().rolling(20).std()
    df["VChg"]  = V.pct_change(5)

    # ---- 增强结构特征 (12 个) ----
    df["OvrGap"]  = (O - C.shift(1)) / (C.shift(1) + 1e-8)    # 隔夜跳空
    df["Range"]   = (H - L) / (C + 1e-8)                       # 日内振幅
    df["ClsPos"]  = (C - L) / (H - L + 1e-8)                  # 收盘在日内位置
    df["VolRat"]  = V / (V.rolling(5).mean() + 1e-8)           # 量比
    df["R1d"]     = C.pct_change(1)                            # 1 日收益
    df["R3d"]     = C.pct_change(3)                            # 3 日动量
    df["R5d"]     = C.pct_change(5)                            # 5 日动量
    df["MA5dst"]  = C / (df["MA5"]  + 1e-8) - 1              # 偏离 MA5 百分比
    df["MA20dst"] = C / (df["MA20"] + 1e-8) - 1              # 偏离 MA20 百分比
    df["RSI_chg"] = df["RSI"].diff(3)                          # RSI 3 日变化
    df["PrevDir"] = (C.shift(1) > C.shift(2)).astype(int)     # 上一日涨跌方向

    # 次日涨(1)跌(0)标签
    df["Target"] = (C.shift(-1) > C).astype(int)
    df.dropna(inplace=True)
    print(f"[Feature] {len(df.columns)} 列 ({len(df)} 行)")
    return df


# ------------------- 3. 数据准备 -------------------

# 26 个输入特征的分组说明
FEATS = [
    "Open", "High", "Low", "Close", "Volume",       # OHLCV 原始价量
    "MA5", "MA10", "MA20", "RSI",                    # 均线 + RSI
    "MACD", "MACD_S", "MACD_H", "Slope", "Volat",    # MACD 族 + 斜率 + 波动
    "OvrGap", "Range", "ClsPos", "VolRat", "VChg",   # 市场微观结构
    "R1d", "R3d", "R5d", "MA5dst", "MA20dst",       # 多尺度动量 + 均线偏离
    "RSI_chg", "PrevDir",                            # RSI 动量 + 上一日方向
]


def prep_flat(df):
    """制作扁平数据: 第 i 天 26 个特征 → 第 i 天的涨跌标签
       注意 Target[i] 已经是次日涨跌, 所以 X[i] 和 y[i] 是对齐的
    """
    X, y = df[FEATS].values.astype(np.float64), df["Target"].values
    n = len(X)
    t_end = int(n * TRAIN)
    v_end = int(n * VAL)
    # 只在训练集上计算均值和标准差, 然后 transform 全部数据
    sc = StandardScaler().fit(X[:t_end])
    X_s = sc.transform(X)
    return X_s[:t_end], y[:t_end], X_s[t_end:v_end], y[t_end:v_end], X_s[v_end:], y[v_end:]


def prep_seq(df, L=SEQ_LEN):
    """制作序列数据: 过去 L 天的滑窗 → 最后一天的涨跌标签
       形状为 (样本数, L, 26)
    """
    X, y = df[FEATS].values.astype(np.float64), df["Target"].values
    n = len(X)
    t_end = int(n * TRAIN)
    v_end = int(n * VAL)
    sc = StandardScaler().fit(X[:t_end])
    X_s = sc.transform(X)

    xs, ys = [], []
    for i in range(n - L):
        xs.append(X_s[i: i + L])
        ys.append(y[i + L - 1])     # 窗口最后一天的标签

    Xa = np.array(xs, dtype=np.float32)
    ya = np.array(ys, dtype=np.float32)
    n2 = len(Xa)
    t2 = int(n2 * TRAIN)
    v2 = int(n2 * VAL)
    return Xa[:t2], ya[:t2], Xa[t2:v2], ya[t2:v2], Xa[v2:], ya[v2:]


# ------------------- 4. 模型定义 -------------------

class MLP(nn.Module):
    """基线 3: 三层前馈网络, 128→64→1"""
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_in, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class LSTM_(nn.Module):
    """基线 4: 单层单向 LSTM, 取最后一步隐状态"""
    def __init__(self, d_in, h=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, 1, batch_first=True, dropout=0.3)
        self.hd = nn.Sequential(
            nn.Linear(h, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.hd(out[:, -1, :])    # 只用最后一个时间步


class AttnLSTM(nn.Module):
    """改进 6: 双向 LSTM + 多头自注意力 + 可学习查询池化

       与基线 LSTM 的差异:
       1. 单向→双向, 每个时间步看到前后文
       2. 增加 4 头自注意力层 + 残差 + LayerNorm
       3. 用一个可学习的向量 q 对所有时间步做交叉注意力,
          代替硬取最后一步
    """
    def __init__(self, d_in, h=64, nh=4):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, 1, batch_first=True, dropout=0.3,
                            bidirectional=True)
        h2 = h * 2                              # 双向拼接后维度翻倍
        self.attn = nn.MultiheadAttention(h2, nh, 0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(h2)            # 自注意力后的残差归一化
        self.q = nn.Parameter(torch.randn(1, 1, h2) * 0.1)  # 可学习查询向量
        self.hd = nn.Sequential(
            nn.Linear(h2, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                   # (B, L, 2h)
        attn_out, _ = self.attn(out, out, out)  # 自注意力
        out = self.norm1(out + attn_out)        # 残差 + 归一化
        q = self.q.expand(x.size(0), -1, -1)    # 扩展到 batch 维度
        pooled, _ = self.attn(q, out, out)      # 可学习查询池化
        return self.hd(pooled.squeeze(1))


# ------------------- 5. 训练 -------------------

def train_epoch(model, loader, opt, crit):
    """训练一个 epoch, 返回平均损失和准确率"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for bx, by in loader:
        bx = bx.to(DEVICE)
        by = by.to(DEVICE).float().unsqueeze(1)
        opt.zero_grad()
        pred = model(bx)
        loss = crit(pred, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防爆炸
        opt.step()
        total_loss += loss.item() * len(bx)
        correct += ((pred > 0).float() == by).sum().item()
        total += len(bx)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_torch(model, loader, crit):
    """评估函数: 返回损失, 准确率, 预测概率, 真实标签"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    probs, truths = [], []
    for bx, by in loader:
        bx = bx.to(DEVICE)
        by = by.to(DEVICE).float().unsqueeze(1)
        pred = model(bx)
        loss = crit(pred, by)
        total_loss += loss.item() * len(bx)
        prob = torch.sigmoid(pred)               # logit → 概率
        correct += ((prob > 0.5).float() == by).sum().item()
        total += len(bx)
        probs.append(prob.cpu().numpy())
        truths.append(by.cpu().numpy())
    return (total_loss / total,
            correct / total,
            np.concatenate(probs).flatten(),
            np.concatenate(truths).flatten())


def fit(model, ld_tr, ld_va, max_ep, lr, name, pw=1.0):
    """统一训练接口, 包含早停和余弦退火学习率

       pw: positive class weight, 补偿涨跌样本不均衡
    """
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 余弦退火 warm restart: 每 40 个 epoch 重置, 周期翻倍
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, 40, 2, eta_min=lr * 0.01)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    best_acc = 0.0
    best_weights = None
    no_improve = 0
    PATIENCE = 30

    for ep in range(1, max_ep + 1):
        train_loss, train_acc = train_epoch(model, ld_tr, opt, crit)
        val_loss, val_acc, _, _ = eval_torch(model, ld_va, crit)
        sch.step(ep + max_ep * 0.01)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [{name}] 早停 @ epoch {ep} "
                      f"(best_val_acc={best_acc:.3f})")
                break
        if ep == 1 or ep % 40 == 0:
            print(f"  [{name}] E{ep:3d} | "
                  f"TL={train_loss:.4f} TA={train_acc:.3f} | "
                  f"VL={val_loss:.4f} VA={val_acc:.3f}")

    model.load_state_dict(best_weights)
    return model


def calc_metrics(y_true, y_prob):
    """计算五项分类指标"""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_true, y_prob),
        "y_true":    y_true,
        "y_prob":    y_prob,
        "y_pred":    y_pred,
    }


# ------------------- 6. 可视化 -------------------

def plot_all(results, xgb_m=None):
    """一次性绘制并保存全部五张分析图"""
    n = len(results)

    # --- 6.1 ROC 曲线 ---
    os.makedirs("result", exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    from matplotlib.cm import tab10
    colors = tab10(np.linspace(0, 1, max(n, 10)))
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        style = "--" if "[B]" in name else "-"    # 基线虚线, 改进实线
        ax1.plot(fpr, tpr, c=colors[i], lw=3, ls=style,
                 label=f"{name} AUC={res['ROC_AUC']:.4f}")
    ax1.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.4, label="Random 0.5")
    ax1.set_xlabel("FPR", fontsize=14)
    ax1.set_ylabel("TPR", fontsize=14)
    ax1.set_title("ROC 曲线", fontsize=16)
    ax1.legend(fontsize=11, loc="lower right")
    ax1.tick_params(labelsize=12)
    fig1.tight_layout()
    fig1.savefig("result/roc_curves.png", dpi=200)
    print("[Plot] result/roc_curves.png")

    # --- 6.2 混淆矩阵 ---
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5.5))
    axes2 = axes2.flatten() if n > 1 else [axes2]
    for ax, (name, res) in zip(axes2, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nAcc={res['Accuracy']:.3f}  "
                     f"F1={res['F1']:.3f}  AUC={res['ROC_AUC']:.3f}",
                     fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=18)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Down", "Up"], fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Down", "Up"], fontsize=12)
    for ax in axes2[len(results):]:
        ax.set_visible(False)
    fig2.tight_layout()
    fig2.savefig("result/confusion_matrices.png", dpi=200)
    print("[Plot] result/confusion_matrices.png")

    # --- 6.3 指标对比柱状图 ---
    fig3, ax3 = plt.subplots(figsize=(16, 7))
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(metric_names))
    width = 0.85 / n
    for i, (name, res) in enumerate(results.items()):
        values = [res[m] for m in metric_names]
        bars = ax3.bar(x + i * width, values, width, label=name, alpha=0.85)
        # 每根柱子上标注数值
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=6.5)
    ax3.set_xticks(x + width * (n - 1) / 2)
    ax3.set_xticklabels(metric_names, fontsize=12)
    ax3.set_ylim(0, 0.59)
    ax3.set_title("指标对比", fontsize=16)
    ax3.legend(fontsize=10, loc="upper right", ncol=3)
    ax3.tick_params(labelsize=11)
    fig3.tight_layout()
    fig3.savefig("result/metrics_comparison.png", dpi=200)
    print("[Plot] result/metrics_comparison.png")

    # --- 6.4 预测概率分布 ---
    show = dict(list(results.items())[:6])
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
    axes4 = axes4.flatten()
    for ax, (name, res) in zip(axes4, show.items()):
        # 红色: 真实的涨; 蓝色: 真实的跌
        ax.hist(res["y_prob"][res["y_true"] == 1],
                bins=25, alpha=0.6, label="Up", color="#e74c3c", density=True)
        ax.hist(res["y_prob"][res["y_true"] == 0],
                bins=25, alpha=0.6, label="Down", color="#3498db", density=True)
        ax.set_title(f"{name}  AUC={res['ROC_AUC']:.4f}", fontsize=12)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)
    for ax in axes4[len(show):]:
        ax.set_visible(False)
    fig4.tight_layout()
    fig4.savefig("result/prob_distributions.png", dpi=200)
    print("[Plot] result/prob_distributions.png")

    # --- 6.5 特征重要性 (XGBoost Gain) ---
    importance = None
    if xgb_m is not None and hasattr(xgb_m, "feature_importances_"):
        importance = xgb_m.feature_importances_
    elif any("RF" in k for k in results):
        for k, v in results.items():
            if v.get("model") and hasattr(v["model"], "feature_importances_"):
                importance = v["model"].feature_importances_
                break
    if importance is not None:
        fig5, ax5 = plt.subplots(figsize=(14, 10))
        nf = min(len(importance), len(FEATS))
        order = np.argsort(importance[:nf])[::-1]    # 降序
        ax5.barh(range(nf), importance[order],
                 color=["#2ecc71" if i < nf // 2 else "#e74c3c"
                        for i in range(nf)])
        ax5.set_yticks(range(nf))
        ax5.set_yticklabels([FEATS[i] for i in order], fontsize=10)
        ax5.set_xlabel("Importance", fontsize=14)
        ax5.set_title("特征重要性 --- XGBoost", fontsize=16)
        ax5.tick_params(labelsize=11)
        ax5.invert_yaxis()                           # 最重要的在上方
        fig5.tight_layout()
        fig5.savefig("result/feature_importance.png", dpi=200)
        print("[Plot] result/feature_importance.png")

    plt.close("all")


# ------------------- 7. 主流程 -------------------

def main():
    print("=" * 60)
    print("  SC2616025 — 股票涨跌二分类: 对照 vs 改进")
    print("=" * 60)

    # 加载数据和特征工程
    df = fetch_data()
    df = add_features(df)
    print(f"[Info] 标签: 涨={df['Target'].sum()}  "
          f"跌={len(df) - df['Target'].sum()}  "
          f"(涨比={df['Target'].mean():.1%})")

    # 准备扁平数据和序列数据
    Xt, yt, Xv, yv, Xe, ye = prep_flat(df)
    Xts, yts, Xvs, yvs, Xes, yes = prep_seq(df)
    print(f"[Split] Flat: T={len(Xt)} V={len(Xv)} E={len(Xe)} | "
          f"Seq: T={len(Xts)} V={len(Xvs)} E={len(Xes)} "
          f"dim={Xts.shape[2]}")

    # 快速制作 PyTorch DataLoader 的辅助函数
    def _make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te):
        tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                               torch.tensor(y_tr, dtype=torch.float32))
        va_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                               torch.tensor(y_va, dtype=torch.float32))
        te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                               torch.tensor(y_te, dtype=torch.float32))
        tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, drop_last=True)
        va_ld = DataLoader(va_ds, BATCH * 2)
        te_ld = DataLoader(te_ds, BATCH * 2)
        return tr_ld, va_ld, te_ld

    ldf, ldvf, ldef = _make_loaders(Xt, yt, Xv, yv, Xe, ye)
    lds, ldvs, ldes = _make_loaders(Xts, yts, Xvs, yvs, Xes, yes)

    results = {}

    # ===================== 对照组 =====================

    # 1. 逻辑回归 (线性基线)
    print(f"\n{'─'*50}\n  [B] 1-LR (逻辑回归)\n{'─'*50}")
    lr = LogisticRegression(max_iter=3000, C=0.1,
                            class_weight="balanced", random_state=SEED)
    lr.fit(Xt, yt)
    results["[B] 1-LR"] = calc_metrics(ye, lr.predict_proba(Xe)[:, 1])
    print(f"  AUC={results['[B] 1-LR']['ROC_AUC']:.4f}  "
          f"Acc={results['[B] 1-LR']['Accuracy']:.4f}")

    # 2. 随机森林 (Bagging 集成基线)
    print(f"\n{'─'*50}\n  [B] 2-RF (随机森林)\n{'─'*50}")
    rf = RandomForestClassifier(200, max_depth=8, min_samples_leaf=8,
                                class_weight="balanced", random_state=SEED,
                                n_jobs=-1)
    rf.fit(Xt, yt)
    results["[B] 2-RF"] = calc_metrics(ye, rf.predict_proba(Xe)[:, 1])
    results["[B] 2-RF"]["model"] = rf       # 留着给特征重要性用
    print(f"  AUC={results['[B] 2-RF']['ROC_AUC']:.4f}  "
          f"Acc={results['[B] 2-RF']['Accuracy']:.4f}")

    # 3. 多层感知机 (前馈网络基线)
    print(f"\n{'─'*50}\n  [B] 3-MLP (前馈网络)\n{'─'*50}")
    mlp = fit(MLP(Xt.shape[1]), ldf, ldvf, EPOCHS, LR, "B-MLP")
    _, _, mlp_prob, mlp_true = eval_torch(mlp, ldef, nn.BCEWithLogitsLoss())
    results["[B] 3-MLP"] = calc_metrics(mlp_true, mlp_prob)
    print(f"  AUC={results['[B] 3-MLP']['ROC_AUC']:.4f}  "
          f"Acc={results['[B] 3-MLP']['Accuracy']:.4f}")

    # 4. 单向 LSTM (序列模型基线)
    print(f"\n{'─'*50}\n  [B] 4-LSTM\n{'─'*50}")
    lstm = fit(LSTM_(Xts.shape[2]), lds, ldvs, EPOCHS, LR, "B-LSTM")
    _, _, lstm_prob, lstm_true = eval_torch(lstm, ldes, nn.BCEWithLogitsLoss())
    results["[B] 4-LSTM"] = calc_metrics(lstm_true, lstm_prob)
    print(f"  AUC={results['[B] 4-LSTM']['ROC_AUC']:.4f}  "
          f"Acc={results['[B] 4-LSTM']['Accuracy']:.4f}")

    # ===================== 改进组 =====================

    # 正样本权重: 跌多涨少, 给涨加点权重
    pos_weight = (len(yt) - yt.sum()) / max(yt.sum(), 1)
    print(f"\n[Improved] pos_weight={pos_weight:.2f}")

    # 5. XGBoost (梯度提升, Boosting 替代 Bagging)
    print(f"\n{'─'*50}\n  ★ 5-XGBoost (梯度提升树)\n{'─'*50}")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.02,
        min_child_weight=1, gamma=0.0,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,      # L1 + L2 正则
        scale_pos_weight=1.0,
        eval_metric="logloss",
        random_state=SEED, n_jobs=-1, verbosity=0)
    xgb_model.fit(Xt, yt, verbose=False)
    results["★ 5-XGBoost"] = calc_metrics(
        ye, xgb_model.predict_proba(Xe)[:, 1])
    print(f"  AUC={results['★ 5-XGBoost']['ROC_AUC']:.4f}  "
          f"Acc={results['★ 5-XGBoost']['Accuracy']:.4f}  "
          f"n_est={xgb_model.n_estimators}")

    # 6. AttnLSTM (双向 + 自注意力 + 可学习查询池化)
    print(f"\n{'─'*50}\n  ★ 6-AttnLSTM (自注意力 LSTM)\n{'─'*50}")
    alstm = fit(AttnLSTM(Xts.shape[2]), lds, ldvs,
                EPOCHS, LR, "AttnLSTM", pw=pos_weight)
    _, _, attn_prob, attn_true = eval_torch(
        alstm, ldes, nn.BCEWithLogitsLoss())
    results["★ 6-AttnLSTM"] = calc_metrics(attn_true, attn_prob)
    print(f"  AUC={results['★ 6-AttnLSTM']['ROC_AUC']:.4f}  "
          f"Acc={results['★ 6-AttnLSTM']['Accuracy']:.4f}")

    # ===================== 汇总 =====================

    print(f"\n{'='*85}")
    print(f"{'Model':<24} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
          f"{'F1':>8} {'AUC':>8}  组别")
    print(f"{'-'*78}")
    for name, res in results.items():
        group = "对照" if "[B]" in name else "改进★"
        print(f"{name:<24} {res['Accuracy']:>8.4f} {res['Precision']:>8.4f} "
              f"{res['Recall']:>8.4f} {res['F1']:>8.4f} "
              f"{res['ROC_AUC']:>8.4f}  {group}")
    print(f"{'='*85}")

    # 最优模型
    best_name = max(results, key=lambda n: results[n]["ROC_AUC"])
    print(f"\n{'='*60}\n  🏆 {best_name}\n"
          f"  AUC={results[best_name]['ROC_AUC']:.4f}  "
          f"Acc={results[best_name]['Accuracy']:.4f}  "
          f"F1={results[best_name]['F1']:.4f}\n{'='*60}")
    print(classification_report(results[best_name]["y_true"],
                                results[best_name]["y_pred"],
                                target_names=["Down", "Up"], zero_division=0))

    # 对照 vs 改进汇总
    control_auc = [v["ROC_AUC"] for k, v in results.items() if "[B]" in k]
    improved_auc = [v["ROC_AUC"] for k, v in results.items() if "★" in k]
    print(f"\n[Summary] 对照组 AUC 均值: {np.mean(control_auc):.4f}  "
          f"(best: {max(control_auc):.4f})")
    print(f"[Summary] 改进组 AUC 均值: {np.mean(improved_auc):.4f}  "
          f"(best: {max(improved_auc):.4f})")
    if np.mean(control_auc) > 0:
        uplift = (np.mean(improved_auc) - np.mean(control_auc)) \
                 / np.mean(control_auc) * 100
        print(f"[Summary] 改进幅度: {uplift:+.1f}%")

    # 输出全部图表
    plot_all(results, xgb_model)
    print(f"\n{'='*60}\n"
          f"  完成! 图表已保存至 result/ 文件夹\n"
          f"{'='*60}")


if __name__ == "__main__":
    main()
