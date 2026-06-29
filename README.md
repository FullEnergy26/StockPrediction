## 快速开始

```bash
pip install numpy pandas matplotlib scikit-learn xgboost torch
python main.py
```

所有输出图片保存至 `result/` 目录。

## 输出图片说明

`roc_curves.png`：六种模型的 ROC 曲线图。横轴为假正率（FPR），纵轴为真正率（TPR）。虚线表示对照组模型，实线表示改进组模型，黑色点线表示随机猜测基线（AUC=0.50）。

`confusion_matrices.png`：六种模型的混淆矩阵热力图。对角线为正确预测数，非对角线为错误预测数，标题标注了准确率、F1 和 AUC。

`metrics_comparison.png`：五项评价指标的柱状对比图。横轴列出准确率、精确率、召回率、F1 和 AUC，每种模型以不同颜色区分，每根柱子标注了四位小数的具体数值。

`prob_distributions.png`：六种模型的预测概率分布图。每个子图叠加绘制真实上涨样本（红色）和真实下跌样本（蓝色）的预测概率直方图。两色分布明显分离时表示模型具备较好的区分能力，高度重叠时表示模型难以区分涨跌。

`feature_importance.png`：XGBoost 模型中二十六维特征的增益型重要性排序图。横轴为重要性得分，纵轴为特征名称，按降序排列。排名靠前的特征对模型决策贡献较大。

