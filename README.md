## 评分卡模型实现函数模块


author: yuxinxin
</br>modify_date: 2018-11-27

使用方法：将 score_card.py 放在与notebook文件同个目录下，在notebook文件里输入:import score_card as sc 即可调用里面的函数

## 函数目录：
### 一. EDA分析.py
1. 变量的分布（可视化）
* plot_cate_var -- 类别型变量分布
* plot_num_col  -- 数值型变量分布
2. 变量的违约率分析（可视化）：
* plot_default_cate -- 类别型变量的违约率分析
* plot_default_num  -- 数值型变量的违约率分析

### 二. 数据预处理.py
1. 缺失值处理
* missing_cal       -- 计算每个变量的缺失率
* plot_missing_var  -- 所有变量缺失值分布图
* plot_missing_user -- 单个样本的缺失分析
* missing_delete_var -- 缺失值剔除（针对单个变量）
* missing_delete_user -- 缺失值剔除（针对单个样本）
* fillna_cate_var   -- 缺失值填充（类别型变量）
* fillna_num_var    -- 缺失值填充（数值型变量）
2. 常变量/同值化处理
* const_delete -- 常变量/同值化处理
3. 降基处理
* descending_cate -- 类别型变量的降基处理

### 三.变量分箱.py
* binning_cate  -- 类别型变量的分箱
* iv_cate       -- 类别型变量的IV明细表
* binning_num   -- 数值型变量的分箱（使用卡方分箱）
* iv_num        -- 数值型变量的IV明细表
* binning_self  -- 自定义分箱
* plot_woe     -- 变量woe的可视化
* woe_monoton  -- 检验变量的woe是否呈单调变化
* woe_large    -- 检验变量某个箱的woe是否过大(大于1),PS:箱体的woe在（-1,1）较合理


### 四.变量筛选.py
* select_xgboost  -- xgboost筛选变量
* select_rf       -- 随机森林筛选变量
* plot_corr       -- 变量相关性可视化
* corr_mapping    -- 变量强相关性映射
* forward_delete_corr -- 逐个剔除相关性高的变量
* forward_delete_pvalue -- 显著性筛选（向前选择法）
* forward_delete_coef   -- 逻辑回归系数符号筛选（每个变量的系数符号需要一致）

### 五.变量woe离散化.py
* woe_df_concat -- 变量woe结果明细表
* woe_transform -- 变量woe转换

### 六.模型评估.py
* plot_roc -- 绘制ROC曲线
* plot_model_ks -- 绘制模型的KS曲线
* plot_learning_curve -- 绘制学习曲线
* cross_verify -- 交叉验证
* plot_matrix_report -- 混淆矩阵/分类结果报告

### 七.评分卡实现和评估.py
* cal_scale -- 评分卡刻度
* score_df_concat -- 变量score的明细表
* score_transform -- 变量score转换
* plot_score_ks -- 绘制评分卡的KS曲线
* plot_PR -- PR曲线
* plot_score_hist -- 好坏用户得分分布图
* score_info -- 得分明细表
* plot_lifting -- 绘制提升图和洛伦兹曲线
* rule_verify -- 设定cutoff点，计算衡量指标

### 八.评分卡监控.py
* score_psi -- 计算评分的PSI
* plot_score_compare -- 评分对比图
* var_stable -- 变量稳定性分析
* plot_var_shift -- 变量偏移分析
