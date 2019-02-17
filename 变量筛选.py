
# coding: utf-8

# In[ ]:


# 变量筛选 

# xgboost筛选变量 
def select_xgboost(df,target,imp_num=None):
    """
    df:数据集
    target:目标变量的字段名
    imp_num:筛选变量的个数
    
    return:
    xg_fea_imp:变量的特征重要性
    xg_select_col:筛选出的变量
    """
    x = df.drop([target],axis=1)
    y = df[target]
    xgmodel = XGBClassifier(random_state=0)
    xgmodel = xgmodel.fit(x,y,eval_metric='auc')
    xg_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':xgmodel.feature_importances_})
    xg_fea_imp = xg_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    xg_select_col = list(xg_fea_imp.col)
    return xg_fea_imp,xg_select_col


# 随机森林筛选变量 
def select_rf(df,target,imp_num=None):
    """
    df:数据集
    target:目标变量的字段名
    imp_num:筛选变量的个数
    
    return:
    rf_fea_imp:变量的特征重要性
    rf_select_col:筛选出的变量
    """
    x = df.drop([target],axis=1)
    y = df[target]
    rfmodel = RandomForestClassifier(random_state=0)
    rfmodel = rfmodel.fit(x,y)
    rf_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':rfmodel.feature_importances_})
    rf_fea_imp = rf_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    rf_select_col = list(rf_fea_imp.col)
    return rf_fea_imp,rf_select_col


# 相关性可视化
def plot_corr(df,col_list,threshold=None,plt_size=None,is_annot=True):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    plt_size:图纸尺寸
    is_annot:是否显示相关系数值
    
    return :相关性热力图
    """
    corr_df = df.loc[:,col_list].corr()
    plt.figure(figsize=plt_size)
    sns.heatmap(corr_df,annot=is_annot,cmap='rainbow',vmax=1,vmin=-1,mask=np.abs(corr_df)<=threshold)
    return plt.show()


# 相关性剔除
def forward_delete_corr(df,col_list,threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    
    return:相关性剔除后的变量
    """
    list_corr = col_list[:]
    for col in list_corr:
        corr = df.loc[:,list_corr].corr()[col]
        corr_index= [x for x in corr.index if x!=col]
        corr_values  = [x for x in corr.values if x!=1]
        for i,j in zip(corr_index,corr_values):
            if abs(j)>=threshold:
                list_corr.remove(i)
    return list_corr


# 相关性变量映射关系 
def corr_mapping(df,col_list,threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    
    return:强相关性变量之间的映射关系表
    """
    corr_df = df.loc[:,col_list].corr()
    col_a = []
    col_b = []
    corr_value = []
    for col,i in zip(col_list[:-1],range(1,len(col_list),1)):
        high_corr_col=[]
        high_corr_value=[]
        corr_series = corr_df[col][i:]
        for i,j in zip(corr_series.index,corr_series.values):
            if abs(j)>=threshold:
                high_corr_col.append(i)
                high_corr_value.append(j)
        col_a.extend([col]*len(high_corr_col))
        col_b.extend(high_corr_col)
        corr_value.extend(high_corr_value)

    corr_map_df = pd.DataFrame({'col_A':col_a,
                                'col_B':col_b,
                                'corr':corr_value})
    return corr_map_df


# 显著性筛选,在筛选前需要做woe转换
def forward_delete_pvalue(x_train,y_train):
    """
    x_train -- x训练集
    y_train -- y训练集
    
    return :显著性筛选后的变量
    """
    col_list = list(x_train.columns)
    pvalues_col=[]
    for col in col_list:
        pvalues_col.append(col)
        x_train2 = sm.add_constant(x_train.loc[:,pvalues_col])
        sm_lr = sm.Logit(y_train,x_train2)
        sm_lr = sm_lr.fit()
        for i,j in zip(sm_lr.pvalues.index[1:],sm_lr.pvalues.values[1:]): 
            if j>=0.05:
                pvalues_col.remove(i)
    
    x_new_train = x_train.loc[:,pvalues_col]
    x_new_train2 = sm.add_constant(x_new_train)
    lr = sm.Logit(y_train,x_new_train2)
    lr = lr.fit()
    print(lr.summary2())
    return pvalues_col


# 逻辑回归系数符号筛选,在筛选前需要做woe转换
def forward_delete_coef(x_train,y_train):
    """
    x_train -- x训练集
    y_train -- y训练集
    
    return :
    coef_col回归系数符号筛选后的变量
    lr_coe：每个变量的系数值
    """
    col_list = list(x_train.columns)
    coef_col = []
    for col in col_list:
        coef_col.append(col)
        x_train2 = x_train.loc[:,coef_col]
        sk_lr = LogisticRegression(random_state=0).fit(x_train2,y_train)
        coef_df = pd.DataFrame({'col':coef_col,'coef':sk_lr.coef_[0]})
        if coef_df[coef_df.coef<0].shape[0]>0:
            coef_col.remove(col)
    
    x_new_train = x_train.loc[:,coef_col]
    lr = LogisticRegression(random_state=0).fit(x_new_train,y_train)
    lr_coe = pd.DataFrame({'col':coef_col,
                           'coef':lr.coef_[0]})
    return coef_col,lr_coe

