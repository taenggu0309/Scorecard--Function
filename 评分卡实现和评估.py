
# coding: utf-8

# In[ ]:


# 评分卡实现

# 评分卡刻度 
def cal_scale(score,odds,PDO,model):
    """
    odds：设定的坏好比
    score:在这个odds下的分数
    PDO: 好坏翻倍比
    model:逻辑回归模型
    
    return :A,B,base_score
    """
    B = 20/(np.log(odds)-np.log(2*odds))
    A = score-B*np.log(odds)
    base_score = A+B*model.intercept_[0]
    print('B: {:.2f}'.format(B))
    print('A: {:.2f}'.format(A))
    print('基础分为：{:.2f}'.format(base_score))
    return A,B,base_score


# 变量得分表
def score_df_concat(woe_df,model,B):
    """
    woe_df: woe结果表
    model:逻辑回归模型
    
    return:变量得分结果表
    """
    coe = list(model.coef_[0])
    columns = list(woe_df.col.unique())
    scores=[]
    for c,col in zip(coe,columns):
        score=[]
        for w in list(woe_df[woe_df.col==col].woe):
            s = round(c*w*B,0)
            score.append(s)
        scores.extend(score)
    woe_df['score'] = scores
    score_df = woe_df.copy()
    return score_df


# 分数转换 
def score_transform(df,target,df_score):
    """
    df:数据集
    target:目标变量的字段名
    df_score:得分结果表
    
    return:得分转化之后的数据集
    """
    df2 = df.copy()
    for col in df2.drop([target],axis=1).columns:
        x = df2[col]
        bin_map = df_score[df_score.col==col]
        bin_res = np.array([0]*x.shape[0],dtype=float)
        for i in bin_map.index:
            lower = bin_map['min_bin'][i]
            upper = bin_map['max_bin'][i]
            if lower == upper:
                x1 = x[np.where(x == lower)[0]]
            else:
                x1 = x[np.where((x>=lower)&(x<=upper))[0]]
            mask = np.in1d(x,x1)
            bin_res[mask] = bin_map['score'][i]
        bin_res = pd.Series(bin_res,index=x.index)
        bin_res.name = x.name
        df2[col] = bin_res
    return df2


# 得分的KS 
def plot_score_ks(df,score_col,target):
    """
    df:数据集
    target:目标变量的字段名
    score_col:最终得分的字段名
    """
    total_bad = df[target].sum()
    total_good = df[target].count()-total_bad
    score_list = list(df[score_col])
    target_list = list(df[target])
    items = sorted(zip(score_list,target_list),key=lambda x:x[0]) 
    step = (max(score_list)-min(score_list))/200 
    
    score_bin=[] 
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201):
        idx = min(score_list)+i*step 
        score_bin.append(idx) 
        target_bin = [x[1] for x in items if x[0]<idx]  
        bad_num = sum(target_bin)
        good_num = len(target_bin)-bad_num 
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
        
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(score_bin,good_rate,color='green',label='good_rate')
    ax.plot(score_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(score_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)


# PR曲线
def plot_PR(df,score_col,target,plt_size=None):
    """
    df:得分的数据集
    score_col:分数的字段名
    target:目标变量的字段名
    plt_size:绘图尺寸
    
    return: PR曲线
    """
    total_bad = df[target].sum()
    score_list = list(df[score_col])
    target_list = list(df[target])
    score_unique_list = sorted(set(list(df[score_col])))
    items = sorted(zip(score_list,target_list),key=lambda x:x[0]) 

    precison_list = []
    tpr_list = []
    for score in score_unique_list:
        target_bin = [x[1] for x in items if x[0]<=score]  
        bad_num = sum(target_bin)
        total_num = len(target_bin)
        precison = bad_num/total_num
        tpr = bad_num/total_bad
        precison_list.append(precison)
        tpr_list.append(tpr)
    
    plt.figure(figsize=plt_size)
    plt.title('PR曲线')
    plt.xlabel('查全率')
    plt.ylabel('精确率')
    plt.plot(tpr_list,precison_list,color='tomato',label='PR曲线')
    plt.legend(loc='best')
    return plt.show()


# 得分分布图
def plot_score_hist(df,target,score_col,plt_size=None,cutoff=None):
    """
    df:数据集
    target:目标变量的字段名
    score_col:最终得分的字段名
    plt_size:图纸尺寸
    cutoff :划分拒绝/通过的点
    
    return :好坏用户的得分分布图
    """    
    plt.figure(figsize=plt_size)
    x1 = df[df[target]==1][score_col]
    x2 = df[df[target]==0][score_col]
    sns.kdeplot(x1,shade=True,label='坏用户',color='hotpink')
    sns.kdeplot(x2,shade=True,label='好用户',color ='seagreen')
    plt.axvline(x=cutoff)
    plt.legend()
    return plt.show()




# 得分明细表 
def score_info(df,score_col,target,x=None,y=None,step=None):
    """
    df:数据集
    target:目标变量的字段名
    score_col:最终得分的字段名
    x:最小区间的左值
    y:最大区间的右值
    step:区间的分数间隔
    
    return :得分明细表
    """
    df['score_bin'] = pd.cut(df[score_col],bins=np.arange(x,y,step),right=True)
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    
    group = df.groupby('score_bin')
    score_info_df = pd.DataFrame()
    score_info_df['用户数'] = group[target].count()
    score_info_df['坏用户'] = group[target].sum()
    score_info_df['好用户'] = score_info_df['用户数']-score_info_df['坏用户']
    score_info_df['违约占比'] = score_info_df['坏用户']/score_info_df['用户数']
    score_info_df['累计用户'] = score_info_df['用户数'].cumsum()
    score_info_df['坏用户累计'] = score_info_df['坏用户'].cumsum()
    score_info_df['好用户累计'] = score_info_df['好用户'].cumsum()
    score_info_df['坏用户累计占比'] = score_info_df['坏用户累计']/bad 
    score_info_df['好用户累计占比'] = score_info_df['好用户累计']/good
    score_info_df['累计用户占比'] = score_info_df['累计用户']/total 
    score_info_df['累计违约占比'] = score_info_df['坏用户累计']/score_info_df['累计用户']
    score_info_df = score_info_df.reset_index()
    return score_info_df


# 绘制提升图和洛伦兹曲线
def plot_lifting(df,score_col,target,bins=10,plt_size=None):
    """
    df:数据集，包含最终的得分
    score_col:最终分数的字段名
    target:目标变量名
    bins:分数划分成的等份数
    plt_size:绘图尺寸
    
    return:提升图和洛伦兹曲线
    """
    score_list = list(df[score_col])
    label_list = list(df[target])
    items = sorted(zip(score_list,label_list),key = lambda x:x[0])
    step = round(df.shape[0]/bins,0)
    bad = df[target].sum()
    all_badrate = float(1/bins)
    all_badrate_list = [all_badrate]*bins
    all_badrate_cum = list(np.cumsum(all_badrate_list))
    all_badrate_cum.insert(0,0)
    
    score_bin_list=[]
    bad_rate_list = []
    for i in range(0,bins,1):
        index_a = int(i*step)
        index_b = int((i+1)*step)
        score = [x[0] for x in items[index_a:index_b]]
        tup1 = (min(score),)
        tup2 = (max(score),)
        score_bin = tup1+tup2
        score_bin_list.append(score_bin)
        label_bin = [x[1] for x in items[index_a:index_b]]
        bin_bad = sum(label_bin)
        bin_bad_rate = bin_bad/bad
        bad_rate_list.append(bin_bad_rate)
    bad_rate_cumsum = list(np.cumsum(bad_rate_list))
    bad_rate_cumsum.insert(0,0)
    
    plt.figure(figsize=plt_size)
    x = score_bin_list
    y1 = bad_rate_list
    y2 = all_badrate_list
    y3 = bad_rate_cumsum
    y4 = all_badrate_cum
    plt.subplot(1,2,1)
    plt.title('提升图')
    plt.xticks(np.arange(bins)+0.15,x,rotation=90)
    bar_width= 0.3
    plt.bar(np.arange(bins),y1,width=bar_width,color='hotpink',label='score_card')
    plt.bar(np.arange(bins)+bar_width,y2,width=bar_width,color='seagreen',label='random')
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.title('洛伦兹曲线图')
    plt.plot(y3,color='hotpink',label='score_card')
    plt.plot(y4,color='seagreen',label='random')
    plt.xticks(np.arange(bins+1),rotation=0)
    plt.legend(loc='best')
    return plt.show()


# 设定cutoff点，衡量有效性
def rule_verify(df,col_score,target,cutoff):
    """
    df:数据集
    target:目标变量的字段名
    col_score:最终得分的字段名    
    cutoff :划分拒绝/通过的点
    
    return :混淆矩阵
    """
    df['result'] = df.apply(lambda x:30 if x[col_score]<=cutoff else 10,axis=1)
    TP = df[(df['result']==30)&(df[target]==1)].shape[0] 
    FN = df[(df['result']==30)&(df[target]==0)].shape[0] 
    bad = df[df[target]==1].shape[0] 
    good = df[df[target]==0].shape[0] 
    refuse = df[df['result']==30].shape[0] 
    passed = df[df['result']==10].shape[0] 
    
    acc = round(TP/refuse,3) 
    tpr = round(TP/bad,3) 
    fpr = round(FN/good,3) 
    pass_rate = round(refuse/df.shape[0],3) 
    matrix_df = pd.pivot_table(df,index='result',columns=target,aggfunc={col_score:pd.Series.count},values=col_score) 
    
    print('精确率:{}'.format(acc))
    print('查全率:{}'.format(tpr))
    print('误伤率:{}'.format(fpr))
    print('规则拒绝率:{}'.format(pass_rate))
    return matrix_df

