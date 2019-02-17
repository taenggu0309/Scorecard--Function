
# coding: utf-8

# In[ ]:


# 绘制变量的得分占比偏移图
def plot_var_shift(df,day_col,score_col,plt_size=None):
    """
    df:变量在一段时间内，每个区间上的得分
    day_col:时间的字段名（天）
    score_col:得分的字段名
    plt_size: 绘图尺寸
    
    return:变量区间得分的偏移图
    """
    day_list = sorted(set(list(df[day_col]))) 
    score_list = sorted(set(list(df[score_col])))
    # 计算每天各个区间得分的占比
    prop_day_list = []
    for day in day_list:
        prop_list = []
        for score in score_list:
            prop = df[(df[day_col]==day)&(df[score_col]==score)].shape[0]/df[df[day_col]==day].shape[0]
            prop_list.append(prop)
        prop_day_list.append(prop_list)
    
    # 将得分占比的转化为画图的格式
    sub_list = []
    for p in prop_day_list:
        p_cumsum = list(np.cumsum(p))
        p_cumsum = p_cumsum[:-1]
        p_cumsum.insert(0,0)
        bar1_list = [1]*int(len(p_cumsum))
        sub = [bar1_list[i]-p_cumsum[i] for i in range(len(p_cumsum))]
        sub_list.append(sub)
    array = np.array(sub_list)
    
    stack_prop_list = [] # 面积图的y值
    bar_prop_list = [] # 堆积柱状图的y
    for i in range(len(score_list)):
        bar_prop = array[:,i]
        bar_prop_list.append(bar_prop)
        stack_prop = []
        for j in bar_prop:
            a = j
            b = j
            stack_prop.append(a)
            stack_prop.append(b)
        stack_prop_list.append(stack_prop)
    
    # 画图的x坐标轴
    x_bar = list(range(1,len(day_list)*2,2)) # 堆积柱状图的x值
    x_stack = []    # 面积图的x值
    for i in x_bar:
        c = i-0.5
        d = i+0.5
        x_stack.append(c)
        x_stack.append(d)
    
    # 绘图
    fig = plt.figure(figsize=plt_size)
    ax1 = fig.add_subplot(1,1,1)
    # 先清除x轴的刻度
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(''.format)) 
    ax1.set_xticks(range(1,len(day_list)*2,2))
    # 将y轴的刻度设置为百分比形式
    def to_percent(temp, position):
        return '%1.0f'%(100*temp) + '%'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(to_percent))
    # 自定义x轴刻度标签
    for a,b in zip(x_bar,day_list): 
        ax1.text(a,-0.08,b,ha='center',va='bottom')
    # 绘制面积图和堆积柱状图
    for i,s in zip(range(len(day_list)),score_list):
        ax1.stackplot(x_stack,stack_prop_list[i],alpha=0.25)
        ax1.bar(x_bar,bar_prop_list[i],width=1,label='得分:{}'.format(s))
        # 添加y轴刻度虚线
        ax1.grid(True, 'major', 'y', ls='--', lw=.5, c='black', alpha=.3)
        ax1.legend(loc='best')
    plt.show()

    
# 计算评分的PSI
def score_psi(df1,df2,id_col,score_col,x,y,step=None):
    """
    df1:建模样本的得分,包含用户id,得分
    df2:上线样本的得分，包含用户id，得分
    id_col:用户id字段名
    score_col:得分的字段名
    x:划分得分区间的left值
    y:划分得分区间的right值
    step:步长
    
    return: 得分psi表
    """
    df1['score_bin'] = pd.cut(df1[score_col],bins=np.arange(x,y,step))
    model_score_group = df1.groupby('score_bin',as_index=False)[id_col].count().                           assign(pct=lambda x:x[id_col]/x[id_col].sum()).                           rename(columns={id_col:'建模样本户数',
                                           'pct':'建模户数占比'})
    df2['score_bin'] = pd.cut(df2[score_col],bins=np.arange(x,y,step))
    online_score_group = df2.groupby('score_bin',as_index=False)[id_col].count().                           assign(pct=lambda x:x[id_col]/x[id_col].sum()).                           rename(columns={id_col:'线上样本户数',
                                           'pct':'线上户数占比'})
    score_compare = pd.merge(model_score_group,online_score_group,on='score_bin',how='inner')
    score_compare['占比差异'] = score_compare['线上户数占比'] - score_compare['建模户数占比']
    score_compare['占比权重'] = np.log(score_compare['线上户数占比']/score_compare['建模户数占比'])
    score_compare['Index']= score_compare['占比差异']*score_compare['占比权重']
    score_compare['PSI'] = score_compare['Index'].sum()
    return score_compare


# 评分比较分布图
def plot_score_compare(df,plt_size=None):
    fig = plt.figure(figsize=plt_size)
    x = df.score_bin
    y1 = df.建模户数占比
    y2 = df.线上户数占比
    width=0.3
    plt.title('评分分布对比图')
    plt.xlabel('得分区间')
    plt.ylabel('用户占比')
    plt.xticks(np.arange(len(x))+0.15,x)
    plt.bar(np.arange(len(y1)),y1,width=width,color='seagreen',label='建模样本')
    plt.bar(np.arange(len(y2))+width,y2,width=width,color='hotpink',label='上线样本')
    plt.legend()
    return plt.show() 


# 变量稳定度分析
def var_stable(score_result,df,var,id_col,score_col,bins):
    """
    score_result:评分卡的score明细表，包含区间，用户数，用户占比,得分
    var：分析的变量名
    df:上线样本变量的得分，包含用户id,变量的value，变量的score
    id_col:df的用户id字段名
    score_col:df的得分字段名
    bins:变量划分的区间
    
    return :变量的稳定性分析表
    """
    model_var_group = score_result.loc[score_result.col==var,                      ['bin','total','totalrate','score']].reset_index(drop=True).                      rename(columns={'total':'建模用户数',
                                      'totalrate':'建模用户占比',
                                      'score':'得分'})
    df['bin'] = pd.cut(df[score_col],bins=bins)
    online_var_group = df.groupby('bin',as_index=False)[id_col].count()                         .assign(pct=lambda x:x[id_col]/x[id_col].sum())                         .rename(columns={id_col:'线上用户数',
                                          'pct':'线上用户占比'})
    var_stable_df = pd.merge(model_var_group,online_var_group,on='bin',how='inner')
    var_stable_df = var_stable_df.iloc[:,[0,3,1,2,4,5]]
    var_stable_df['得分'] = var_stable_df['得分'].astype('int64')
    var_stable_df['建模样本权重'] = np.abs(var_stable_df['得分']*var_stable_df['建模用户占比'])
    var_stable_df['线上样本权重'] = np.abs(var_stable_df['得分']*var_stable_df['线上用户占比'])
    var_stable_df['权重差距'] = var_stable_df['线上样本权重'] - var_stable_df['建模样本权重']
    return var_stable_df

