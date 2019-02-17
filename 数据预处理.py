
# coding: utf-8

# In[ ]:


# 数据预处理

# 每个变量缺失率的计算
def missing_cal(df):
    """
    df :数据集
    
    return：每个变量的缺失率
    """
    missing_series = df.isnull().sum()/df.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index':'col',
                                            0:'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct',ascending=False).reset_index(drop=True)
    return missing_df

# 变量的缺失分布图
def plot_missing_var(df,plt_size=None):
    """
    df: 数据集
    plt_size :图纸的尺寸
    
    return: 缺失分布图（直方图形式)
    """
    missing_df = missing_cal(df)
    plt.figure(figsize=plt_size)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = missing_df['missing_pct']
    plt.hist(x=x,bins=np.arange(0,1.1,0.1),color='hotpink',ec='k',alpha=0.8)
    plt.ylabel('缺失值个数')
    plt.xlabel('缺失率')
    return plt.show()


# 单个样本的缺失分布
def plot_missing_user(df,plt_size=None):
    """
    df: 数据集
    plt_size: 图纸的尺寸
    
    return :缺失分布图（折线图形式）
    """
    missing_series = df.isnull().sum(axis=1)
    list_missing_num  = sorted(list(missing_series.values))
    plt.figure(figsize=plt_size)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(df.shape[0]),list_missing_num)
    plt.ylabel('缺失变量个数')
    plt.xlabel('sanples')
    return plt.show()


# 缺失值剔除（单个变量）
def missing_delete_var(df,threshold=None):
    """
    df:数据集
    threshold:缺失率删除的阈值
    
    return :删除缺失后的数据集
    """
    df2 = df.copy()
    missing_df = missing_cal(df)
    missing_col_num = missing_df[missing_df.missing_pct>=threshold].shape[0]
    missing_col = list(missing_df[missing_df.missing_pct>=threshold].col)
    df2 = df2.drop(missing_col,axis=1)
    print('缺失率超过{}的变量个数为{}'.format(threshold,missing_col_num))
    return df2


# 缺失值剔除（单个样本）
def missing_delete_user(df,threshold=None):
    """
    df:数据集
    threshold:缺失个数删除的阈值
    
    return :删除缺失后的数据集
    """
    df2 = df.copy()
    missing_series = df.isnull().sum(axis=1)
    missing_list = list(missing_series)
    missing_index_list = []
    for i,j in enumerate(missing_list):
        if j>=threshold:
            missing_index_list.append(i)
    df2 = df2[~(df2.index.isin(missing_index_list))]
    print('缺失变量个数在{}以上的用户数有{}个'.format(threshold,len(missing_index_list)))
    return df2


# 缺失值填充（类别型变量）
def fillna_cate_var(df,col_list,fill_type=None):
    """
    df:数据集
    col_list:变量list集合
    fill_type: 填充方式：众数/当做一个类别
    
    return :填充后的数据集
    """
    df2 = df.copy()
    for col in col_list:
        if fill_type=='class':
            df2[col] = df2[col].fillna('unknown')
        if fill_type=='mode':
            df2[col] = df2[col].fillna(df2[col].mode()[0])
    return df2


# 数值型变量的填充
# 针对缺失率在5%以下的变量用中位数填充
# 缺失率在5%--15%的变量用随机森林填充,可先对缺失率较低的变量先用中位数填充，在用没有缺失的样本来对变量作随机森林填充
# 缺失率超过15%的变量建议当做一个类别
def fillna_num_var(df,col_list,fill_type=None,filled_df=None):
    """
    df:数据集
    col_list:变量list集合
    fill_type:填充方式：中位数/随机森林/当做一个类别
    filled_df :已填充好的数据集，当填充方式为随机森林时 使用
    
    return:已填充好的数据集
    """
    df2 = df.copy()
    for col in col_list:
        if fill_type=='median':
            df2[col] = df2[col].fillna(df2[col].median())
        if fill_type=='class':
            df2[col] = df2[col].fillna(-999)
        if fill_type=='rf':
            rf_df = pd.concat([df2[col],filled_df],axis=1)
            known = rf_df[rf_df[col].notnull()]
            unknown = rf_df[rf_df[col].isnull()]
            x_train = known.drop([col],axis=1)
            y_train = known[col]
            x_pre = unknown.drop([col],axis=1)
            rf = RandomForestRegressor(random_state=0)
            rf.fit(x_train,y_train)
            y_pre = rf.predict(x_pre)
            df2.loc[df2[col].isnull(),col] = y_pre
    return df2


# 常变量/同值化处理
def const_delete(df,col_list,threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold:同值化处理的阈值
    
    return :处理后的数据集
    """
    df2 = df.copy()
    const_col = []
    for col in col_list:
        const_pct = df2[col].value_counts().iloc[0]/df2[df2[col].notnull()].shape[0]
        if const_pct>=threshold:
            const_col.append(col)
    df2 = df2.drop(const_col,axis=1)
    print('常变量/同值化处理的变量个数为{}'.format(len(const_col)))
    return df2


# 分类型变量的降基处理
def descending_cate(df,col_list,threshold=None):
    """
    df: 数据集
    col_list:变量list集合
    threshold:降基处理的阈值
    
    return :处理后的数据集
    """
    df2 = df.copy()
    for col in col_list:
        value_series = df[col].value_counts()/df[df[col].notnull()].shape[0]
        small_value = []
        for value_name,value_pct in zip(value_series.index,value_series.values):
            if value_pct<=threshold:
                small_value.append(value_name)
        df2.loc[df2[col].isin(small_value),col]='other'
    return df2

