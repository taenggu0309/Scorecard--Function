
# coding: utf-8

# In[ ]:


# 变量woe离散化

# 变量woe结果表
def woe_df_concat(bin_df):
    """
    bin_df:list形式，里面存储每个变量的分箱结果
    
    return :woe结果表
    """
    woe_df_list =[]
    for df in bin_df:
        woe_df = df.reset_index().assign(col=df.index.name).rename(columns={df.index.name:'bin'})
        woe_df_list.append(woe_df)
    woe_result = pd.concat(woe_df_list,axis=0)
    # 为了便于查看，将字段名列移到第一列的位置上
    woe_result1 = woe_result['col']
    woe_result2 = woe_result.iloc[:,:-1]
    woe_result_df = pd.concat([woe_result1,woe_result2],axis=1)
    woe_result_df = woe_result_df.reset_index(drop=True)
    return woe_result_df

# woe转换
def woe_transform(df,target,df_woe):
    """
    df:数据集
    target:目标变量的字段名
    df_woe:woe结果表
    
    return:woe转化之后的数据集
    """
    df2 = df.copy()
    for col in df2.drop([target],axis=1).columns:
        x = df2[col]
        bin_map = df_woe[df_woe.col==col]
        bin_res = np.array([0]*x.shape[0],dtype=float)
        for i in bin_map.index:
            lower = bin_map['min_bin'][i]
            upper = bin_map['max_bin'][i]
            if lower == upper:
                x1 = x[np.where(x == lower)[0]]
            else:
                x1 = x[np.where((x>=lower)&(x<=upper))[0]]
            mask = np.in1d(x,x1)
            bin_res[mask] = bin_map['woe'][i]
        bin_res = pd.Series(bin_res,index=x.index)
        bin_res.name = x.name
        df2[col] = bin_res
    return df2

