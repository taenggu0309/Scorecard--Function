
# coding: utf-8

# In[ ]:


# 模型评估

# AUC 
def plot_roc(y_label,y_pred):
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:ROC曲线
    """
    tpr,fpr,threshold = metrics.roc_curve(y_label,y_pred) 
    AUC = metrics.roc_auc_score(y_label,y_pred) 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tpr,fpr,color='blue',label='AUC=%.3f'%AUC) 
    ax.plot([0,1],[0,1],'r--')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title('ROC')
    ax.legend(loc='best')
    return plt.show(ax)


# KS 
def plot_model_ks(y_label,y_pred):
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:KS曲线
    """
    pred_list = list(y_pred) 
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list)-total_bad 
    items = sorted(zip(pred_list,label_list),key=lambda x:x[0]) 
    step = (max(pred_list)-min(pred_list))/200 
    
    pred_bin=[]
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201): 
        idx = min(pred_list)+i*step 
        pred_bin.append(idx) 
        label_bin = [x[1] for x in items if x[0]<idx] 
        bad_num = sum(label_bin)
        good_num = len(label_bin)-bad_num  
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(pred_bin,good_rate,color='green',label='good_rate')
    ax.plot(pred_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(pred_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)


# 交叉验证
def cross_verify(x,y,estimators,fold,scoring='roc_auc'):
    """
    x:自变量的数据集
    y:target的数据集
    estimators：验证的模型
    fold：交叉验证的策略
    scoring:评级指标，默认auc
    
    return:交叉验证的结果
    """
    cv_result = cross_val_score(estimator=estimators,X=x,y=y,cv=fold,n_jobs=-1,scoring=scoring)
    print('CV的最大AUC为:{}'.format(cv_result.max()))
    print('CV的最小AUC为:{}'.format(cv_result.min()))
    print('CV的平均AUC为:{}'.format(cv_result.mean()))
    plt.figure(figsize=(6,4))
    plt.title('交叉验证的评价指标分布图')
    plt.boxplot(cv_result,patch_artist=True,showmeans=True,
            boxprops={'color':'black','facecolor':'yellow'},
            meanprops={'marker':'D','markerfacecolor':'tomato'},
            flierprops={'marker':'o','markerfacecolor':'red','color':'black'},
            medianprops={'linestyle':'--','color':'orange'})
    return plt.show()


# 学习曲线
def plot_learning_curve(estimator,x,y,cv=None,train_size = np.linspace(0.1,1.0,5),plt_size =None):
    """
    estimator :画学习曲线的基模型
    x:自变量的数据集
    y:target的数据集
    cv:交叉验证的策略
    train_size:训练集划分的策略
    plt_size:画图尺寸
    
    return:学习曲线
    """
    from sklearn.model_selection import learning_curve
    train_sizes,train_scores,test_scores = learning_curve(estimator=estimator,
                                                          X=x,
                                                          y=y,
                                                          cv=cv,
                                                          n_jobs=-1,
                                                          train_sizes=train_size)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.figure(figsize=plt_size)
    plt.xlabel('Training-example')
    plt.ylabel('score')
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Training-score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='cross-val-score')
    plt.legend(loc='best')
    return plt.show()


# 混淆矩阵 /分类报告
def plot_matrix_report(y_label,y_pred): 
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:混淆矩阵
    """
    matrix_array = metrics.confusion_matrix(y_label,y_pred)
    plt.matshow(matrix_array, cmap=plt.cm.summer_r)
    plt.colorbar()

    for x in range(len(matrix_array)): 
        for y in range(len(matrix_array)):
            plt.annotate(matrix_array[x,y], xy =(x,y), ha='center',va='center')

    plt.xlabel('True label')
    plt.ylabel('Predict label')
    print(metrics.classification_report(y_label,y_pred))
    return plt.show()

