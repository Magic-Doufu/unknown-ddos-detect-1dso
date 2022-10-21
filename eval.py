ds_lst = [None, 'Wednesday-WorkingHours',
                'Friday-WorkingHours-Afternoon-DDoS',
                'syn_data',
                'DrDoS_NTP',
                'DrDoS_UDP',
                'DrDoS_LDAP',
                'DrDoS_SSDP',
                'DrDoS_MSSQL',
                'DrDoS_DNS',
                'DrDoS_NetBIOS',
                'DrDoS_SNMP']
fl_lst = [None, 'BENIGN',
                'DDoS',
                'Syn',
                'DrDoS_NTP',
                'DrDoS_UDP',
                'DrDoS_LDAP',
                'DrDoS_SSDP',
                'DrDoS_MSSQL',
                'DrDoS_DNS',
                'DrDoS_NetBIOS',
                'DrDoS_SNMP']
close_set_record = True
import utils.util as util, matplotlib.pyplot as plt, numpy as np
import DHRTool, torch, argparse, time, os.path as bpath
from contextlib import redirect_stdout
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score
from sklearn import linear_model

def argParser():
    parser = argparse.ArgumentParser(description='選擇模型與資料集')
    parser.add_argument("-m", "--model",
                        help='選擇模型，如1或2。',
                        default=2,
                        type=int)
    # parser.add_argument("-exD", "--exdataset",
    #                     help=f'資料集選擇：{list(enumerate(ds_lst))}',
    #                     default=0,
    #                     type=int)
    # parser.add_argument('-exC', "--exclass", help="額外的類別")
    parser.add_argument('-s', '--seeds', default=[0,42,123,222,419,844,918,1344,65536,815149],
                        help='指定種子')
    parser.add_argument('-sp', '--savepath', default='./save_models',
                        help='模型儲存路徑')
    parser.add_argument('-lp', '--logpath', default='./logs',
                        help='Log儲存路徑')
    return parser

def getDataLoader(eth, flter=None):
    if eth < 3:
        dName = f'{ds_lst[eth]}.pcap_{dataver}'
        datatyp = datatype
    else:
        dName = f'{ds_lst[eth]}'
        datatyp = 'M2019'
    X, y = model_tool.datasetLoad(dataName=dName, dataType=datatyp, classFilter=flter)
    y = y.to_numpy()
    X_test, y_test = torch.Tensor(X), torch.Tensor(y) # transform to torch tensor
    ts_dataset = TensorDataset(X_test, y_test) # create your datset
    return DataLoader(ts_dataset, batch_size=model_tool.batch_size), dName # create your dataloader

from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
def create_sgd_ocsvm(nu=0.5, n_comp=20, tol=1e-7, gamma=0.1):
    clf = make_pipeline(
        Nystroem(gamma=gamma, random_state=_sd, n_components=n_comp, n_jobs=-1),
        linear_model.SGDOneClassSVM(
            nu=nu,
            shuffle=True,
            fit_intercept=True,
            random_state=_sd,
            tol=tol,
        )
    )
    return clf

#種子迴圈
def predictDatas(orgdata, unknowndata, dname, flter):
    fig_path = f'./{datatype}/{_sd}'
    model_path = f'{args.savepath}/{datatype}_sd_{_sd}.pth'
    # 檢查圖片存放路徑，如不存在就產生
    util.pathCreate(fig_path)

    # 固定種子
    util.fixed_seed(_sd)
    # 載入模型
    net, slcp_l = model_tool.modelGen(sl_w=1.0)
    net, slcp_l = model_tool.modelLoad(modelPath=model_path, model=net, slcpl=slcp_l)
    net, slcp_l = net.cuda(), slcp_l.cuda()

    # 評估閉集指標
    eval_org = model_tool.predictData(net, orgdata, slcp_l)
    eval_unknown = model_tool.predictData(net, unknowndata, slcp_l)

    if close_set_record:
        acc = accuracy_score(y_true=eval_org.groundtruth, y_pred=eval_org.predict)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true=eval_org.groundtruth, y_pred=eval_org.predict, average='micro')
        with open(f'{args.logpath}/{ds_lst[1]}.pcap_{dataver}.log','a') as fd:
            with redirect_stdout(fd):
                print(f'Test on:{datatype} Seed:{_sd}')
                print(acc, precision, recall, fscore)
        fig, ax = plt.subplots(figsize=(8,6))
        ConfusionMatrixDisplay.from_predictions(y_true=eval_org.groundtruth, y_pred=eval_org.predict, cmap='Blues', values_format='.3f', ax=ax, normalize='true', display_labels=model_tool.pre.sort_lst, xticks_rotation=-45)
        ax.set_title(f'Model: {datatype}     fileVer: {dataver}    ACC:{acc}')
        plt.tight_layout()
        plt.savefig(f'{fig_path}/Matrix_origin_{_sd}.png', transparent=False, facecolor='white')
        plt.close()

    # SSE Loss閾值確定
    thold = np.percentile(eval_org.reconst,[99])[0]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title(f'{dname} thold:{thold} outof 99%:{eval_unknown[eval_unknown.reconst > thold].shape[0] * 100 / eval_unknown.shape[0]}%')
    ax.set_ylabel('times')
    ax.set_xlabel('SSE loss')
    ax.hist(eval_org.reconst, 250, facecolor='g', alpha=0.5)
    ax.hist(eval_unknown.reconst , 250, facecolor='r', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{fig_path}/{dname}_{str(flter[0])}_{_sd}_SSE.png', transparent=False, facecolor='white')
    plt.close()
    nc_lst = [10, 50, 100, 150, 200]
    nu_lst = [0.1, 0.3, 0.5, 0.7, 0.9]
    gma_lst = [0.1, 0.5, 0.9]
    # OCSVM
    classes_lst = model_tool.pre.sort_lst
    print(f'Test on:{datatype} Seed:{_sd} Filter:{str(flter[0])}')
    path_lg = f'{args.logpath}/{datatype}_linear_{dname}_outlier.csv'
    if not bpath.exists(path_lg):
        with open(path_lg,'a') as fd:
            with redirect_stdout(fd):
                # print('filter, sd, nu, gamma, n_components, time, undect, dect(m), dect(u), sse_dect, dect_R')
                print('filter, sd, nu, time, undect, dect(m), dect(u), sse_dect, dect_R')
    # for n_c in nc_lst:
        # for _gamma in gma_lst:
    for _nu in nu_lst:
        start_time = time.time()
        svm_models = dict()
        for _ in range(len(classes_lst)):
            fltr = (eval_org.predict == _) & (eval_org.predict == eval_org.groundtruth)
            # print(pre.sort_lst[_], (eval_org[fltr]).shape[0])
            if (eval_org[fltr]).shape[0] < 1:
                continue
            _X = (eval_org.loc[fltr, ['o_x', 'o_y', 'o_z']]).to_numpy()
            # clf = create_sgd_ocsvm(nu=_nu, n_comp= 5 if (_X.shape[0] < 100) else n_c, gamma=_gamma)
            clf = linear_model.SGDOneClassSVM(nu = _nu, tol = 1e-7, random_state=_sd)
            clf.fit(_X)
            svm_models[classes_lst[_]] = clf
        # print('未知數量, 未知數量位於閾值內, 未知數量位於閾值外, 閾值外比例')
        total_outlier, total_dect, total_undect = 0, 0, 0
        for _ in range(len(classes_lst)):
            flt_thd = (eval_unknown.reconst <= thold)
            fltr_prd = (eval_unknown.predict == _)
            fltr = (eval_org.predict == _) & (eval_org.predict == eval_org.groundtruth)
            # 被判斷為該類的數量要大於1才能比對
            if (eval_unknown[fltr_prd]).shape[0] < 1:
                continue
            RSVM = svm_models.get(classes_lst[_])
            if RSVM == None:
                #表示訓練集沒有預測出該類，因此沒有SVM數據。
                #print(f'預測為 {classes_lst[_]} 類 共：{(eval_unknown[fltr_prd]).shape[0]}')
                pass
            try:
                tmp = eval_unknown.loc[flt_thd & fltr_prd, ['o_x', 'o_y', 'o_z']]
                # 預測所有該類在SSE閾值內的分數
                rawhtssc, htssc = RSVM.score_samples(eval_org.loc[fltr, ['o_x', 'o_y', 'o_z']].to_numpy()), RSVM.score_samples(tmp[~np.isnan(tmp).any(axis=1)].to_numpy())
            except:
                pass
            else:
                # 撈取SVM閾值
                L_lim, H_lim = np.percentile(rawhtssc,[0.5,99.5])
                # 測試集在閾值外的比例
                outsider_c = htssc[(htssc<L_lim) | (htssc>H_lim)].shape[0]
                # outsider_r = 100 * outsider_c / htssc.shape[0]
                # 類別, 未知數量, 未知數量位於閾值內, 未知數量位於閾值外, 閾值外比例
                # print(f'L_Lim:{L_lim}, H_lim:{H_lim}')
                # print(classes_lst[_], htssc.shape[0], htssc.shape[0] - outsider_c, outsider_c, outsider_r)
                total_outlier += outsider_c
                if (_ == 0):
                    total_undect += (htssc.shape[0] - outsider_c)
                else:
                    total_dect += htssc.shape[0]
        # print(f'檢測時間:{time.time() - start_time}s 未檢出:{total_undect}, 檢出(惡意){total_dect}, 檢出(未知){total_outlier}, SSE檢出{eval_unknown[eval_unknown.reconst > thold].shape[0]}, 總防禦率:{(eval_unknown.shape[0] - total_undect) / eval_unknown.shape[0]}')
        # filter, nu, gamma, n_components, time, 未檢出, 檢出(惡意), 檢出(未知), SSE檢出, 防禦率
        with open(path_lg,'a') as fd:
            with redirect_stdout(fd):
                # print(f'{str(flter[0])},{_sd},{_nu},{_gamma},{n_c},{time.time() - start_time},{total_undect},{total_dect},{total_outlier},{eval_unknown[eval_unknown.reconst > thold].shape[0]},{(eval_unknown.shape[0] - total_undect) / eval_unknown.shape[0]}')
                print(f'{str(flter[0])},{_sd},{_nu},{time.time() - start_time},{total_undect},{total_dect},{total_outlier},{eval_unknown[eval_unknown.reconst > thold].shape[0]},{(eval_unknown.shape[0] - total_undect) / eval_unknown.shape[0]}')

if __name__ == '__main__':
    parser = argParser()
    args = parser.parse_args()
    datatype = util.m_lst[args.model]
    dataver = util.ver_lst.get(datatype)
    seeds = args.seeds
    print('未知偵測部分，具有不確定性結果，但整體來說可以偵測出來。')
    #檢查Log存放路徑，如不存在就產生
    util.pathCreate(args.logpath)

    model_tool = DHRTool.DDoS_DHR()
    #載入訓練資料集
    orgTestDataloader, _n = getDataLoader(1)
    for _i in range(2, ds_lst.__len__()):
        #載入未知資料集
        # if _i < 11:
        #     continue
        for _ in range(1, _i + 1, _i - 1):
            # if (_i == 4) and (_ == 1):
            #     continue
            unknownTestDataloader, filename = getDataLoader(_i, [fl_lst[_]])
            for _sd in seeds:
                predictDatas(orgTestDataloader, unknownTestDataloader, filename, [fl_lst[_]])
            close_set_record = None