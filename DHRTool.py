import os, torch, pandas as pd, utils.model.DHRNet as models, numpy as np 
from utils.preprocessREVI import preprocessor
import torch.nn as nn
from utils.losses.SLCPLoss import SLCPLoss

from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn import linear_model
import itertools, matplotlib.pyplot as plt

class DDoS_DHR:
    def __init__(self, dataset_prefix='datasets', model_prefix='save_models', batch_size=256, model_type='M1') -> None:
        self.modelType = model_type
        self.prefix = dataset_prefix
        self.batch_size = batch_size
        self.model_prefix = model_prefix
        self.datashape0=0
        self.pre = preprocessor()
    def datasetLoad(self, dataName, classFilter=None, dataType='M1'):
        # print(classFilter, dataType)
        print(f'載入資料集:{dataName}')
        df0, dt0, self.columns = self.pre.data_preprocess(f'./{self.prefix}/{dataName}.csv', mn=dataType, filter_l=classFilter)
        df0 = self.pre.data_reshape(reshape_data=df0, reshape_cnn='1d')
        self.datashape0 = dt0.columns.__len__()
        return df0, dt0
    """
        產模型
    """
    def modelGen(self, latent=3, sl_w=0.1):
        print(f'產出模型 瓶頸層:{latent}, SLCPL權重:{sl_w}')
        net = models.DHRNet(self.datashape0, latent=latent)
        slcp_l = SLCPLoss(self.datashape0, latent, weight=sl_w)
        return net, slcp_l
    """
        讀模型
    """
    def modelLoad(self, modelPath, model, slcpl):
        print(f'讀取模型 {modelPath}')
        cp_metas = torch.load(modelPath,map_location="cpu")
        model.load_state_dict(cp_metas.get('model_state_dict'))
        slcpl.points = torch.nn.Parameter(torch.from_numpy(cp_metas['centers']).float())
        return model, slcpl
    """
        驗證函數
    """
    def predictData(self, net, testloader, slcp_l):
        net.eval()
        reconst_criterion = torch.nn.MSELoss(reduction='none')
        pos_lst, grounTruth_lst, SSE_lst, predict_lst = np.empty((0,3)), np.empty((0)), np.empty((0)), np.empty((0))
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                _, targets = torch.max(labels.data, 1)
                _, cf, _r, _e = net(inputs)
                logits, sloss = slcp_l(cf, 0, labels=targets)
                bcel = reconst_criterion(_r, inputs)
                bcel = torch.sum(bcel.detach(), 2)
                _, predicted = torch.max(logits.data, 1)
                pos_lst = np.vstack((pos_lst, cf.detach().cpu().numpy()))
                grounTruth_lst = np.hstack((grounTruth_lst, targets.detach().cpu().numpy()))
                SSE_lst = np.hstack((SSE_lst, bcel.detach().cpu().numpy().flatten()))
                predict_lst = np.hstack((predict_lst, predicted.detach().cpu().numpy()))
        dfopt = pd.DataFrame(np.hstack((pos_lst, predict_lst.reshape(-1,1), grounTruth_lst.reshape(-1,1), SSE_lst.reshape(-1,1))), columns=['o_x', 'o_y', 'o_z', 'predict', 'groundtruth', 'reconst'] )
        return dfopt

    """
        訓練
    """
    def epochTrain(self,net, trainloader, optimizer, s_loss, rloss_weight=1.0):
        net.train()
        correct=0
        total=0
        total_loss = 0.0
        total_s_loss = 0.0
        total_reconst_loss = 0.0
        iter=0
        reconst_criterion = nn.MSELoss(reduction='sum')

        for data in trainloader:

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            _, targets = torch.max(labels.data, 1)

            # forward + backward + optimize
            _lg, cf, reconstruct, _ = net(inputs)
            logits, sloss = s_loss(cf, 0, labels=targets)

            reconst_loss = reconst_criterion(reconstruct, inputs) / labels.shape[0]
        
            # zero the parameter gradients
            optimizer.zero_grad()

            loss = sloss + reconst_loss * rloss_weight

            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            total_s_loss = total_s_loss + sloss
            if rloss_weight > 0:
                total_reconst_loss = total_reconst_loss + reconst_loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == targets).sum().item()
            total += labels.size(0)
            iter = iter + 1

        return {'acc':(100 * (correct / total)), 'sloss':(total_s_loss/iter), 'rloss':(total_reconst_loss/iter), 'tloss':(total_loss/iter)}
    """
        驗證
    """
    def epochTest(self, net, testloader, s_loss, rloss_weight=1.0):
        net.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        total_s_loss = 0.0
        total_reconst_loss = 0.0
        iter=0
        reconst_criterion = nn.MSELoss(reduction = 'sum')

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                _, targets = torch.max(labels.data, 1)

                _lg, cf, reconstruct, _ = net(inputs)
                logits, sloss = s_loss(cf, 0, labels=targets)

                reconst_loss = reconst_criterion(reconstruct, inputs) / labels.shape[0]

                loss = sloss + reconst_loss * rloss_weight
                total_loss = total_loss + loss.item()
                total_s_loss = total_s_loss + sloss
                if rloss_weight > 0:
                    total_reconst_loss = total_reconst_loss + reconst_loss.item()

                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == targets).sum().item()
                total += labels.size(0)
                iter = iter + 1

        return {'acc':(100 * (correct / total)), 'sloss':(total_s_loss/iter), 'rloss':(total_reconst_loss/iter), 'tloss':(total_loss/iter)}

    # """
    #     SSE函數繪製
    # """
    # def SSE_Filter(self, eval_org, eval_unknown):
    #     fig, ax = plt.subplots(figsize=(10,5))
    #     thold = np.percentile(eval_org.reconst,[99])[0]
    #     ax.set_title(f'{self.modelType} thold:{thold} outof 99%:{eval_unknown[eval_unknown.reconst > np.percentile(eval_org.reconst,[99])[0]].shape[0] * 100 / eval_unknown.shape[0]}%')
    #     ax.set_ylabel('times')
    #     ax.set_xlabel('SSE loss')
    #     ax.hist(eval_org.reconst, 250, facecolor='g', alpha=0.5)
    #     ax.hist(eval_unknown.reconst , 250, facecolor='r', alpha=0.5)
    #     return ax, thold
    #     # plt.savefig(f'./SSE/{self.modelType}_fxsd/{self.}_{seed}.png', transparent=False, facecolor='white')
    #     # # plt.show()
    #     # plt.close()
    # def outlierGet(self, eval_org, eval_unknown, sd, thold):
    #     svm_models = dict()
    #     for _ in range(len(self.pre.sort_lst)):
    #         fltr = (eval_org.predict == _) & (eval_org.predict == eval_org.groundtruth)
    #         # print(pre.sort_lst[_], (eval_org[fltr]).shape[0])
    #         if (eval_org[fltr]).shape[0] < 1:
    #             continue
    #         clf = linear_model.SGDOneClassSVM(nu = 0.5, tol = 1e-7, random_state=sd)
    #         clf.fit(eval_org.loc[fltr, ['o_x', 'o_y', 'o_z']])
    #         svm_models[self.pre.sort_lst[_]] = clf

    #     umt, odr, utp = 0, 0, 0
    #     for _ in range(len(self.pre.sort_lst)):
    #         flt_thd = (eval_unknown.reconst <= thold)
    #         fltr_prd = (eval_unknown.predict == _)
    #         fltr = (eval_org.predict == _) & (eval_org.predict == eval_org.groundtruth)
    #         # 被判斷為該類的數量要大於1才能比對
    #         if (eval_unknown[fltr_prd]).shape[0] < 1:
    #             continue
    #         # print(f'{pre.sort_lst[_]} 類 剩餘總量:{(eval_unknown[flt_thd & fltr_prd]).shape[0]} SSE過濾率：{ 100 * (1 - (eval_unknown[flt_thd & fltr_prd]).shape[0] / (eval_unknown[fltr_prd]).shape[0])}')
    #         umt += (eval_unknown[flt_thd & fltr_prd]).shape[0]
    #         # 計算陽性率
    #         # 被判斷為該類的惡意數量
    #         cf_r, cf_g = eval_unknown[flt_thd & fltr_prd & (eval_unknown.c_flag == 'r')], eval_unknown[flt_thd & fltr_prd & (eval_unknown.c_flag == 'g')]
    #         # print(f'判斷為該類的惡意數量{cf_r.__len__()}, 判斷為該類的良性數量{cf_g.__len__()}')
    #         RSVM = svm_models.get(self.pre.sort_lst[_])
    #         if RSVM == None:
    #             print(f'{self.pre.sort_lst[_]} 類 {cf_r.__len__() + cf_g.__len__()}')
    #             # odr += cf_r.__len__()
    #             pass
    #         try:
    #             tmp = eval_unknown.loc[flt_thd & fltr_prd, ['o_x', 'o_y', 'o_z']]#
    #             rawhtssc, htssc = RSVM.score_samples(eval_org.loc[fltr, ['o_x', 'o_y', 'o_z']]), RSVM.score_samples(tmp[~np.isnan(tmp).any(axis=1)])
    #         except:
    #             # print(eval_unknown[flt_thd & fltr_prd].shape[0])
    #             pass
    #         else:
    #             # print(np.percentile(rawhtssc,[0.15, 1, 5, 10, 15,50,99.7]), np.percentile(htssc,[1,90,99]))
    #             L_lim, H_lim = np.percentile(rawhtssc,[0.5,99.5])
    #             # print(f'SVM_score閾值:{L_lim},{H_lim}')
    #             insider_r = rawhtssc[(rawhtssc>L_lim) & (rawhtssc<H_lim)].shape[0] / rawhtssc.shape[0]
    #             outsider_r = htssc[(htssc<L_lim) | (htssc>H_lim)].shape[0]/htssc.shape[0]
    #             utp += htssc[(htssc<L_lim) | (htssc>H_lim)].shape[0]
    #             if (_ > 0):
    #                 odr += cf_r.__len__()
    #             elif (RSVM != None):
    #                 odr += htssc[(htssc<L_lim) | (htssc>H_lim)].shape[0]
    #             print(f'{self.pre.sort_lst[_]} 類 域內該類:{insider_r * 100}%, 未知數量:{htssc.shape[0]}, TP:{htssc[(htssc<L_lim) | (htssc>H_lim)].shape[0]}, 檢出率{outsider_r * 100}%')
    #     print(f'總檢出:{odr}, 被當未知{utp}, SSE後總未知{umt}, FDR:{100 * odr / umt}%, TDR:{100* ((eval_unknown[eval_unknown.reconst > thold]).shape[0] + odr) / eval_unknown.shape[0]}')

    # def cfmxPlot(self, eval_org, eval_unknown):
    #     print(100 * eval_org[eval_org.predict == eval_org.groundtruth].shape[0] / eval_org.shape[0])
    #     # eval_org.predict[eval_org.predict > 0] = 1
    #     # eval_org.groundtruth[(eval_org.groundtruth > 0) | (eval_org.raw_gt < 1)] = 1
    #     # print(precision_recall_fscore_support(y_true=eval_org.groundtruth, y_pred=eval_org.predict, average='binary'))
    #     print(precision_recall_fscore_support(y_true=eval_org.groundtruth, y_pred=eval_org.predict, average='weighted'))
    #     fig, ax = plt.subplots(figsize=(10,10))
    #     plt.tight_layout()
    #     ConfusionMatrixDisplay.from_predictions(y_true=eval_org.groundtruth, y_pred=eval_org.predict, cmap='Blues', values_format='.3f', ax=ax, normalize='true' ,display_labels=dt0.columns.to_list(), xticks_rotation=-45)
    #     ax.set_title(f'Model: {M_lst[msel]}      fileVer: {tr_evlst[msel]}    ACC:{100 * eval_org[eval_org.predict == eval_org.groundtruth].shape[0] / eval_org.shape[0]}')
    #     # plt.savefig(f'./SSE/{M_lst[msel]}_fxsd/origin_{sd}.png', transparent=False, facecolor='white')
    #     plt.show()