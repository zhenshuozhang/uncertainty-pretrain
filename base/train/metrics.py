import torch
import torch.nn.functional as F
import numpy as np

from torchmetrics.functional.classification import binary_calibration_error

from scipy.stats import norm as gaussian
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibrationDisplay

import matplotlib.pyplot as plt

def classification_metrics(preds, lbs, note=None):

    result_metrics_dict = dict()

    roc_auc_list = list()
    prc_auc_list = list()
    ece_list = list()
    mce_list = list()
    nll_list = list()
    brier_list = list()

    roc_auc_valid_flag = True
    prc_auc_valid_flag = True
    ece_valid_flag = True
    mce_valid_flag = True
    nll_valid_flag = True
    brier_valid_flag = True

    for i in range(lbs.shape[-1]):
        
        if torch.sum(lbs[:,i] == 1) > 0 and torch.sum(lbs[:,i] == -1) > 0:
            is_valid = lbs[:,i]**2 > 0
            lbs_ = (lbs[is_valid,i] + 1)/2
            preds_ = preds[is_valid,i]
        
            #lbs_ = lbs[:, i]
            #preds_ = preds[:, i]
            #lbs_ = (lbs_ + 1)/2
            if len(lbs_) < 1:
                continue
            if (lbs_ < 0).any():
                raise ValueError("Invalid label value encountered!")
            if (lbs_ == 0).all() or (lbs_ == 1).all():  # skip tasks with only one label type, as Uni-Mol did.
                continue
            preds_ = torch.sigmoid(preds_)
            # --- roc-auc ---
            try:
                roc_auc = roc_auc_score(lbs_, preds_)
                roc_auc_list.append(roc_auc)
            except Exception as e:
                roc_auc_valid_flag = False
                print("roc-auc error: ", e)

            # --- prc-auc ---
            try:
                p, r, _ = precision_recall_curve(lbs_, preds_)
                prc_auc = auc(r, p)
                prc_auc_list.append(prc_auc)
            except Exception as e:
                prc_auc_valid_flag = False
                print("prc-auc error: ", e)

            # --- ece ---
            try:
                ece = binary_calibration_error(preds_, lbs_, n_bins=15).item()
                ece_list.append(ece)
                """
                prob_true, prob_pred = calibration_curve(lbs_, preds_, n_bins=15)
                preds_weights = np.ones_like(preds_)/float(len(preds_))
                prob_weights = np.ones_like(prob_pred)/float(len(prob_pred))
                p_scaled = (preds_ - preds_.min()) / (preds_.max() - preds_.min())
                print(prob_pred)
                print(prob_true)
                disp = CalibrationDisplay(prob_true, prob_pred, preds_)
                disp.plot()
                #plt.hist(prob_pred, bins=15,weights=prob_weights)
                plt.hist(preds_, bins=15,weights=preds_weights)
                plt.savefig(f"./calibration_curve/{note}.jpg")
                plt.cla()
                """
            except Exception as e:
                ece_valid_flag = False
                print("ece error: ", e)

            # --- mce ---
            try:
                mce = binary_calibration_error(preds_, lbs_, norm='max', n_bins=15).item()
                mce_list.append(mce)
            except Exception as e:
                mce_valid_flag = False
                print("mce error: ", e)

            # --- nll ---
            try:
                nll = F.binary_cross_entropy(
                    input=preds_,
                    target=lbs_.to(torch.float),
                    reduction='mean'
                ).item()
                nll_list.append(nll)
            except Exception as e:
                print("nll error: ", e)
                nll_valid_flag = False

            # --- brier ---
            try:
                brier = brier_score_loss(lbs_, preds_)
                brier_list.append(brier)
            except Exception as e:
                brier_valid_flag = False
                print("brier error: ", e)

    if roc_auc_valid_flag:
        roc_auc_avg = np.mean(roc_auc_list)
        result_metrics_dict['roc-auc'] = {'all': roc_auc_list, 'macro-avg': roc_auc_avg}

    if prc_auc_valid_flag:
        prc_auc_avg = np.mean(prc_auc_list)
        result_metrics_dict['prc-auc'] = {'all': prc_auc_list, 'macro-avg': prc_auc_avg}

    if ece_valid_flag:
        ece_avg = np.mean(ece_list)
        result_metrics_dict['ece'] = {'all': ece_list, 'macro-avg': ece_avg}

    if mce_valid_flag:
        mce_avg = np.mean(mce_list)
        result_metrics_dict['mce'] = {'all': mce_list, 'macro-avg': mce_avg}

    if nll_valid_flag:
        nll_avg = np.mean(nll_list)
        result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}

    if brier_valid_flag:
        brier_avg = np.mean(brier_list)
        result_metrics_dict['brier'] = {'brier': brier_list, 'macro-avg': brier_avg}

    return result_metrics_dict

def task_cls_metrics(preds, lbs, task_index=0):

    result_metrics_dict = dict()

    roc_auc_list = list()
    prc_auc_list = list()
    ece_list = list()
    mce_list = list()
    nll_list = list()
    brier_list = list()

    roc_auc_valid_flag = True
    prc_auc_valid_flag = True
    ece_valid_flag = True
    mce_valid_flag = True
    nll_valid_flag = True
    brier_valid_flag = True

    i = task_index

    if torch.sum(lbs[:,i] == 1) > 0 and torch.sum(lbs[:,i] == -1) > 0:
        is_valid = lbs[:,i]**2 > 0
        lbs_ = (lbs[is_valid,i] + 1)/2
        preds_ = preds[is_valid,i]
    
        #lbs_ = lbs[:, i]
        #preds_ = preds[:, i]
        #lbs_ = (lbs_ + 1)/2
        if len(lbs_) < 1:
            return
        if (lbs_ < 0).any():
            raise ValueError("Invalid label value encountered!")
        if (lbs_ == 0).all() or (lbs_ == 1).all():  # skip tasks with only one label type, as Uni-Mol did.
            return
        preds_ = torch.sigmoid(preds_)
        # --- roc-auc ---
        try:
            roc_auc = roc_auc_score(lbs_, preds_)
            roc_auc_list.append(roc_auc)
        except Exception as e:
            roc_auc_valid_flag = False
            print("roc-auc error: ", e)

        # --- prc-auc ---
        try:
            p, r, _ = precision_recall_curve(lbs_, preds_)
            prc_auc = auc(r, p)
            prc_auc_list.append(prc_auc)
        except Exception as e:
            prc_auc_valid_flag = False
            print("prc-auc error: ", e)

        # --- ece ---
        try:
            ece = binary_calibration_error(preds_, lbs_, n_bins=10).item()
            ece_list.append(ece)
        except Exception as e:
            ece_valid_flag = False
            print("ece error: ", e)

        # --- mce ---
        try:
            mce = binary_calibration_error(preds_, lbs_, n_bins=10, norm='max').item()
            mce_list.append(mce)
        except Exception as e:
            mce_valid_flag = False
            print("mce error: ", e)

        # --- nll ---
        try:
            nll = F.binary_cross_entropy(
                input=preds_,
                target=lbs_.to(torch.float),
                reduction='mean'
            ).item()
            nll_list.append(nll)
        except Exception as e:
            print("nll error: ", e)
            nll_valid_flag = False

        # --- brier ---
        try:
            brier = brier_score_loss(lbs_, preds_)
            brier_list.append(brier)
        except Exception as e:
            brier_valid_flag = False
            print("brier error: ", e)

    if roc_auc_valid_flag:
        roc_auc_avg = np.mean(roc_auc_list)
        result_metrics_dict['roc-auc'] = {'all': roc_auc_list, 'macro-avg': roc_auc_avg}

    if prc_auc_valid_flag:
        prc_auc_avg = np.mean(prc_auc_list)
        result_metrics_dict['prc-auc'] = {'all': prc_auc_list, 'macro-avg': prc_auc_avg}

    if ece_valid_flag:
        ece_avg = np.mean(ece_list)
        result_metrics_dict['ece'] = {'all': ece_list, 'macro-avg': ece_avg}

    if mce_valid_flag:
        mce_avg = np.mean(mce_list)
        result_metrics_dict['mce'] = {'all': mce_list, 'macro-avg': mce_avg}

    if nll_valid_flag:
        nll_avg = np.mean(nll_list)
        result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}

    if brier_valid_flag:
        brier_avg = np.mean(brier_list)
        result_metrics_dict['brier'] = {'brier': brier_list, 'macro-avg': brier_avg}

    return result_metrics_dict

def regression_calibration_error(lbs, preds, variances, n_bins=20):
    sigma = np.sqrt(variances)
    phi_lbs = gaussian.cdf(lbs, loc=preds.reshape(-1, 1), scale=sigma.reshape(-1, 1))

    expected_confidence = np.linspace(0, 1, n_bins+1)[1:-1]
    observed_confidence = np.zeros_like(expected_confidence)

    for i in range(0, len(expected_confidence)):
        observed_confidence[i] = np.mean(phi_lbs <= expected_confidence[i])

    calibration_error = np.mean((expected_confidence.ravel() - observed_confidence.ravel()) ** 2)

    return calibration_error

def regression_metrics(preds, variances, lbs):

    if len(preds.shape) == 1:
        preds = preds[:, np.newaxis]

    if variances is not None and len(variances.shape) == 1:
        variances = variances[:, np.newaxis]

    # --- rmse ---
    result_metrics_dict = dict()

    rmse_valid_flag = True
    mae_valid_flag = True
    var_valid_flag = True
    nll_valid_flag = True
    ce_valid_flag = True

    rmse_list = list()
    mae_list = list()
    var_list = list()
    nll_list = list()
    ce_list = list()

    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i]
        preds_ = preds[:, i]
        if variances is not None:
            vars_ = variances[:, i]

        # --- rmse ---
        try:
            rmse = mean_squared_error(lbs_, preds_, squared=False)
            rmse_list.append(rmse)
        except:
            rmse_valid_flag = False

        # --- mae ---
        try:
            mae = mean_absolute_error(lbs_, preds_)
            mae_list.append(mae)
        except:
            mae_valid_flag = False
        
        try:
            var = vars_
            var_list.append(var.mean())
        except:
            var_valid_flag = False

        # --- Gaussian NLL ---
        try:
            #process_result(pred=preds_, lbs=lbs_, var=vars_)
            nll = F.gaussian_nll_loss(preds_, lbs_, vars_).item()
            nll_list.append(nll)
        except:
            nll_valid_flag = False

        # --- calibration error ---
        try:
            ce = regression_calibration_error(lbs_, preds_, vars_)
            ce_list.append(ce)
        except Exception as e:
            print('ce error: ')
            ce_valid_flag = False

    if rmse_valid_flag:
        rmse_avg = np.mean(rmse_list)
        result_metrics_dict['rmse'] = {'all': rmse_list, 'macro-avg': rmse_avg}

    if mae_valid_flag:
        mae_avg = np.mean(mae_list)
        result_metrics_dict['mae'] = {'all': mae_list, 'macro-avg': mae_avg}
    
    if var_valid_flag:
        var_avg = np.mean(var_list)
        result_metrics_dict['var'] = {'all': var_list, 'macro-avg': var_avg}
    
    if nll_valid_flag:
        nll_avg = np.mean(nll_list)
        result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}

    if ce_valid_flag:
        ce_avg = np.mean(ce_list)
        result_metrics_dict['ce'] = {'all': ce_list, 'macro-avg': ce_avg}

    return result_metrics_dict

def task_reg_metrics(preds, variances, lbs, task_index):

    if len(preds.shape) == 1:
        preds = preds[:, np.newaxis]

    if variances is not None and len(variances.shape) == 1:
        variances = variances[:, np.newaxis]

    # --- rmse ---
    result_metrics_dict = dict()

    rmse_valid_flag = True
    mae_valid_flag = True
    nll_valid_flag = True
    ce_valid_flag = True

    rmse_list = list()
    mae_list = list()
    nll_list = list()
    ce_list = list()

    i = task_index
    lbs_ = lbs[:, i]
    preds_ = preds[:, i]
    if variances is not None:
        vars_ = variances[:, i]
    else:
        vars_ = None

    # --- rmse ---
    try:
        rmse = mean_squared_error(lbs_, preds_, squared=False)
        rmse_list.append(rmse)
    except:
        rmse_valid_flag = False

    # --- mae ---
    try:
        mae = mean_absolute_error(lbs_, preds_)
        mae_list.append(mae)
    except:
        mae_valid_flag = False

    # --- Gaussian NLL ---
    try:
        nll = F.gaussian_nll_loss(preds_, lbs_, vars_).item()
        nll_list.append(nll)
    except:
        nll_valid_flag = False

    # --- calibration error ---
    try:
        ce = regression_calibration_error(lbs_, preds_, vars_)
        ce_list.append(ce)
    except Exception as e:
        print('ce error', e)
        ce_valid_flag = False

    if rmse_valid_flag:
        rmse_avg = np.mean(rmse_list)
        result_metrics_dict['rmse'] = {'all': rmse_list, 'macro-avg': rmse_avg}

    if mae_valid_flag:
        mae_avg = np.mean(mae_list)
        result_metrics_dict['mae'] = {'all': mae_list, 'macro-avg': mae_avg}

    if nll_valid_flag:
        nll_avg = np.mean(nll_list)
        result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}

    if ce_valid_flag:
        ce_avg = np.mean(ce_list)
        result_metrics_dict['ce'] = {'all': ce_list, 'macro-avg': ce_avg}
    else:
        result_metrics_dict['ce'] = {'all': None, 'macro-avg': None}

    return result_metrics_dict

def process_result(pred, lbs, var):
    f_nll = lambda input, target, var: 0.5 * (torch.log(var) + (input - target)**2 / var)
    print(pred.mean())
    print(pred.var())
    print(lbs.mean())
    print(lbs.var())
    print(var.mean())
    #n = (torch.log(vars_.mean())+((preds_.mean() - lbs_.mean())*(preds_.mean() - lbs_.mean()))/vars_.mean())/2
    n = f_nll(pred, lbs, var)
    index = torch.argmax(n)
    print('note: ',pred[index])
    print(lbs[index])
    print(var[index])
    n_mean = f_nll(pred.mean(), lbs.mean(), var.mean())
    print(n_mean)

    col1 = (0.6, 0.8, 1.0) # 浅蓝色
    col2 = (0.8, 1.0, 0.6) # 浅绿色
    col3 = (0.9, 0.6, 1.0) # 浅紫色

    m = torch.randperm(pred.shape[0])
    mask = m[:100]
    
    # 创建图形
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('input')
    ax1.set_ylabel('predction & label', color='blue')
    ax1.plot(pred[mask], color=col1, label='pred')
    ax1.plot(lbs[mask], color=col2, label='label')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 添加双轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('var', color='red')
    ax2.plot(var[mask], color=col3, label='var')
    ax2.tick_params(axis='y', labelcolor='red')

    # 添加图例
    fig.legend(loc='upper left')

    plt.savefig("./calibration_curve/reg.jpg")