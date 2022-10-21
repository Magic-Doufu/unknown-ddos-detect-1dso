import utils.util as util
import DHRTool, torch, argparse
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

def argParser():
    parser = argparse.ArgumentParser(description='選擇模型與資料集')
    parser.add_argument("-m", "--model",
                        help='選擇模型，如1或2。',
                        default=1,
                        type=int)
    parser.add_argument('-lr', "--learningrate", help="學習率", default=3e-3, type=float)
    parser.add_argument('-wdcy', "--weightdecay", help="權重衰減", default=3e-5, type=float)
    parser.add_argument('-s', '--seeds', default=[0,42,123,222,419,844,918,1344,65536,815149],
                        help='指定種子')
    parser.add_argument('-sp', '--savepath', default='./save_models',
                        help='模型儲存路徑')
    return parser

parser = argParser()
args = parser.parse_args()
datatype = util.m_lst[args.model]
dataver = util.ver_lst.get(datatype)
seeds = args.seeds

model_tool = DHRTool.DDoS_DHR()
dataname = f'Wednesday-WorkingHours.pcap_{dataver}'
X, y = model_tool.datasetLoad(dataName=dataname, dataType=datatype)
y = y.to_numpy()
#檢查模型存放路徑，如不存在就產生
util.pathCreate(args.savepath)
def train_loop(_sd):
    # 固定種子
    util.fixed_seed(_sd)

    # 分割並產出dataloader
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=_sd, train_size=0.8)
    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test) # transform to torch tensor
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    tr_dataset = TensorDataset(X_train, y_train) # create your datset
    ts_dataset = TensorDataset(X_test, y_test) # create your datset
    train_dataloader = DataLoader(tr_dataset, batch_size=model_tool.batch_size, shuffle=True) # create your dataloader
    test_dataloader = DataLoader(ts_dataset, batch_size=model_tool.batch_size, shuffle=True) # create your dataloader
    # 產出模型
    net, slcp_l = model_tool.modelGen(sl_w=1.0)
    net, slcp_l = net.cuda(), slcp_l.cuda()
    # 重構誤差權重, 最小loss初始化, epochs
    rloss_w, min_loss = 1, 1
    epochs = 100
    # 最佳化器
    optimizer = optim.Adam(list(net.parameters()) + list(slcp_l.parameters()), lr=args.learningrate, weight_decay=args.weightdecay)
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f'<--------- epoch: {epoch + 1 }--------->')
        train_acc = model_tool.epochTrain(net, trainloader=train_dataloader, optimizer=optimizer, s_loss=slcp_l)
        print(f"Train acc:{train_acc.get('acc')}%, sloss:{train_acc.get('sloss')}, rloss:{train_acc.get('rloss')} and total loss:{train_acc.get('tloss')}")
        test_acc = model_tool.epochTest(net, testloader=test_dataloader, s_loss=slcp_l)
        print(f"Test acc:{test_acc.get('acc')}%, sloss:{test_acc.get('sloss')}, rloss:{test_acc.get('rloss')} and total loss:{test_acc.get('tloss')}")
        if (train_acc.get('tloss') + test_acc.get('tloss')) / 2 < min_loss:
            min_loss = (train_acc.get('tloss') + test_acc.get('tloss')) / 2
            print(f'min_loss:{min_loss}, epochs:{epoch + 1}')
            torch.save({'epoch':epoch,
                            'model_state_dict':net.state_dict(),
                            'train_acc':train_acc.get('acc'),
                            'train_loss':train_acc.get('tloss'),
                            'val_acc':test_acc.get('acc'),
                            'val_loss':test_acc.get('tloss'),
                            'sloss_w':rloss_w,
                            'centers':slcp_l.points.detach().cpu().numpy()},
                f"{args.savepath}/{datatype}_sd_{_sd}.pth")
#種子迴圈
for _sd in seeds:
    train_loop(_sd)