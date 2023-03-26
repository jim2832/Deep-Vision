import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def load_data():
    dataset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True)
    images = dataset.data
    labels = np.array(dataset.targets)
    classes = dataset.classes
    return images, labels, classes

def rel_err(a, b):
    return torch.max(torch.abs(a - b) / (torch.maximum(torch.tensor(1e-8), torch.abs(a) + torch.abs(b))))

ReLU_F_Ans = {(2,5): [[0.09762701, 0.43037873, 0.20552675, 0.08976637, 0.00000000], [0.29178823, 0.00000000, 0.78354600, 0.92732552, 0.00000000]],
              (3,4): [[0.58345008, 0.05778984, 0.13608912, 0.85119328], [0.00000000, 0.00000000, 0.00000000, 0.66523969], [0.55631350, 0.74002430, 0.95723668, 0.59831713]],
              (7,1): [[0.00000000], [0.56105835], [0.00000000], [0.27984204], [0.00000000], [0.88933783], [0.04369664]]}
ReLU_B_Ans = {(2,5): [[1., 1., 1., 1., 0.], [1., 0., 1., 1., 0.]],
              (3,4): [[1., 1., 1., 1.], [0., 0., 0., 1.], [1., 1., 1., 1.]],
              (7,1): [[0.], [1.], [0.], [1.], [0.], [1.], [1.]]}
FC_F_Ans = {(3,12,4): [[4.39296838, 2.74169043, 4.87948061, 2.12806858], [4.15973249, 2.67554720, 5.12027539, 2.06837809], [3.38263392, 2.17648150, 3.67256300, 1.60918374]],
            (7,2,2): [[1.67124587, 1.11823805], [2.32782206, 1.54333434], [1.59182420, 1.12606827], [1.86775078, 1.25991412], [1.90274891, 1.33479189], [1.99137841, 1.36181377], [1.71304272, 1.14277117]],
            (2,10,1): [[4.55737345], [3.78495605]]}
FC_dw_Ans = {(3,12,4): [[0.15344125, 0.30643143, 0.14575764, 0.15696668], [0.30307481, 0.44684002, 0.34205553, 0.21663028], [0.14686680, 0.20463775, 0.11047041, 0.13865895], [0.29312252, 0.19592198, 0.32729083, 0.12801923],
                        [0.18424601, 0.13955896, 0.19113185, 0.09630192], [0.24164752, 0.40166434, 0.26348565, 0.19518408], [0.16991717, 0.32312752, 0.19463525, 0.14528135], [0.36079245, 0.48916752, 0.39255710, 0.25273089],
                        [0.31717869, 0.53523705, 0.32277747, 0.27485827], [0.21985435, 0.31340653, 0.27487719, 0.13499362], [0.17594457, 0.35556504, 0.13378319, 0.20434426], [0.25657949, 0.35434448, 0.30212073, 0.16629209]],
             (7,2,2): [[0.26137986, 0.22967393], [0.36618800, 0.28642302]],
             (2,10,1): [[0.24861085], [0.25695815], [0.24634177], [0.28489117], [0.28321942], [0.19045496], [0.38450911], [0.16236499], [0.18745906], [0.38526085]]}
FC_db_Ans = {(3,12,4): [0.43141910, 0.55524365, 0.47897169, 0.28509907],
             (7,2,2): [0.50543560, 0.43863018],
             (2,10,1): [0.43375871]}
SCE_pred_Ans = {(5,2): [1, 0, 1, 1, 0],
                (3,4): [1, 1, 1],
                (10,7): [1, 5, 2, 5, 2, 5, 4, 6, 1, 0]}
SCE_loss_Ans = {(5,2): 0.69616684, (3,4): 1.39425582, (10,7): 1.96053199}
SCE_B_Ans = {(5,2): [[-0.10829966,  0.10829966], [ 0.10289320, -0.10289320], [ 0.08893354, -0.08893354], [-0.12232680,  0.12232680], [-0.07177634,  0.07177634]],
             (3,4): [[ 0.06951818,  0.10829752,  0.06577236, -0.24358806], [-0.26984137,  0.11441441,  0.05055302,  0.10487394], [ 0.07148374, -0.23416956,  0.07491390,  0.08777191]],
             (10,7): [[ 0.00963800,  0.02147800, -0.08592770,  0.01264197,  0.01087989,  0.01811235,  0.01317750],
                      [ 0.01370346,  0.00790903,  0.01439456, -0.08568496,  0.01438446,  0.01994469,  0.01534877],
                      [-0.08719655,  0.01383552,  0.01795446,  0.00949184,  0.01740877,  0.01747629,  0.01102968],
                      [ 0.01028525,  0.01239401,  0.01300710,  0.01599029, -0.08598138,  0.02429228,  0.01001245],
                      [-0.08724387,  0.01216356,  0.01989053,  0.01333547,  0.01650142,  0.01321776,  0.01213514],
                      [ 0.01091543,  0.01884282,  0.01122323,  0.01189817,  0.01413322, -0.07778438,  0.01077150],
                      [ 0.01622575,  0.00772725,  0.01863636,  0.01121560,  0.01864198,  0.01285201, -0.08529896],
                      [-0.08825278,  0.01498779,  0.01273846,  0.01518897,  0.01271976,  0.01552439,  0.01709340],
                      [ 0.00996876,  0.01868611,  0.01647608,  0.01219100,  0.01577705, -0.08972978,  0.01663077],
                      [ 0.02186867,  0.01187373, -0.08316989,  0.00985084,  0.01767385,  0.01153245,  0.01037035]]}

def ReLU_Tests(layer):
    pass_tests = True
    np.random.seed(0)
    shapes = [(2,5), (3,4), (7,1)]
    for test_shape in shapes:
        x = np.random.uniform(-1, 1, size=test_shape)
        val = layer.forward(torch.from_numpy(x))
        truth = ReLU_F_Ans[test_shape]
        if (rel_err(torch.tensor(truth), val) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth} but got {val.cpu().numpy()} during forward')
            pass_tests = False
        val = layer.backward(torch.ones(test_shape))
        truth = ReLU_B_Ans[test_shape]
        if (rel_err(torch.tensor(truth), val) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth} but got {val.cpu().numpy()} during backward')
            pass_tests = False
    if pass_tests:
        print('Results of ReLU function tests: All passed.')

def FCL_Tests(layer):
    pass_tests = True
    np.random.seed(0)
    shapes = [(3,12,4), (7,2,2), (2,10,1)]
    for test_shape in shapes:
        N, D, F = test_shape
        x = torch.from_numpy(np.random.rand(N, D))
        w = torch.from_numpy(np.random.rand(D, F))
        b = torch.from_numpy(np.random.rand(F))
        val = layer.forward(x, w, b)
        truth = FC_F_Ans[test_shape]
        if (rel_err(torch.tensor(truth), val) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth} but got {val.cpu().numpy()} during forward')
            pass_tests = False
        dw, db = layer.backward(torch.from_numpy(np.random.rand(N, F)))
        truth_dw = FC_dw_Ans[test_shape]
        if (rel_err(torch.tensor(truth_dw), dw) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_dw} but got {val.cpu().numpy()} during backward of weights')
            pass_tests = False
        truth_db = FC_db_Ans[test_shape]
        if (rel_err(torch.tensor(truth_db), db) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_db} but got {val.cpu().numpy()} during backward of bias')
            pass_tests = False
    if pass_tests:
        print('Results of fully connected layer forward and backward tests: All passed.')

def SCE_Tests(layer):
    pass_tests = True
    np.random.seed(0)
    shapes = [(5,2), (3,4), (10,7)]
    for test_shape in shapes:
        N, F = test_shape
        out = torch.from_numpy(np.random.rand(N, F))
        y = torch.from_numpy(np.random.randint(0, F, size=(N)))
        pred, loss = layer.forward(y, out)
        truth_pred = SCE_pred_Ans[test_shape]
        if (rel_err(torch.tensor(truth_pred), pred) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_pred} but got {pred.cpu().numpy()} during softmax prediction')
            pass_tests = False
        truth_loss = SCE_loss_Ans[test_shape]
        if (rel_err(torch.tensor(truth_loss), loss) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_loss} but got {loss.cpu().numpy()} during CE loss forward')
            pass_tests = False
        val = layer.backward()
        truth = SCE_B_Ans[test_shape]
        if (rel_err(torch.tensor(truth), val) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth} but got {val.cpu().numpy()} during CE loss backward')
            pass_tests = False
    if pass_tests:
        print('Results of softmax and cross entropy forward and backward tests: All passed.')


def plot_curves(cand, train, valid):
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['font.size'] = 12
    plt.subplot(1, 2, 1)
    for idx in range(len(train)):
        plt.plot(train[idx], label=str(cand[idx]))
    plt.title('Training accuracy log')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    for idx in range(len(valid)):
        plt.plot(valid[idx], label=str(cand[idx]))
    plt.title('Validation accuracy log')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_result(train_loss, train_acc, valid_loss, valid_acc):
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['font.size'] = 12
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='training')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='training')
    plt.plot(valid_acc, label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

class Dataset(object):
    def __init__(self, images, labels):
        num = images.shape[0]
        self.images = images.reshape(num, -1).astype(np.float64) / 255
        self.labels = labels
    def __len__(self):
        num_data = self.images.shape[0]
        return num_data
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

class Dataloader(object):
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.indice = np.array(range(len(self.dataset))) 
        self.batch_size = batch_size
    def __len__(self):
        num_batch = len(self.dataset) // self.batch_size
        return num_batch
    def __getitem__(self, idx):
        batch_data = self.dataset[self.indice[idx: idx+self.batch_size]]
        return batch_data
    def shuffle(self):
        np.random.shuffle(self.indice)