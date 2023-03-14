import numpy as np
import torch
import torchvision

def load_data():
    dataset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True)
    labels = np.array(dataset.targets)
    mask = labels == dataset.classes.index('cat')
    mask[labels == dataset.classes.index('dog')] = True
    images = dataset.data[mask]
    labels = np.array(dataset.targets)[mask]
    labels[labels == dataset.classes.index('cat')] = 0
    labels[labels == dataset.classes.index('dog')] = 1
    classes = ['cat', 'dog']
    return images, labels, classes

def rel_err(a, b):
    return torch.max(torch.abs(a - b) / (torch.maximum(torch.tensor(1e-8), torch.abs(a) + torch.abs(b))))

def Sigmoid_Of_Zero(model):
    val = model.sigmoid(torch.tensor(0))
    truth = 0.5
    if (rel_err(torch.tensor(truth).to(model.dv), val) < 1e-6).cpu().numpy():
        return 'Correct'
    else:
        return f'Expected {truth} but got {val.cpu().numpy()}'

def Sigmoid_Of_Zero_Array(model):
    val = torch.sum(model.sigmoid(torch.tensor([0, 0, 0, 0, 0])))
    truth = 2.5
    if (rel_err(torch.tensor(truth).to(model.dv), val) < 1e-6).cpu().numpy():
        return 'Correct'
    else:
        return f'Expected {truth} but got {val.cpu().numpy()}'

def Sigmoid_Of_Hundred(model):
    val = model.sigmoid(torch.tensor(100))
    truth = 1.0
    if (rel_err(torch.tensor(truth).to(model.dv), val) < 1e-6).cpu().numpy():
        return 'Correct'
    else:
        return f'Expected {truth} but got {val.cpu().numpy()}'

def Sigmoid_Of_Hundred_Array(model):
    val = torch.sum(model.sigmoid(torch.tensor([100, 100, 100, 100, 100])))
    truth = 5.0
    if (rel_err(torch.tensor(truth).to(model.dv), val) < 1e-6).cpu().numpy():
        return 'Correct'
    else:
        return f'Expected {truth} but got {val.cpu().numpy()}'

def Forward_Test(model):
    x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64).reshape(3, 2)
    val = model.forward(x)
    truth = [0.50006411, 0.50017232, 0.50028053]
    if (rel_err(torch.tensor(truth).to(model.dv), val) < 1e-6).cpu().numpy():
        return 'Correct'
    else:
        return f'Expected {truth} but got {val.cpu().numpy()}'

def Backward_Test(model):
    x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64).reshape(3, 2)
    y = torch.tensor([0, 1, 1], dtype=torch.float64).to(model.dv)
    pred_y = model.forward(x)
    L = torch.pow(y - pred_y, 2) # using MSE loss
    dL = 2 * pred_y - 2 * y
    val = model.backward(dL)
    truth = -1.249252247115394
    if (rel_err(torch.tensor(truth).to(model.dv), val) < 1e-6).cpu().numpy():
        return 'Correct'
    else:
        return f'Expected {truth} but got {val.cpu().numpy()}'

def Model_Tests(model):
    print('Result of sigmoid function with single value 0:', Sigmoid_Of_Zero(model))
    print('Result of sigmoid function with all-zero array:', Sigmoid_Of_Zero_Array(model))
    print('Result of sigmoid function with single value 100:', Sigmoid_Of_Hundred(model))
    print('Result of sigmoid function with all-100 array:', Sigmoid_Of_Hundred_Array(model))
    print('Result of forward:', Forward_Test(model))
    print('Result of backward:', Backward_Test(model))

def Loss_Test(loss_func):
    val1, val2 = loss_func(torch.tensor([0.0, 1.0]), torch.tensor([0.32, 0.75]))
    truth1 = [0.38566248, 0.28768207]
    truth2 = [1.47058824, -1.33333333]
    log = 'Result of forward: '
    if (rel_err(torch.tensor(truth1).to(loss_func.dv), val1) < 1e-6).cpu().numpy():
      log += 'Correct'
    else:
      log += f'Expected {truth1} but got {val1.cpu().numpy()}'
    log += '\nResult of backward: '
    if (rel_err(torch.tensor(truth2).to(loss_func.dv), val2) < 1e-6).cpu().numpy():
      log += 'Correct'
    else:
      log += f'Expected {truth2} but got {val2.cpu().numpy()}'
    print(log)

def Optimizer_Test(optimizer):
    optimizer.step(torch.tensor([-2, 9]))
    val = optimizer.model.W
    truth = [0.20017641, -0.89995998]
    log = 'Result of optimizer test: '
    if (rel_err(torch.tensor(truth).to(optimizer.model.dv), val) < 1e-6).cpu().numpy():
        log += 'Correct'
    else:
        log += f'Expected {truth} but got {val.cpu().numpy()}'
    print(log)

def evaluate(ground_truth, pred):
    pred[pred > 1e-3] = 1
    pred[pred != 1] = 0
    return np.count_nonzero(pred == ground_truth)