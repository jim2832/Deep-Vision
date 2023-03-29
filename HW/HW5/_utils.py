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

ReLU_F_Ans = {(10, 4): [[0.09762701, 0.43037873, 0.20552675, 0.08976637], [0.00000000, 0.29178823, 0.00000000, 0.78354600],
                        [0.92732552, 0.00000000, 0.58345008, 0.05778984], [0.13608912, 0.85119328, 0.00000000, 0.00000000],
                        [0.00000000, 0.66523969, 0.55631350, 0.74002430], [0.95723668, 0.59831713, 0.00000000, 0.56105835],
                        [0.00000000, 0.27984204, 0.00000000, 0.88933783], [0.04369664, 0.00000000, 0.00000000, 0.54846738],
                        [0.00000000, 0.13686790, 0.00000000, 0.23527099], [0.22419145, 0.23386799, 0.88749616, 0.36364060]],
              (8, 3): [[0.00000000, 0.00000000, 0.39526239], [0.00000000, 0.33353343, 0.34127574], [0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.14039354, 0.00000000],
                       [0.97674768, 0.00000000, 0.00000000], [0.00000000, 0.30621665, 0.00000000], [0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.31265918, 0.00000000]],
              (2, 9): [[0.00000000, 0.00000000, 0.64198646, 0.00000000, 0.67588981, 0.00000000, 0.95291893, 0.00000000, 0.95352218],
                       [0.20969104, 0.47852716, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]]}
ReLU_B_Ans = {(10, 4): [[1., 1., 1., 1.], [0., 1., 0., 1.], [1., 0., 1., 1.], [1., 1., 0., 0.], [0., 1., 1., 1.],
                        [1., 1., 0., 1.], [0., 1., 0., 1.], [1., 0., 0., 1.], [0., 1., 0., 1.], [1., 1., 1., 1.]],
              (8, 3): [[0., 0., 1.], [0., 1., 1.], [0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 1., 0.]],
              (2, 9): [[0., 0., 1., 0., 1., 0., 1., 0., 1.], [1., 1., 0., 0., 0., 0., 0., 0., 0.]]}
FC_F_Ans = {(10, 4, 1): [[1.62995249], [1.46033227], [1.76497232], [1.33030290], [1.63318018],
                         [1.73679571], [1.14585410], [1.26678666], [1.12948590], [1.85589088]],
            (8, 3, 5): [[0.91399305, 0.48850400, 1.16412460, 0.12519177, 1.49823749],
                        [0.63099183, 0.44698110, 1.01368140, 0.09393353, 1.40649754],
                        [0.80280087, 0.43649257, 1.06788519, 0.10705599, 1.38236667],
                        [1.28461197, 0.60509998, 1.40882386, 0.17556120, 1.76553359],
                        [0.86501939, 0.46706368, 1.12932852, 0.11771889, 1.44762929],
                        [1.57857772, 0.92922991, 1.88814527, 0.25550533, 2.47470954],
                        [0.81650291, 0.58419886, 1.25005062, 0.13351854, 1.70162624],
                        [0.66628862, 0.39935920, 0.96420905, 0.08857088, 1.30613603]],
            (2, 9, 2): [[2.27259524, 3.55244122], [1.96233062, 3.61372585]]}
FC_dx_Ans = {(10, 4, 1): [[0.24109961, 0.29309018, 0.46785790, 0.04038948], [0.07563419, 0.09194390, 0.14676944, 0.01267039],
                          [0.04635002, 0.05634491, 0.08994301, 0.00776465], [0.11339898, 0.13785227, 0.22005266, 0.01899682],
                          [0.13075690, 0.15895323, 0.25373598, 0.02190465], [0.20499024, 0.24919421, 0.39778705, 0.03434037],
                          [0.15768071, 0.19168288, 0.30598210, 0.02641498], [0.35532820, 0.43195095, 0.68952042, 0.05952528],
                          [0.03668592, 0.04459684, 0.07118964, 0.00614570], [0.07509284, 0.09128582, 0.14571894, 0.01257970]],
             (8, 3, 5): [[1.04139565, 0.94706062, 1.18663975], [0.64918964, 0.78587550, 1.04506714], [0.85676850, 1.33839025, 1.89798028], [1.23557532, 1.58980048, 2.17885261],
                         [1.02779713, 1.30790016, 1.79651704], [0.72759426, 0.81968480, 1.11252186], [0.62382429, 0.73724675, 0.98413616], [0.98694446, 1.21720755, 1.65442036]],
             (2, 9, 2): [[0.81499753, 0.55547147, 0.92955084, 1.49124463, 0.80250664, 0.84830477, 0.52631198, 0.19131050, 0.57628182],
                         [0.08518027, 0.11706465, 0.13335865, 0.18824738, 0.02917710, 0.13168650,0.09127619, 0.03466832, 0.13130709]]}
FC_dw_Ans = {(10, 4, 1): [[0.20680191], [0.25928037], [0.16903349], [0.27952008]],
             (8, 3, 5): [[0.14995712, 0.21359406, 0.13443060, 0.15386216, 0.19217473],
                         [0.25639357, 0.30537321, 0.25206810, 0.30182445, 0.29389802],
                         [0.18765288, 0.18897341, 0.15708167, 0.17953237, 0.24446228]],
             (2, 9, 2): [[0.21156603, 0.21076436], [0.43193517, 0.39802230], [0.17294278, 0.14176457],
                         [0.21607914, 0.22500241], [0.41647549, 0.32752265], [0.38390616, 0.33463686],
                         [0.32860406, 0.25811084], [0.06000919, 0.10513163], [0.43952939, 0.39032876]]}
FC_db_Ans = {(10, 4, 1): [0.39971795], (8, 3, 5): [0.49955308, 0.57976758, 0.51414666, 0.57931643, 0.57737398], (2, 9, 2): [0.47996011, 0.43455428]}
SCE_pred_Ans = {(10, 4): [1, 3, 0, 1, 3, 0, 3, 3, 3, 2], (8, 3): [0, 2, 1, 2, 1, 2, 2, 2], (2, 9): [0, 6]}
SCE_loss_Ans = {(10, 4): 1.22930837, (8, 3): 1.17401948, (2, 9): 2.20667323}
SCE_B_Ans = {(10, 4): [[ 0.02362655,  0.02790336, -0.07506379,  0.02353388], [ 0.02057661,  0.02569752,  0.02086530, -0.06713943],
                       [-0.06720474,  0.01835795,  0.02761465,  0.02123214], [ 0.02734938, -0.06089517,  0.01663793,  0.01690786],
                       [ 0.01294272, -0.07083600,  0.02761812,  0.03027516], [ 0.03074802,  0.02569678,  0.01833270, -0.07477750],
                       [ 0.01667984, -0.07189781,  0.01710344,  0.03811452], [ 0.02526151, -0.07730612,  0.01953073,  0.03251388],
                       [ 0.02538185,  0.02839797,  0.01639004, -0.07016986], [ 0.02236971,  0.02247820, -0.06883299,  0.02398509]],
             (8, 3): [[ 0.05648762,  0.03565061, -0.09213823], [-0.08757070,  0.03928081,  0.04828989], [ 0.03626254, -0.06216216,  0.02589962], [ 0.03558377,  0.03393077, -0.06951454],
                      [ 0.03871669,  0.04790836, -0.08662505], [ 0.03475174, -0.09189662,  0.05714488], [-0.08734283,  0.03992181,  0.04742102], [ 0.04996088, -0.10077589,  0.05081502]],
             (2, 9): [[ 0.08716825,  0.04328319,  0.05522322, -0.45306455,  0.05596446, 0.04686656,  0.05720034,  0.06298141,  0.04437712],
                      [-0.43547217,  0.05689605,  0.04209858,  0.05448212,  0.03546566, 0.05743024,  0.08177083,  0.04439795,  0.06293074]]}

def ReLU_Tests(layer):
    pass_tests = True
    np.random.seed(0)
    shapes = [(10,4), (8,3), (2,9)]
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
    shapes = [(10,4,1), (8,3,5), (2,9,2)]

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
        dx, dw, db = layer.backward(torch.from_numpy(np.random.rand(N, F)))
        truth_dx = FC_dx_Ans[test_shape]
        if (rel_err(torch.tensor(truth_dx), dx) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_dx} but got {dx.cpu().numpy()} during backward of weights')
            pass_tests = False
        truth_dw = FC_dw_Ans[test_shape]
        if (rel_err(torch.tensor(truth_dw), dw) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_dw} but got {dw.cpu().numpy()} during backward of weights')
            pass_tests = False
        truth_db = FC_db_Ans[test_shape]
        if (rel_err(torch.tensor(truth_db), db) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_db} but got {db.cpu().numpy()} during backward of bias')
            pass_tests = False
    if pass_tests:
        print('Results of fully connected layer forward and backward tests: All passed.')

def SCE_Tests(layer):
    pass_tests = True
    np.random.seed(0)
    shapes = [(10,4), (8,3), (2,9)]
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

def Network_Test(params, params_grad):
    pass_tests = True
    truth_p = {'w1': np.array([[ 1.41124188,  0.32012577,  0.78299039,  1.79271456,  1.49404639, -0.78182230,  0.76007073, -0.12108577],
                               [-0.08257508,  0.32847880,  0.11523486,  1.16341881,  0.60883018,  0.09734001,  0.35509059,  0.26693946],
                               [ 1.19526326, -0.16412661,  0.25045416, -0.68327659, -2.04239185,  0.52289488,  0.69154896, -0.59373202],
                               [ 1.81580370, -1.16349254,  0.03660681, -0.14974708,  1.22622337,  1.17548702,  0.12395794,  0.30253002],
                               [-0.71022860, -1.58463717, -0.27832972,  0.12507918,  0.98423254,  0.96190388, -0.30986145, -0.24184220]], dtype=np.float64),
               'b1': np.array([0., 0., 0., 0., 0., 0., 0., 0.]),
               'w2': np.array([[-0.83884237, -1.13601435, -1.36501615,  1.56062032], [-0.40772175, -0.35045944, -1.00223629,  0.62199228],
                               [-1.29111828, -0.17019222, -0.71637325,  0.30952200], [-0.40864411, -0.94450575, -0.02254578,  0.34266550],
                               [ 0.05321378,  0.24197752, -0.50745767, -0.29019293], [-0.53796836, -0.28764253, -0.65051703, -1.38102608],
                               [ 0.14194091, -0.32142475, -1.30415868,  0.37022580], [-0.72583869,  0.04155632,  0.58327245,  0.10318633]], dtype=np.float64),
               'b2': np.array([0., 0., 0., 0.]),
               'w3': np.array([[ 0.91152055, -0.98786066], [ 0.32187331, -0.54784807], [-0.69663772, -0.46307973], [-0.24924203,  0.04493227]], dtype=np.float64),
               'b3': np.array([0., 0.]),
               'w4': np.array([[-0.93211987,  0.72066119,  0.37252995], [-1.22899495,  1.19060176,  1.51671134]], dtype=np.float64), 'b4': np.array([0., 0., 0.])}
    truth_g = {'w4': np.array([[ 0.00000000,  0.00000000,  0.00000000], [-0.04815858,  0.02312220,  0.02503638]], dtype=np.float64), 'b4': np.array([-0.11412154, -0.07186340,  0.18598495], dtype=np.float64),
               'w3': np.array([[0.00000000, 0.00000000], [0.00000000, 0.00000000], [0.00000000, 0.00000000], [0.00000000, 2.77504208]], dtype=np.float64),
               'b3': np.array([0.00000000, 0.51123684], dtype=np.float64),
               'w2': np.array([[0.00000000, 0.00000000, 0.00000000, 0.14630114], [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                               [0.00000000, 0.00000000, 0.00000000, 0.02836261], [0.00000000, 0.00000000, 0.00000000, 0.06563443],
                               [0.00000000, 0.00000000, 0.00000000, 0.12556390], [0.00000000, 0.00000000, 0.00000000, 0.08274408],
                               [0.00000000, 0.00000000, 0.00000000, 0.04269934], [0.00000000, 0.00000000, 0.00000000, 0.00000000]], dtype=np.float64),
               'b2': np.array([0.00000000, 0.00000000, 0.00000000, 0.02297103], dtype=np.float64),
               'w1': np.array([[ 0.06323962,  0.00000000,  0.01254248,  0.01388553, -0.01175923, -0.05596209,  0.01500233,  0.00000000],
                               [ 0.01434526,  0.00000000,  0.00284513,  0.00314979, -0.00266746, -0.01269443,  0.00340312,  0.00000000],
                               [ 0.03508684,  0.00000000,  0.00695887,  0.00770402, -0.00652430, -0.03104909,  0.00832365,  0.00000000],
                               [ 0.08033392,  0.00000000,  0.01593284,  0.01763892, -0.01493787, -0.07108919,  0.01905761,  0.00000000],
                               [ 0.06695020,  0.00000000,  0.01327841,  0.01470026, -0.01244920, -0.05924566,  0.01588259,  0.00000000]], dtype=np.float64),
               'b1': np.array([ 0.03584906,  0.00000000,  0.00711004,  0.00787138, -0.00666603, -0.03172360,  0.00850447,  0.00000000], dtype=np.float64)}
    for key, val in params.items():
        if (rel_err(torch.from_numpy(truth_p[key]), val) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_p[key]} but got {val.cpu().numpy()} of {key}')
            pass_tests = False
    for key, val in params_grad.items():
        if (rel_err(torch.from_numpy(truth_g[key]), val) < 1e-6).cpu().numpy():
            pass
        else:
            print(f'Expected {truth_g[key]} but got {val.cpu().numpy()} of {key}')
            pass_tests = False
    if pass_tests:
        print('Test of network: Passed.')

def SGD_Test(optim):
    pass_test = True
    truth_w = [torch.tensor([[-0.39940000, -0.34682105, -0.29424211, -0.24166316, -0.18908421],
                             [-0.13650526, -0.08392632, -0.03134737,  0.02123158,  0.07381053],
                             [ 0.12638947,  0.17896842,  0.23154737,  0.28412632,  0.33670526],
                             [ 0.38928421,  0.44186316,  0.49444211,  0.54702105,  0.59960000]]),
               torch.tensor([[-0.39880000, -0.34627368, -0.29374737, -0.24122105, -0.18869474],
                             [-0.13616842, -0.08364211, -0.03111579,  0.02141053,  0.07393684],
                             [ 0.12646316,  0.17898947,  0.23151579,  0.28404211,  0.33656842],
                             [ 0.38909474,  0.44162105,  0.49414737,  0.54667368,  0.59920000]])]
    params = {}; params_grad={}
    params['w'] = torch.linspace(-0.4, 0.6, 20, dtype=torch.float64).reshape(4,5)
    params_grad['w'] = torch.linspace(-0.6, 0.4, 20, dtype=torch.float64).reshape(4,5)
    for iter in range(2):
        optim.step(params, params_grad)
        if (rel_err(truth_w[iter], params['w']) < 1e-6).cpu().numpy():
            pass
        else:
            pass_test = False
            print(f'Expected {truth_w[iter]} for updated weights but got {params["w"]} in iteration {iter+1}')
    if pass_test:
        print('Test of SGD: Passed.')

def SGDM_Test(optim):
    pass_test = True
    truth_w = [torch.tensor([[-0.39940000, -0.34682105, -0.29424211, -0.24166316, -0.18908421],
                             [-0.13650526, -0.08392632, -0.03134737,  0.02123158,  0.07381053],
                             [ 0.12638947,  0.17896842,  0.23154737,  0.28412632,  0.33670526],
                             [ 0.38928421,  0.44186316,  0.49444211,  0.54702105,  0.59960000]]),
               torch.tensor([[-0.39826000, -0.34578105, -0.29330211, -0.24082316, -0.18834421],
                             [-0.13586526, -0.08338632, -0.03090737,  0.02157158,  0.07405053],
                             [ 0.12652947,  0.17900842,  0.23148737,  0.28396632,  0.33644526],
                             [ 0.38892421,  0.44140316,  0.49388211,  0.54636105,  0.59884000]])]
    truth_v = [torch.tensor([[ 6.00000000e-04,  5.47368421e-04,  4.94736842e-04,  4.42105263e-04,  3.89473684e-04],
                             [ 3.36842105e-04,  2.84210526e-04,  2.31578947e-04,  1.78947368e-04,  1.26315789e-04],
                             [ 7.36842105e-05,  2.10526316e-05, -3.15789474e-05, -8.42105263e-05, -1.36842105e-04],
                             [-1.89473684e-04, -2.42105263e-04, -2.94736842e-04, -3.47368421e-04, -4.00000000e-04]]),
               torch.tensor([[ 1.14000000e-03,  1.04000000e-03,  9.40000000e-04,  8.40000000e-04,  7.40000000e-04],
                             [ 6.40000000e-04,  5.40000000e-04,  4.40000000e-04,  3.40000000e-04,  2.40000000e-04],
                             [ 1.40000000e-04,  4.00000000e-05, -6.00000000e-05, -1.60000000e-04, -2.60000000e-04],
                             [-3.60000000e-04, -4.60000000e-04, -5.60000000e-04, -6.60000000e-04, -7.60000000e-04]])]
    params = {}; params_grad={}
    params['w'] = torch.linspace(-0.4, 0.6, 20, dtype=torch.float64).reshape(4,5)
    params_grad['w'] = torch.linspace(-0.6, 0.4, 20, dtype=torch.float64).reshape(4,5)
    for iter in range(2):
        optim.step(params, params_grad)
        if (rel_err(truth_w[iter], params['w']) < 1e-6).cpu().numpy():
            pass
        else:
            pass_test = False
            print(f'Expected {truth_w[iter]} for updated weights but got {params["w"]} in iteration {iter+1}')
        if (rel_err(truth_v[iter], optim.velocity['w']) < 1e-6).cpu().numpy():
            pass
        else:
            pass_test = False
            print(f'Expected {truth_v[iter]} for velocity but got {optim.velocity["w"]} in iteration {iter+1}')
    if pass_test:
        print('Test of SGD_Momentum: Passed.')

def Adam_Test(optim):
    pass_test = True
    truth_w = [torch.tensor([[-0.39900000, -0.34636842, -0.29373684, -0.24110526, -0.18847368],
                             [-0.13584211, -0.08321053, -0.03057895,  0.02205263,  0.07468421],
                             [ 0.12731579,  0.17994737,  0.23057895,  0.28321053,  0.33584211],
                             [ 0.38847368,  0.44110526,  0.49373684,  0.54636842,  0.59900000]]),
               torch.tensor([[-0.39800000, -0.34536842, -0.29273684, -0.24010526, -0.18747368],
                             [-0.13484211, -0.08221053, -0.02957895,  0.02305263,  0.07568421],
                             [ 0.12831579,  0.18094737,  0.22957895,  0.28221053,  0.33484211],
                             [ 0.38747368,  0.44010526,  0.49273684,  0.54536842,  0.59800000]])]
    truth_m = [torch.tensor([[-0.06000000, -0.05473684, -0.04947368, -0.04421053, -0.03894737],
                             [-0.03368421, -0.02842105, -0.02315789, -0.01789474, -0.01263158],
                             [-0.00736842, -0.00210526,  0.00315789,  0.00842105,  0.01368421],
                             [ 0.01894737,  0.02421053,  0.02947368,  0.03473684,  0.04000000]]),
               torch.tensor([[-0.11400000, -0.10400000, -0.09400000, -0.08400000, -0.07400000],
                             [-0.06400000, -0.05400000, -0.04400000, -0.03400000, -0.02400000],
                             [-0.01400000, -0.00400000,  0.00600000,  0.01600000,  0.02600000],
                             [ 0.03600000,  0.04600000,  0.05600000,  0.06600000,  0.07600000]])]
    truth_v = [torch.tensor([[3.60000000e-04, 2.99612188e-04, 2.44764543e-04, 1.95457064e-04, 1.51689751e-04],
                             [1.13462604e-04, 8.07756233e-05, 5.36288089e-05, 3.20221607e-05, 1.59556787e-05],
                             [5.42936288e-06, 4.43213296e-07, 9.97229917e-07, 7.09141274e-06, 1.87257618e-05],
                             [3.59002770e-05, 5.86149584e-05, 8.68698061e-05, 1.20664820e-04, 1.60000000e-04]]),
               torch.tensor([[7.19640000e-04, 5.98924765e-04, 4.89284321e-04, 3.90718670e-04, 3.03227812e-04],
                             [2.26811745e-04, 1.61470471e-04, 1.07203989e-04, 6.40122992e-05, 3.18954017e-05],
                             [1.08532964e-05, 8.85983380e-07, 1.99346260e-06, 1.41757341e-05, 3.74327978e-05],
                             [7.17646537e-05, 1.17171302e-04, 1.73652742e-04, 2.41208975e-04, 3.19840000e-04]])]
    params = {}; params_grad={}
    params['w'] = torch.linspace(-0.4, 0.6, 20, dtype=torch.float64).reshape(4,5)
    params_grad['w'] = torch.linspace(-0.6, 0.4, 20, dtype=torch.float64).reshape(4,5)
    for iter in range(2):
        optim.step(params, params_grad)
        if (rel_err(truth_w[iter], params['w']) < 1e-6).cpu().numpy():
            pass
        else:
            pass_test = False
            print(f'Expected {truth_w[iter]} for updated weights but got {params["w"]} in iteration {iter+1}')
        if (rel_err(truth_m[iter], optim.momentum['w']) < 1e-6).cpu().numpy():
            pass
        else:
            pass_test = False
            print(f'Expected {truth_m[iter]} for momentum but got {optim.momentum["w"]} in iteration {iter+1}')
        if (rel_err(truth_v[iter], optim.velocity['w']) < 1e-6).cpu().numpy():
            pass
        else:
            pass_test = False
            print(f'Expected {truth_v[iter]} for velocity but got {optim.velocity["w"]} in iteration {iter+1}')
    if pass_test:
        print('Test of Adam: Passed.')

def plot_curves(cand, curve_type, train, valid):
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['font.size'] = 12
    plt.subplot(1, 2, 1)
    for idx in range(len(train)):
        plt.plot(train[idx], label=str(cand[idx]))
    plt.title(f'Training {curve_type} log')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    for idx in range(len(valid)):
        plt.plot(valid[idx], label=str(cand[idx]))
    plt.title(f'Validation {curve_type} log')
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

def load_small_dataset(train_ratio=0.8, valid_ratio=0.2):
    dataset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True)
    images = dataset.data[:5000]
    labels = np.array(dataset.targets[:5000])
    num_train = int(5000 * train_ratio)
    train_dataset = Dataset(images[:num_train], labels[:num_train])
    valid_dataset = Dataset(images[num_train:], labels[num_train:])
    return train_dataset, valid_dataset