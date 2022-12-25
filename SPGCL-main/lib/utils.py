import os
from numbers import Number
import logging
import math
import lib.layers as layers
import torch
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False, main_file=False):
    logger = logging.getLogger(__name__)
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        if saving:
            info_file_handler = logging.FileHandler(logpath, mode="a")
            info_file_handler.setLevel(level)
            logger.addHandler(info_file_handler)
        if displaying:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            logger.addHandler(console_handler)
    # logger.propagate = False
    # logger.info(filepath)
    if main_file:
        with open(filepath, "r") as f:
            logger.info(f.read())

        for f in package_files:
            logger.info(f)
            with open(f, "r") as package_f:
                logger.info(package_f.read())

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


def isnan(tensor):
    return (tensor != tensor)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def standard_normal_logprob(z):
    # density of z in a standard normal distribution
    logZ = -0.5 * math.log(2 * math.pi)
    logZ = logZ - z.pow(2) / 2
    return logZ


def standard_uniform_logprob(z):
    return torch.zeros(z.shape).to(z)


def count_nfe(model):

    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):

    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def build_model_tabular(args, dims, regularization_fns=[]):
    """
    args.num_blocks: number of cnf
    """
    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            rademacher=args.rademacher,
        )
        # the func that gets input
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    model = layers.SequentialFlow(chain)

    return model


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MAPE(labels, predicts, mask):
    """
        Mean absolute percentage. Assumes ``y >= 0``.
        do mask when zero exists in matrix, for the zero element will lead to huge results
        Defined as ``(y - y_pred).abs() / y.abs()``
    """
    loss = np.abs(predicts - labels) / (np.abs(labels) + 1)
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss)/non_zero_len


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def evaluation(labels, predicts, args, acc_threshold=0.05):
    """
    evalution the labels with predicts
    rmse, Root Mean Squared Error
    mae，mean_absolute_error
    F_norm， Frobenius norm
    Args:
        labels:
        predicts:
        acc_threshold: if lower than this threshold we regard it as accurate
    Returns:
    """
    labels = labels.reshape([-1, args.pre_len])
    predicts = predicts.reshape([-1, args.pre_len])
    rmse = mean_squared_error(y_true=labels, y_pred=predicts, squared=True)
    mae = mean_absolute_error(y_true=labels, y_pred=predicts)
    r2 = r2_score(y_true=labels, y_pred=predicts)
    evs = explained_variance_score(y_true=labels, y_pred=predicts)
    a, b = labels, predicts
    acc = a[np.abs(a - b) < np.abs(acc_threshold)]
    acc = np.size(acc) / np.size(a)
    return rmse, mae, acc, r2, evs


def x2z(x, model, memory=100):
    # flow x to z
    sample_z = []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    zeros_std = torch.zeros(x.shape[0], 1).to(x)
    with torch.no_grad():
        for ii in torch.split(inds, int(memory ** 2)):
            sample_z.append(model.cnf(x[ii], zeros_std, reverse=False))
    sample_z = torch.cat(sample_z, 0)
    return model.shape_trans(sample_z)


def x2z_with_deformation(x, model, deformation, memory=100):
    # flow x to z
    sample_z = []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    zeros_std = torch.zeros(x.shape[0], 1).to(x)
    de = torch.zeros(x.shape[0], 2).to(x)
    with torch.no_grad():
        for ii in torch.split(inds, int(memory ** 2)):
            sample_z.append(model.cnf(x[ii] + torch.cat((de, deformation), 1), zeros_std, reverse=False))
    sample_z = torch.cat(sample_z, 0)
    return model.shape_trans(sample_z)


def re_r2(x):
    return 1/(1+(np.exp((-x))))


def re_var(x):
    a = np.exp(x)
    b = np.exp((-x))
    return ((a-b) / (a+b) + 1) / 2


def nor_r2evs(x, y):
    r2 = 1/(1+(np.exp((-x))))
    x = y
    var = 1/(1+(np.exp((-y))))
    return r2, var
