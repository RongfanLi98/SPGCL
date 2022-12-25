import torch
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors


def dense_adj_from_mean(loc, args, normed=False):
    path = os.path.join(args.data_dir, args.data, "{}_dense_adj.npy".format(args.data))
    if os.path.exists(path):
        A = np.load(path)
        return torch.from_numpy(A)
    else:
        Dis = torch.cdist(loc, loc, p=2).float()  # distance matrix
        m = Dis.mean()

        A = torch.eye(Dis.size(0), dtype=torch.float32)
        A[Dis < m] = 1
        A[A == 0] = float('-inf')

        remain_persent = (A[A == 1].sum() / (A.shape[0] * A.shape[1])).item()
        print("Sparsity of A(0 is empty):", remain_persent * 100, "%")
        print("Cut persent:", 100 - remain_persent * 100, "%")

        if normed:
            # degree matrix
            degree = A.sum(1)
            D = torch.diag(torch.pow(degree, -0.5))
            normed_A = D.mm(A).mm(D)
            np.save(path, normed_A)
            return normed_A
        else:
            np.save(path, A)
            return A


def sparse_adj_from_mean(loc, args, normed=True):
    path = os.path.join(args.data_dir, args.data, "{}_sparse_adj.npy".format(args.data))
    if os.path.exists(path):
        A = np.load(path)
    else:
        Dis = torch.cdist(loc, loc, p=2).float()  # distance matrix
        m = Dis.mean()

        A = torch.eye(Dis.size(0), dtype=torch.float32)
        A[Dis < m] = 1
        np.save(path, A)

    remain_persent = (A.sum() / (A.shape[0] * A.shape[1])).item()
    print("Sparsity of A(0 is empty):", remain_persent * 100, "%")
    print("Cut persent:", 100 - remain_persent * 100, "%")

    edge_index = torch.from_numpy(A).to_sparse()
    return edge_index


def sparse_adj_from_KNN(loc, args):
    path = os.path.join(args.data_dir, args.data, "{}_KNN_adj.npy".format(args.data))
    # if file exists:
    if os.path.exists(path):
        A = np.load(path)
    else:
        # Find the nearest k points
        nearestneighbors_euclidean = NearestNeighbors(
            n_neighbors=args.positive_init + 1, metric="euclidean").fit(loc)
        A = nearestneighbors_euclidean.kneighbors_graph(loc).toarray().T
        np.save(path, A)

    remain_persent = (A.sum() / (A.shape[0] * A.shape[1])).item()
    print("Sparsity of A(0 is empty):", remain_persent * 100, "%")
    print("Cut persent:", 100 - remain_persent * 100, "%")

    edge_index = torch.from_numpy(A).to_sparse()
    return edge_index


def sparse_adj_from_KFF(loc, args):
    path = os.path.join(args.data_dir, args.data, "{}_KFF_adj.npy".format(args.data))
    # if file exists:
    if os.path.exists(path):
        A = np.load(path)
    else:
        # loc = loc.cpu()
        # Find the nearest k points
        nearestneighbors_euclidean = NearestNeighbors(
            n_neighbors=loc.shape[0], metric="euclidean").fit(loc)
        A = nearestneighbors_euclidean.kneighbors_graph(loc, mode="distance").toarray()
        B = A.copy()
        B.sort()
        boarder = B[:, -args.negative_init]
        for (i, item) in enumerate(A):
            A[i] = np.where(A[i] >= boarder[i], np.ones_like(A[i]), np.zeros_like(A[i]))
        A = A.T
        np.save(path, A)

    remain_persent = (A.sum() / (A.shape[0] * A.shape[1])).item()
    print("Sparsity of A(0 is empty):", remain_persent * 100, "%")
    print("Cut persent:", 100 - remain_persent * 100, "%")

    edge_index = torch.from_numpy(A).to_sparse()
    return edge_index
