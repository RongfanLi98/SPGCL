import torch
import math
import os
import copy
import numpy as np
from lib.utils import get_logger, evaluation, makedirs, MAPE, masked_mape_np
from torch.utils.tensorboard import SummaryWriter
from lib.adj_from_loc import dense_adj_from_mean, sparse_adj_from_mean, sparse_adj_from_KNN, sparse_adj_from_KFF
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import time
import pandas as pd


class SPGCLTrainer(object):
    """
    Standard SPGCL trainer
    """

    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args):
        super(SPGCLTrainer, self).__init__()
        self.model = model
        self.predict_loss = loss
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.temperature = args.temperature
        self.locs = None
        self.no_update_pos = 0
        self.no_update_neg = 0
        self.dynamic_delta = args.delta
        self.dynamic_delta_negative = args.delta_negative

        # The largest number of edges: N * density > init + positive_per_K * round
        self.round_max = int((self.args.eta * self.args.num_nodes - self.args.positives) / self.args.positive_per_K)

        if self.args.sparse:  # mask does not need gradient
            self.graph = train_loader.dataset.dataset.edge_index.to(self.args.device).detach()
            self.graph_negative = train_loader.dataset.dataset.edge_index_negative.to(self.args.device).detach()
        else:
            self.graph = dense_adj_from_mean(self.locs, args).to(self.args.device).detach()
        # init positive graph and negative graph
        self.init_graph = self.graph
        self.init_n_graph = self.graph_negative

        self.graph_times = torch.zeros([self.graph.shape[0], self.graph.shape[0]]).cuda()
        self.scaler = scaler
        self.train_per_epoch = len(train_loader)

        if val_loader is not None and test_loader is not None:
            self.val_per_epoch = len(val_loader) if self.val_loader else len(test_loader)
        self.best_path = os.path.join(self.args.save_dir, 'best_model.pth')

        if self.args.writer and self.args.mode == "train":
            runs_dir = os.path.join(self.args.save_dir, 'runs', time.strftime("%m%d-%H-%M"))
            makedirs(runs_dir)
            self.writer = SummaryWriter(runs_dir)
            self.embedding_e_counter = 1
        self.pre_counter = 1
        self.K_counter = 0
        self.K_Round_counter = 0

        self.logger = get_logger(logpath=self.args.log_dir, filepath=os.path.abspath(__file__))
        self.logger.info('Experiment log path in: {}'.format(self.args.log_dir))
        if not args.debug:
            self.logger.info(args)
            parameters = self.return_parameters()
            log_message = "Total parameters is {}.".format(parameters)
            self.logger.info(log_message)
        else:
            self.args.epochs = 2
            self.args.log_freq = 1
            self.args.val_freq = 1
            self.args.early_stop_patience = 5
            self.round_max = 2

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_e_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, xx, label) in enumerate(val_dataloader):
                # x = [B, N, N, seq_len] -> [B * N , N, relative_seq_len], y = [B, N, pre_len] -> [B * N , pre_len]
                x = x.reshape([-1, self.args.num_nodes, self.args.seq_len])
                xx = xx.reshape([self.args.num_nodes, self.args.ini_seq_len]) 
                label = label.reshape([-1, self.args.pre_len])

                # Y_pre = [N, pre_len]
                Y_pre = self.model([xx, x, self.graph])
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)

                eloss = self.predict_loss(input=Y_pre, target=label)

                if not torch.isnan(eloss):
                    total_e_loss += eloss.item()

        val_e_loss = total_e_loss / len(val_dataloader)
        self.logger.info('Val Epoch {}: average Loss: {:.4f}'.format(epoch, val_e_loss))
        return val_e_loss

    def train_epoch(self, epoch, mode="regular"):
        # Given grpah and input, get the prediction and scores on all nodes, and calculate the total SPGCL_LOSS.
        self.model.train()
        total_e_loss = 0.0
        total_CL_loss = 0.0
        total_PU_loss = 0.0
        total_p_loss = 0.0
        scores = torch.zeros(self.graph.size()).to(self.args.device)

        for batch_idx, (x, xx, label) in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            # x = [B, N, N, seq_len] -> [B * N , N, relative_seq_len], y = [B, N, pre_len] -> [B * N , pre_len]
            x = x.reshape([-1, self.args.num_nodes, self.args.seq_len])  # x is relationship features
            xx = xx.reshape([self.args.num_nodes, self.args.ini_seq_len])  # xx is absolute features
            label = label.reshape([-1, self.args.pre_len])
            self.optimizer.zero_grad()

            x, xx, label = self.augmentation([x, xx, label])  # data augmentation

            if mode == "regular":
                # Y_pre = [N, pre_len]
                Y_pre, score_matrix = self.model([xx, x, self.graph], require_socre=True)

                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                ploss = self.predict_loss(input=Y_pre, target=label)  # ploss is predict loss
                CL_loss = self.contrastive_loss(self.graph, graph_negative=self.graph_negative,
                                                scores=(score_matrix + 1) / 2,
                                                args=self.args)
                PU_loss = self.PU_loss(self.graph, score_matrix, eta_p=self.args.eta)
                eloss = CL_loss + self.args.gamma_1 * PU_loss + self.args.gamma_2 * ploss
                scores = (score_matrix + 1) / 2

            elif mode == "prediction":
                # train prediction
                Y_pre = self.model.get_prediction([xx, x, self.graph])
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                eloss = self.predict_loss(input=Y_pre, target=label)
            elif mode == "embeddings":
                # train embedding learning
                x, r_embeddings, score_matrix = self.model.get_embedding([xx, x, self.graph])
                score_act = (score_matrix + 1) / 2
                CL_loss = self.contrastive_loss(self.graph, graph_negative=self.graph_negative, scores=score_act,
                                                args=self.args)
                PU_loss = self.PU_loss(self.graph, score_matrix, eta_p=self.args.eta)
                eloss = CL_loss + self.args.gamma_1 * PU_loss
                scores += (score_act.detach() + 1) / 2
            if self.args.mode != "test":
                eloss.backward()
            total_e_loss += eloss.item()
            if mode == "embeddings":
                total_CL_loss += CL_loss.item()
                total_PU_loss += PU_loss.item()
            if mode == "regular":
                total_p_loss += ploss.item()
                total_CL_loss += CL_loss.item()
                total_PU_loss += PU_loss.item()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            if batch_idx % self.args.log_freq == 0 and batch_idx > 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_idx + 1, self.train_per_epoch,
                                                                             eloss.item()))

        total_e_loss /= self.train_per_epoch
        total_CL_loss /= self.train_per_epoch
        total_PU_loss /= self.train_per_epoch
        total_p_loss /= self.train_per_epoch
        if mode == "embeddings":
            scores /= self.train_per_epoch
            return total_e_loss, scores, total_CL_loss, total_PU_loss
        if mode == "prediction":
            return total_e_loss
        return scores, total_e_loss, total_CL_loss, total_PU_loss, total_p_loss

    def train(self, args):
        if self.args.conti and os.path.exists(self.best_path):
            print("Loading from", self.best_path)
            check_point = torch.load(self.best_path)
            self.model.load_state_dict(check_point['state_dict'])
            self.args = args
            self.model.to(self.args.device)
        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        early_stop = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()

        # using warm up
        # self.train_embeddings()
        # print("After embedding tarning, the graph contains {} edges, and each node has average {} edges.
        # ".format(self.graph._nnz(), int(self.graph._nnz()/self.graph.size()[0])))
        # self.train_prediction()

        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()
            scores, epoch_losses, epoch_CL_losses, epoch_PU_losses, epoch_ploss = self.train_epoch(epoch,
                                                                                                   mode="regular")

            train_loss_list.append(epoch_losses)
            epoch_end = time.time()

            if epoch_losses > 1e20:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            self.logger.info(
                'All--Epoch {}: averaged Loss: {:.4f} time: {} '.format(epoch, epoch_losses, epoch_end - epoch_start))

            if self.args.writer:
                self.writer.add_scalar('Train/total_loss', epoch_losses, self.pre_counter)
                self.writer.add_scalar('Train/CL_loss', epoch_CL_losses, self.pre_counter)
                self.writer.add_scalar('Train/PU_loss', epoch_PU_losses, self.pre_counter)
                self.writer.add_scalar('Train/Predict_loss', epoch_ploss, self.pre_counter)
            self.pre_counter += 1

            if epoch % self.args.val_freq == 0:
                if not self.val_loader:
                    val_dataloader = self.test_loader
                else:
                    val_dataloader = self.val_loader

                val_epoch_loss = self.val_epoch(self.pre_counter, val_dataloader)
                val_loss_list.append(val_epoch_loss)
                if self.args.writer:
                    self.writer.add_scalar('Valid/predict_loss', val_epoch_loss, epoch)

                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    early_stop = 0
                    best_state = True
                    print("Current best {}, validation SPGCL_LOSS {}".format(epoch_losses, val_epoch_loss))
                elif early_stop < self.args.early_stop_patience:
                    early_stop += 1
                    best_state = False
                else:
                    self.logger.info("Validation performance didn\'t improve for {} epochs.".format(early_stop))
                    break
                if best_state:
                    best_model = copy.deepcopy(self.model.state_dict())
                    self.save_checkpoint()
                    print("save at ", epoch)

            if epoch % self.args.update_freq == 0:  # the frequency to update graph
                # please refer to algorithm 2 to get more details
                # slack standard, by self.round_max make the density of graph to approach eta
                self.dynamic_delta = self.args.delta - (self.args.delta - 1 + self.args.eta) / \
                                     (self.args.epochs + 1) * epoch
                self.dynamic_delta_negative = self.args.delta_negative + (self.args.eta - self.args.delta_negative) / \
                                              (self.args.epochs + 1) * epoch
                # update positive edges set and negative edges set
                update_nodes = self.update_graph(scores=scores)
                update_nodes_negative = self.update_graph_add_negative(scores=scores)
                # check if there exists edges which are mistakenly labelled and move them to out
                self.check_node(scores, lambda_K=10)

                pd.DataFrame(np.array(
                    [epoch, self.graph.to_dense().shape[0] ** 2, self.graph._nnz(), self.graph_negative._nnz(),
                     update_nodes, update_nodes_negative, self.args.delta, self.args.delta_negative]).reshape(1, 8)) \
                    .to_csv(args.save_dir + r'\{}_graph_data_change.csv'.format(args.data), mode='a+', header=None)

                all_edges = self.graph.shape[0] ** 2
                p_edges = self.graph._nnz()
                n_edges = self.graph_negative._nnz()
                self.logger.info(
                    'DP:{}({:.2f}%), DN:{}, Rest edges:{}'.format(p_edges, 100 * p_edges / all_edges, n_edges,
                                                                  all_edges - p_edges - n_edges))

        training_time = time.time() - start_time
        self.logger.info(
            "Total training time: {:.4f}min, best predict_loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        self.model.load_state_dict(best_model)
        self.save_checkpoint()
        self.logger.info('Current best model saved!')
        normed_judge, real_judge, pems_result = self.test(self.model, self.args, self.test_loader, self.scaler)
        if self.args.writer:
            self.writer.close()
        return normed_judge, real_judge, pems_result

    def train_prediction(self):
        # train prediction until convergence
        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        early_stop = 0
        break_count = 0
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()
            epoch_losses = self.train_epoch(epoch, mode="prediction")
            epoch_end = time.time()

            if epoch_losses > 1e20:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            self.logger.info('Prediction--Epoch {}: averaged Loss: {:.4f} time: {} '.format(epoch, epoch_losses,
                                                                                            epoch_end - epoch_start))

            if self.args.writer:
                self.writer.add_scalar('Train_prediction/prediction_loss', epoch_losses, self.pre_counter)
            self.pre_counter += 1

            if epoch_losses < best_loss:
                best_loss = epoch_losses
                early_stop = 0
                best_state = True
                print("Current best", epoch_losses)
            elif early_stop < self.args.early_stop_patience:
                early_stop += 1
                best_state = False
            else:
                self.logger.info("Embedding performance didn\'t improve for {} epochs.".format(early_stop))
                break
                # save the best state
            if best_state:
                best_model = copy.deepcopy(self.model.state_dict())

        # save the best model to file
        self.model.load_state_dict(best_model)
        self.save_checkpoint()
        self.logger.info('Current best model saved!')

    def train_embeddings(self):
        if self.args.conti and os.path.exists(self.best_path):
            print("Loading from", self.best_path)
            check_point = torch.load(self.best_path)
            self.model.load_state_dict(check_point['state_dict'])
            self.model.to(self.args.device)
        # train embedding until convergence (warm up)
        best_model = copy.deepcopy(self.model.state_dict())
        break_count = 0
        train_loss_list = []
        start_time = time.time()
        K_best_loss = float('inf')
        K_early_stop = 0

        for K in range(self.round_max):
            early_stop = 0
            best_loss = float('inf')
            for epoch in range(1, self.args.epochs + 1):
                epoch_start = time.time()
                epoch_losses, scores, epoch_CL_losses, epoch_PU_losses = self.train_epoch(epoch, now_round=K,
                                                                                          mode="embeddings")
                train_loss_list.append(epoch_losses)
                epoch_end = time.time()

                if epoch_losses > 1e20:
                    self.logger.warning('Gradient explosion detected. Ending...')
                    break

                self.logger.info(
                    'Embedding--Train Round {} Epoch {}: averaged Loss: {:.4f} time: {}  Rest edges:{}'.format(
                        K, epoch, epoch_losses, epoch_end - epoch_start, self.graph.shape[
                            0] ** 2 - self.graph._nnz() - self.graph_negative._nnz()))

                if self.args.writer:
                    self.writer.add_scalar('Train_Embedding/Global_Embedding_loss', epoch_losses,
                                           self.embedding_e_counter)
                    self.writer.add_scalar('Train_Embedding/Global_CL_loss', epoch_CL_losses, self.embedding_e_counter)
                    self.writer.add_scalar('Train_Embedding/Global_PU_loss', epoch_PU_losses, self.embedding_e_counter)
                    self.writer.add_scalar('Embedding_loss/K{}'.format(self.K_Round_counter), epoch_losses,
                                           self.embedding_e_counter)
                    self.embedding_e_counter += 1

                if epoch_losses < best_loss:
                    best_loss = epoch_losses
                    early_stop = 0
                    best_state = True
                    print("Current best", epoch_losses)
                elif early_stop < self.args.early_stop_patience_K:
                    early_stop += 1
                    best_state = False
                else:
                    self.logger.info("Embedding performance didn\'t improve for {} epochs.".format(early_stop))
                    break
                    # save the best state
                if best_state:
                    best_model = copy.deepcopy(self.model.state_dict())

            # after updating a K graph
            self.K_Round_counter += 1
            if best_loss < K_best_loss:

                K_best_loss = best_loss
                K_early_stop = 0

                # slack standard, by self.round_max make the density of graph to approach eta
                self.dynamic_delta = self.args.delta - (self.args.delta - 1 + self.args.eta) / \
                                     (self.args.epochs + 1) * epoch
                self.dynamic_delta_negative = self.args.delta_negative + (self.args.eta - self.args.delta_negative) / \
                                              (self.args.epochs + 1) * epoch

                update_nodes = self.update_graph(scores=scores)
                update_nodes_negative = self.update_graph_add_negative(scores=scores)
                self.check_node(scores)

                if update_nodes != 0:
                    self.no_update_pos = 0
                else:
                    self.no_update_pos += 1
                if update_nodes_negative != 0:
                    self.no_update_neg = 0
                else:
                    self.no_update_neg += 1

                if self.no_update_pos >= self.args.lambda_p or self.no_update_neg >= self.args.lambda_n:
                    self.logger.info('Positive edges {} turns no update, Negative edges {} turns no update'.format(
                        self.no_update_pos, self.no_update_neg))
                    break
                if update_nodes < self.args.min_update_nodes:
                    print("K_early_stop + 1 since update nodes are less than threshold.")
                    K_early_stop += 1  # early stop
                K_best_loss += 2

                self.model.load_state_dict(best_model)
                self.save_checkpoint()
                self.logger.info('{} round best model saved!'.format(K))

            elif K_early_stop < 4:
                K_early_stop += 1
            else:
                break

        g = self.graph.to_dense()
        g = torch.where(scores < 0.8, torch.zeros_like(g), g) + self.init_graph.to_dense()
        # Avoid value > 1
        g = torch.where(g > 1, torch.ones_like(g), g)
        self.graph = g.to_sparse().coalesce()

        self.logger.info("Save embedding from {} graph. Cost {:.4}".format(K, time.time() - start_time))
        # save the best model to file
        self.model.load_state_dict(best_model)
        self.save_checkpoint()
        self.logger.info('Current best model saved!')

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args
        }
        torch.save(state, self.best_path)
        torch.save(state, os.path.join(self.args.save_dir, 'best_model_{}.pth'.format(self.pre_counter)))
        self.logger.info("Saving current best model to " + self.best_path)

    def return_parameters(self):
        # log net info
        size = 0
        for name, parameters in self.model.named_parameters():
            size += int(parameters.numel())
        return size

    def contrastive_loss(self, graph, graph_negative, scores, args=None, sparse=True):
        positive_index = graph.to_dense() if sparse else graph
        negative_index = graph_negative.to_dense() if sparse else graph_negative

        N = scores.shape[0]
        P_samples = args.positive_init - 1
        NU_samples = args.nu_samples
        NU_samples = N - P_samples if N - P_samples < NU_samples else NU_samples
        Q = args.Q  # U ratio
        assert Q <= 0.5

        scores = scores / self.temperature
        # sample from positive. randomly choose P_samples positive edges and return mask
        p_mask = torch.rand(size=(N, N)).cuda() * positive_index
        # [N, P_samples]
        _, indices = p_mask.topk(k=P_samples, dim=1, largest=True)
        positive_samples = scores[torch.arange(0, N).repeat(P_samples, 1).T, indices]

        # sample from NU. randomly  choose NU_samples negative and unlabelled edges and return mask
        nu_mask = torch.rand(size=(N, N)).cuda() * (1 - positive_index)
        # [N, NU_samples]
        _, indices = nu_mask.topk(k=NU_samples, dim=1, largest=True)
        nu_samples = scores[torch.arange(0, N).repeat(NU_samples, 1).T, indices]

        # [N, P_samples + NU_samples]
        all_samples = torch.cat([positive_samples, nu_samples], dim=1)
        target = torch.cat([torch.ones_like(positive_samples), torch.zeros_like(nu_samples)], dim=1)

        c_loss = self.cross_entropy(input=all_samples, target=target)  # pytorch > 1.10
        return c_loss

    def PU_loss(self, graph, scores, eta_p=0.4):
        # eta_p is the prior probability of positive edges

        def sigmoid_loss(input, reduction='elementwise_mean'):
            # y must be -1/+1
            # NOTE: torch.nn.functional.sigmoid is 1 / (1 + exp(-x)). BUT sigmoid SPGCL_LOSS should be 1 / (1 + exp(x))
            return torch.sigmoid(-input)

        mask = graph.to_dense()  # 1 means positive edges and 0 means negative edges
        positive_mask = (mask == 1)
        negative_mask = (mask == 0)

        p_size = positive_mask.sum(dim=0)  # for dim 0
        n_size = negative_mask.sum(dim=0)
        pu_loss_1 = (sigmoid_loss(scores) * positive_mask).sum(dim=0)
        pu_loss_2 = (sigmoid_loss(-scores) * negative_mask).sum(dim=0)
        pu_loss_3 = (sigmoid_loss(-scores) * positive_mask).sum(dim=0)

        tmp = 1 / n_size * pu_loss_2 - eta_p / p_size * pu_loss_3
        PU_loss = eta_p / p_size * pu_loss_1 + (1 - eta_p) / n_size * pu_loss_2 + torch.where(tmp > 0, tmp,
                                                                                              torch.zeros_like(tmp))
        PU_loss = PU_loss.sum() / scores.shape[0]
        return PU_loss

    def update_graph_add_negative(self, scores):
        # first, get rid of the edges which are already in positive and negative graphs
        scores = scores + self.graph.to_dense() + self.graph_negative.to_dense()
        scores = torch.where(scores > 1, torch.ones_like(scores), scores)  # make sure scores are smaller than 1

        # second, choose beta edges using topN and largest=False
        value_list, neighbor_list = torch.topk(scores.T, self.args.positive_per_K, dim=1, largest=False)

        # abscissa is idx [N, p_K],after transpose: ordinate is idx [p_K, N]-> [N * p_K, 1]

        value_list = value_list.T.reshape(self.args.num_nodes * self.args.positive_per_K)
        neighbor_list = neighbor_list.T.reshape(self.args.num_nodes * self.args.positive_per_K)
        idx_list = torch.arange(0, self.args.num_nodes).repeat(self.args.positive_per_K).to(self.args.device)

        # [2, new_E], be cautious than neighbors are in front of core node
        new_neighbors = torch.stack((neighbor_list[value_list < self.dynamic_delta_negative],
                                     idx_list[value_list < self.dynamic_delta_negative]), dim=0)
        new_neighbors = torch.sparse_coo_tensor(new_neighbors, torch.full_like(new_neighbors[0], 1),
                                                size=(self.args.num_nodes, self.args.num_nodes))

        # prohibit to absorb edges in positive and negative sets
        new_neighbors = new_neighbors - self.graph - self.graph_negative
        new_neighbors = new_neighbors.to_dense()
        new_neighbors = torch.where(new_neighbors < 0, torch.zeros_like(new_neighbors), new_neighbors)
        new_neighbors = new_neighbors.to_sparse()

        stage_mean = value_list.mean().item()
        self.logger.info(
            "INCLUDE {} : {}={}*{} edges are included in {} N-graph. Mean negative scores is {:.3f}".format(
                self.K_counter, new_neighbors._nnz(), value_list.shape[0], self.args.positive_per_K,
                self.graph.size()[0], stage_mean))
        self.graph_negative = (self.graph_negative + new_neighbors).coalesce()
        if self.K_counter == 0:
            torch.save(self.graph_negative,
                       os.path.join(self.args.graph_dir, "grpah_neg_per{}_0.pt".format(self.args.positive_per_K)))
        self.K_counter += 1
        torch.save(new_neighbors, os.path.join(self.args.graph_dir,
                                               "grpah_neg_per{}_{}.pt".format(self.args.positive_per_K,
                                                                              self.K_counter)))
        return new_neighbors._nnz()

    def update_graph(self, scores):
        # first, get rid of the edges which are already in positive and negative graphs
        scores_all_point = scores
        scores = scores - scores.sparse_mask(self.graph).to_dense() - scores.sparse_mask(self.graph_negative).to_dense()
        # [N, N] -> [N, neighbors]
        value_list, neighbor_list = torch.topk(scores.T, self.args.positive_per_K, dim=1, largest=True)

        # abscissa is idx [N, p_K], after transpose: ordinate is idx [p_K, N]-> [N * p_K, 1]
        value_list = value_list.T.reshape(self.args.num_nodes * self.args.positive_per_K)
        neighbor_list = neighbor_list.T.reshape(self.args.num_nodes * self.args.positive_per_K)
        idx_list = torch.arange(0, self.args.num_nodes).repeat(self.args.positive_per_K).to(self.args.device)

        # [2, new_E], be cautious than neighbors are in front of core node
        new_neighbors = torch.stack(
            (neighbor_list[value_list > self.dynamic_delta], idx_list[value_list > self.dynamic_delta]), dim=0)
        new_neighbors = torch.sparse_coo_tensor(new_neighbors, torch.full_like(new_neighbors[0], 1),
                                                size=(self.args.num_nodes, self.args.num_nodes))

        # prohibit to absorb edges in positive and negative sets
        new_neighbors = new_neighbors - self.graph - self.graph_negative
        new_neighbors = new_neighbors.to_dense()
        new_neighbors = torch.where(new_neighbors < 0, torch.zeros_like(new_neighbors), new_neighbors)
        new_neighbors = new_neighbors.to_sparse()

        # absorb nodes when score > delta
        value_list_new = value_list[value_list > self.dynamic_delta]
        self.args.stage_mean = (value_list_new.mean()).item()

        self.logger.info(
            "INCLUDE {}: {}={}*{} edges are included in {} graph. Mean positive scores is {:.3f}".
                format(self.K_counter, new_neighbors._nnz(), value_list.shape[0], self.args.positive_per_K,
                       self.graph.size()[0], self.args.stage_mean))
        self.graph = (self.graph + new_neighbors).coalesce()

        if self.K_counter == 0:
            torch.save(self.graph,
                       os.path.join(self.args.graph_dir, "grpah_per{}_0.pt".format(self.args.positive_per_K)))
            torch.save(torch.ones_like(scores), os.path.join(self.args.graph_dir, "score_per{}_0.pt".format(
                self.args.positive_per_K)))
            torch.save(scores_all_point, os.path.join(self.args.graph_dir, "all_score_per{}_0.pt".format(
                self.args.positive_per_K)))

        torch.save(new_neighbors, os.path.join(self.args.graph_dir,
                                               "grpah_per{}_{}.pt".format(self.args.positive_per_K, self.K_counter)))
        torch.save(scores, os.path.join(self.args.graph_dir, "score_per{}_{}.pt".format(self.args.positive_per_K,
                                                                                        self.K_counter)))
        torch.save(scores_all_point, os.path.join(self.args.graph_dir,
                                                  "all_score_per{}_{}.pt".format(self.args.positive_per_K,
                                                                                 self.K_counter)))
        return new_neighbors._nnz()

    def check_node(self, scores, lambda_K=10):
        # refer to Algorithm 2, step 13 to 18
        graph = self.graph.to_dense()
        graph_negative = self.graph_negative.to_dense()
        graph_times = self.graph_times

        false_positive = torch.where(scores < self.dynamic_delta, torch.ones_like(scores), torch.zeros_like(scores))
        false_positive = false_positive * graph

        false_negative = torch.where(scores > self.dynamic_delta_negative, torch.ones_like(scores),
                                     torch.zeros_like(scores))
        false_negative = false_negative * graph_negative

        # This epoch the nodes are mistakenly labeled will be labelled to 1
        graph_error = false_positive + false_negative

        # x + x * y, make sure the labels are labelled continuously
        graph_times = graph_error + graph_error * graph_times

        # check lambda_k: >lambda_k be 1; else be 0
        error_edges = torch.where(graph_times >= lambda_K, torch.ones_like(graph_times), torch.zeros_like(graph_times))

        positive2unlabel = error_edges * graph
        negative2unlabel = error_edges * graph_negative
        self.logger.info("New positive {}, negative {}.".format(positive2unlabel.sum(), negative2unlabel.sum()))

        # move these edges to unlabeled set
        graph = graph - positive2unlabel
        graph_negative = graph_negative - negative2unlabel

        # update graph_times
        graph_times = graph_times - lambda_K * (positive2unlabel + negative2unlabel)

        # make sure the ini graph by KNN are not removed
        graph += self.init_graph.to_dense()
        graph_negative += self.init_n_graph.to_dense()
        graph = torch.where(graph == 2, torch.ones_like(graph), graph)
        graph_negative = torch.where(graph_negative == 2, torch.ones_like(graph_negative), graph_negative)

        self.graph = graph.to_sparse()
        self.graph_negative = graph_negative.to_sparse()
        self.graph_times = graph_times

    def augmentation(self, data, augmentation_mode="GaussianNoise"):
        # Other augmentations are implemented in data loader
        x, xx, label = data
        if augmentation_mode == "GaussianNoise":
            scalers = [x.mean() / 20, xx.mean() / 20]
            x = x + torch.randn_like(x) * scalers[0]
            xx = xx + torch.randn_like(xx) * scalers[1]
        return x, xx, label

    def test(self, model, args, data_loader, scaler, path=None, mode="train"):
        if path:
            if os.path.exists(path):
                check_point = torch.load(path)
                state_dict = check_point['state_dict']
                # args = check_point['args']
                model.load_state_dict(state_dict)
                model.to(args.device)
                print("Load saved model")

        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (x, xx, label) in enumerate(data_loader):
                torch.cuda.empty_cache()
                x = x.reshape([-1, self.args.num_nodes, self.args.seq_len])
                xx = xx.reshape([self.args.num_nodes, self.args.ini_seq_len])
                label = label.reshape([-1, self.args.pre_len])
                Y_pre = model([xx, x, self.graph])

                # Save normed values
                if args.real_value:
                    Y_pre = scaler.transform(Y_pre)
                y_pred.append(Y_pre.cpu().numpy())
                y_true.append(label.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        real_y_true = scaler.inverse_transform(y_true)
        real_y_pred = scaler.inverse_transform(y_pred)
        rmse_ts_n, mae_ts_n, acc_ts_n, r2_ts_n, var_ts_n = evaluation(y_true, y_pred, args, args.acc_threshold)
        rmse_ts_r, mae_ts_r, acc_ts_r, r2_ts_r, var_ts_r = evaluation(real_y_true, real_y_pred, args,
                                                                      args.acc_real_threshold)

        self.logger.info("Testing results {} {} {} {} {}".format(rmse_ts_n, mae_ts_n, acc_ts_n, r2_ts_n, var_ts_n))
        self.logger.info("Testing results(r) {} {} {} {} {}".format(rmse_ts_r, mae_ts_r, acc_ts_r, r2_ts_r, var_ts_r))

        if "PEMS" in args.save_dir:
            y_true = np.transpose(y_true, (2, 1, 0))
            y_pred = np.transpose(y_pred, (2, 1, 0))
            real_y_true = np.transpose(real_y_true, (2, 1, 0))
            real_y_pred = np.transpose(real_y_pred, (2, 1, 0))

            mae = mean_absolute_error(real_y_true.reshape(-1), real_y_pred.reshape(-1))
            rmse = mean_squared_error(real_y_true.reshape(-1), real_y_pred.reshape(-1)) ** 0.5
            # real_y_true.
            real_y_true_mask = (1 - (real_y_true < 15))
            mape1 = MAPE(real_y_true.reshape(-1), real_y_pred.reshape(-1), real_y_true_mask.reshape(-1))
            mape2 = masked_mape_np(real_y_true.reshape(-1), real_y_pred.reshape(-1), 0)

            print("MAE: {}, RMSE: {}, MAPE1: {}, MAPE2: {}".format(mae, rmse, mape1, mape2))
            self.logger.info("MAE: {}, RMSE: {}, MAPE1: {}, MAPE2: {}".format(mae, rmse, mape1, mape2))
        if "HZY" in args.save_dir:
            mae = rmse = mape1 = mape2 = 0

        pd.DataFrame(y_true.mean(axis=2)).to_csv(args.save_dir + r'\{}_true.csv'.format(args.data), mode='a+', header=False)  # (1, 307, 12)
        pd.DataFrame(y_pred.mean(axis=2)).to_csv(args.save_dir + r'\{}_pred.csv'.format(args.data), mode='a+', header=False)
        pd.DataFrame(real_y_true.mean(axis=2)).to_csv(args.save_dir + r'\{}_real_true.csv'.format(args.data), mode='a+', header=False)
        pd.DataFrame(real_y_pred.mean(axis=2)).to_csv(args.save_dir + r'\{}_real_pred.csv'.format(args.data), mode='a+', header=False)

        # if "HZY" in self.args.data:
        return np.array([rmse_ts_n, mae_ts_n, acc_ts_n, r2_ts_n, var_ts_n]).astype('float64'), np.array(
            [rmse_ts_r, mae_ts_r, acc_ts_r, r2_ts_r, var_ts_r]).astype('float64'), np.array(
            [mae, rmse, mape1, mape2]).astype('float64')

    def pure_test(self, model, args, data_loader, scaler, path=None, mode="test"):
        test_eta = 0.32     # define the graph's positive edges density

        if path:
            if os.path.exists(path):
                check_point = torch.load(path)
                state_dict = check_point['state_dict']
                model.load_state_dict(state_dict)
                model.to(args.device)
                print("Load saved model")

        start_time = time.time()

        # update and build graph
        model.eval()
        with torch.no_grad():
            for epoch in range(1, self.args.epochs + 1):
                torch.cuda.empty_cache()
                epoch_start = time.time()
                scores, epoch_losses, epoch_CL_losses, epoch_PU_losses, epoch_ploss = self.train_epoch(epoch,
                                                                                                       mode="regular")
                self.dynamic_delta = self.args.delta - (self.args.delta - 1 + self.args.eta) / \
                                     (self.args.epochs + 1) * epoch
                self.dynamic_delta_negative = self.args.delta_negative + (self.args.eta - self.args.delta_negative) / \
                                              (self.args.epochs + 1) * epoch
                update_nodes = self.update_graph(scores=scores)
                if update_nodes < 2:
                    break
                update_nodes_negative = self.update_graph_add_negative(scores=scores)
                self.check_node(scores, lambda_K=10)

                all_edges = self.graph.shape[0] ** 2
                p_edges = self.graph._nnz()
                if p_edges / all_edges > test_eta:
                    break
                n_edges = self.graph_negative._nnz()
                self.logger.info(
                    'DP:{}({:.2f}%), DN:{}, Rest edges:{}'.format(p_edges, 100 * p_edges / all_edges, n_edges,
                                                                  all_edges - p_edges - n_edges))

                # predict
                model.eval()
                y_pred = []
                y_true = []
                with torch.no_grad():
                    for batch_idx, (x, xx, label) in enumerate(data_loader):
                        torch.cuda.empty_cache()
                        x = x.reshape([-1, self.args.num_nodes, self.args.seq_len])
                        xx = xx.reshape([self.args.num_nodes, self.args.ini_seq_len])
                        label = label.reshape([-1, self.args.pre_len])
                        Y_pre = model([xx, x, self.graph])

                        # Save normed values
                        if args.real_value:
                            Y_pre = scaler.transform(Y_pre)
                        y_pred.append(Y_pre.cpu().numpy())
                        y_true.append(label.cpu().numpy())

                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                real_y_true = scaler.inverse_transform(y_true)
                real_y_pred = scaler.inverse_transform(y_pred)
                rmse_ts_n, mae_ts_n, acc_ts_n, r2_ts_n, var_ts_n = evaluation(y_true, y_pred, args, args.acc_threshold)
                rmse_ts_r, mae_ts_r, acc_ts_r, r2_ts_r, var_ts_r = evaluation(real_y_true, real_y_pred, args,
                                                                              args.acc_real_threshold)

                self.logger.info(
                    "Testing results {} {} {} {} {}".format(rmse_ts_n, mae_ts_n, acc_ts_n, r2_ts_n, var_ts_n))
                self.logger.info(
                    "Testing results(r) {} {} {} {} {}".format(rmse_ts_r, mae_ts_r, acc_ts_r, r2_ts_r, var_ts_r))

                y_true = np.transpose(y_true, (2, 1, 0))
                y_pred = np.transpose(y_pred, (2, 1, 0))
                real_y_true = np.transpose(real_y_true, (2, 1, 0))
                real_y_pred = np.transpose(real_y_pred, (2, 1, 0))

                mae = mean_absolute_error(real_y_true.reshape(-1), real_y_pred.reshape(-1))
                rmse = mean_squared_error(real_y_true.reshape(-1), real_y_pred.reshape(-1)) ** 0.5

                for test_threshold in [0.5, 1, 5, 10, 20, 30, 40, 60]:
                    real_y_true_mask = (1 - (real_y_true < test_threshold))
                    mape1 = MAPE(real_y_true.reshape(-1), real_y_pred.reshape(-1), real_y_true_mask.reshape(-1))
                    print("Threshould={}, MAPE={:.4f}".format(test_threshold, mape1))
                    self.logger.info("Threshould={}, MAPE={:.4f}".format(test_threshold, mape1))

                mape2 = masked_mape_np(real_y_true.reshape(-1), real_y_pred.reshape(-1), 0)

                print("RMSE: {}, MAE: {}, MAPE1: {}, MAPE2: {}".format(rmse, mae, mape1, mape2))
                self.logger.info("RMSE: {}, MAE: {}, MAPE1: {}, MAPE2: {}".format(rmse, mae, mape1, mape2))

        pd.DataFrame(y_true.mean(axis=2)).to_csv(args.save_dir + r'\{}_true.csv'.format(args.data), mode='a+',
                                                 header=False)  # (1, 307, 12)
        pd.DataFrame(y_pred.mean(axis=2)).to_csv(args.save_dir + r'\{}_pred.csv'.format(args.data), mode='a+',
                                                 header=False)
        pd.DataFrame(real_y_true.mean(axis=2)).to_csv(args.save_dir + r'\{}_real_true.csv'.format(args.data), mode='a+',
                                                      header=False)
        pd.DataFrame(real_y_pred.mean(axis=2)).to_csv(args.save_dir + r'\{}_real_pred.csv'.format(args.data), mode='a+',
                                                      header=False)

        return np.array([rmse_ts_n, mae_ts_n, acc_ts_n, r2_ts_n, var_ts_n]).astype('float64'), np.array(
            [rmse_ts_r, mae_ts_r, acc_ts_r, r2_ts_r, var_ts_r]).astype('float64'), np.array(
            [mae, rmse, mape1, mape2]).astype('float64')

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
