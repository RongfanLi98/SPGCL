import numpy as np
import torch
import torch.nn as nn


class SPGCL(torch.nn.Module):
    def __init__(self, args=None, sparse=True):
        super(SPGCL, self).__init__()
        num_in_features_R = args.seq_len
        num_in_features_X = args.ini_seq_len
        W_len = 24  # Feature projecting length
        E_len = 24  # Embedding length
        R_len = E_len//2
        self.linear_proj = nn.Linear(num_in_features_R, W_len, bias=True)
        self.linear_proj_abs = nn.Linear(num_in_features_X, W_len, bias=True)
        self.args = args

        # c_aggregate: aggregate neighbors into i to obtain context embedding
        self.c_aggregate = GAT(num_of_layers=2, num_heads_per_layer=[3, 3], num_features_per_layer=[E_len, E_len, E_len], layer_type="aggregate", num_relation=E_len)
        # predict: aggregate neighbors into i to make prediction
        self.predict = GAT(num_of_layers=2, num_heads_per_layer=[3, 3], num_features_per_layer=[E_len, E_len, E_len], layer_type="aggregate", num_relation=E_len)
        # predict_abs: base GAT
        self.predict_abs = GAT(num_of_layers=2, num_heads_per_layer=[3, 3], num_features_per_layer=[E_len, E_len, E_len], layer_type="base")

        # Linear mapping X to target length
        self.r_proj = nn.Sequential(
            nn.Linear(W_len, W_len, bias=True),
            nn.Linear(W_len, W_len, bias=True),
            nn.Linear(W_len, E_len, bias=True),
            nn.LeakyReLU(),
        )
        # Mapping prediction to pre_len
        self.Y_proj = nn.Sequential(
            nn.Linear(E_len, E_len, bias=True),
            nn.Linear(E_len, R_len, bias=True),
            nn.Linear(R_len, args.pre_len, bias=True)
        )

        self.Softmax = nn.Softmax()
        self.Tanh = nn.Tanh()

        # init
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.linear_proj_abs.weight)
        self.Y_proj.apply(self.init_weights)
        self.r_proj.apply(self.init_weights)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.data_type = torch.float64
        self.print_self()

    def forward(self, data, require_socre=False, direct=True, score_mask=True):
        # input the mask and features at step k: return Y: [N,N,W_len], [N,N,E_len]
        x, r_embeddings, score_matrix = self.get_embedding(data, score_mask=score_mask)
        if score_mask:
            eta = 1 - self.args.eta
            # Rescale to [0, 1]
            score_act = (score_matrix + 1) / 2
            score_mask = torch.where(score_act > eta, torch.ones_like(score_act), score_act)
            score_mask = torch.where(score_act <= eta, torch.zeros_like(score_mask), score_mask)
            score_mask = score_mask.to_sparse().coalesce().detach()
            # [N,F] [N,N,E]
        else:
            score_mask = score_matrix.to_sparse().detach()
        prediction, _, _ = self.predict([x, r_embeddings, score_mask])
        if direct:
            if score_mask._nnz() != 0:
                prediction_abs, _ = self.predict_abs([x, score_mask])
                prediction = (prediction_abs+prediction)/2
        _ = _.detach()

        # [N, H * E_len] -> [N, pre_len]: equal as Y = WX
        Y = self.Y_proj(prediction)
        if require_socre:
            return Y, score_matrix
        else:
            return Y

    def get_prediction(self, data):
        with torch.no_grad():
            # [N,N,relative_len] -> r_embeddings = [N, N, E_len]
            x, r_embeddings, score_matrix = self.get_embedding(data, loss=False)

        eta = 1 - self.args.eta
        # Rescale to [0, 1]
        score_act = (score_matrix + 1) / 2
        score_mask = torch.where(score_act > eta, torch.ones_like(score_act), score_act)
        score_mask = torch.where(score_act <= eta, torch.zeros_like(score_mask), score_mask)
        score_mask = score_mask.to_sparse().coalesce()
        prediction, _, _ = self.predict([x, r_embeddings, score_mask])
        # prediction_abs, _ = self.predict_abs([x, connectivity_mask])      # use absolute features in GAT
        # prediction = (prediction_abs+prediction)/2                        # use absolute features in GAT

        # [N, H * E_len] -> [N, pre_len]
        Y = self.Y_proj(prediction)
        return Y

    def get_embedding(self, data, score_mask=True, context=True):
        # xx is absolute features, and in_node_features is related features
        xx, in_nodes_features, connectivity_mask = data

        xx = xx.to(torch.float32)  # Comment if necessary
        # in_nodes_features = in_nodes_features.to(torch.float32)

        # Eq.13 W=[N, N, W_len] <- in_nodes_features=[N, N, seq_len]
        W = self.linear_proj(in_nodes_features)
        xx = self.linear_proj_abs(xx)
        # Eq.13 r_embeddings=[N, N, E_len] <- W=[N, N, W_len]. r_embeddings is embedding of relationship
        r_embeddings = self.r_proj(W)

        if context:
            # Eq. 14 compute context embedding=[N, E_len]
            c_embeddings, _, _ = self.c_aggregate((W, r_embeddings, connectivity_mask))
            # Eq.15. return loss，else return node embedding
            # score=[N, N] \in [-1, 1]. Calculate cosine similarity
            score_matrix = -nn.functional.pirwise_distance(c_embeddings.unsqueeze(dim=1), r_embeddings, p=2)
            score_matrix = (score_matrix - score_matrix.min())/(score_matrix.max()-score_matrix.min())
            score_matrix = score_matrix * 2 - 1     # to [-1, 1]
        else:
            # Or the scores can be calculated by cosine similarity
            score_matrix = nn.functional.cosine_similarity(r_embeddings, r_embeddings.transpose(0, 1), dim=2)
        if score_mask:
            return xx, r_embeddings, score_matrix
        else:
            return xx, r_embeddings, connectivity_mask.to_dense()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def print_self(self):
        total_param = 0
        for name, parameters in self.named_parameters():
            print(name, ':', parameters.size())
            total_param += np.prod(parameters.size())
        print('Net\'s total params:', total_param)


class GAT(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.
    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.
    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.
    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False, sparse=True, layer_type=None, r_proj=None, num_relation=0):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        if layer_type == "aggregate":
            base_layer = AggregateGATLayer
        else:
            base_layer = BaseGATLayer
        gat_layers = []
        for i in range(num_of_layers):
            layer = base_layer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights,
                num_relation=num_relation * num_heads_per_layer[i]
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, num_relation=0):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.randn(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.randn(1, num_of_heads, num_out_features))

        self.proj_param = nn.Parameter(torch.randn(num_of_heads, num_in_features, num_out_features))
        self.features_sum = nn.Parameter(torch.randn(num_out_features, 1))
        self.reset_relationship = nn.Parameter(torch.randn(num_of_heads, num_of_heads))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.randn(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.randn(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.sigmoid = nn.Sigmoid()
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(out_nodes_features.dim()-2)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT) if batch GAT, head_dim is 2
            self.head_dim = 2 if out_nodes_features.dim() == 4 else 1
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class BaseGATLayer(GATLayer):
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, num_relation=0):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights, num_relation)

    def forward(self, data):
        # edge_index is sparse tensor
        batch_features, edge_index = data
        out_nodes_features = self.aggregate((batch_features, edge_index.indices()))
        return (out_nodes_features, edge_index)

    def aggregate(self, data):
        # [N, FIN], [2, E]
        in_nodes_features, edge_index = data  # [N,N,F],[2,E]=>edge_index is dense matrix
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        # [N*N, num_of_heads, num_out_features]
        # nodes_features_proj: N*N,NH,F_out, it can be thought as h = W*h
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments

        # input features [N*N, NH, F_out] * [1,NH,F_out] =  [ (N*N, NH, 1) ] => [N*N, NH]
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        # [E, NH]
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source,
                                                                                           scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)

        # leakyReLU and softmax    [E,NH]
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT

        # aggreate the attention of edges:  [E,NH,F_OUT]
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        # make edges' attention into node featrues [N, NH, F_OUT]
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.
        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:
        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class AggregateGATLayer(BaseGATLayer):
    """
    Require Relationship Matrix, we use this matrix to aggregate features: here we only need to compute Hadmard product
    then activate it and use mask, compute the mean value of multi-heads
    """

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, num_relation=0):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights, num_relation)

        # map r
        self.r_linear_proj = nn.Linear(num_relation, num_of_heads * num_out_features, bias=False)
        self.neighbors_mean = True

    def forward(self, data):
        # [N, num_in_features], [N, N, num_relation], [N, N]
        in_nodes_features, relationships, connectivity_mask = data
        assert in_nodes_features.shape[0] == relationships.shape[0]

        # [N, num_in_features]-> [N, H*E_len], [N, N, num_relation] -> [N, N, H*E_len]
        in_nodes_features = self.linear_proj(in_nodes_features)
        relationships = self.r_linear_proj(relationships)
        in_nodes_features = self.dropout(in_nodes_features)  # in the official GAT imp they did dropout here as well

        # R mul X，[N, N, H*E_len] mul [N, H*E_len] = [N, N, H*E_len]
        aggregate = torch.mul(relationships, in_nodes_features)
        # mask=[N,N]
        dense_mask = connectivity_mask.to_dense()
        aggregate = torch.mul(aggregate, dense_mask.unsqueeze(-1))
        # [N, N, H*E_len] -> [N, H*E_len]
        aggregate = aggregate.sum(dim=1)
        if self.neighbors_mean:
            c_embeddings = aggregate.sum(dim=1) / dense_mask.sum(dim=1)

        # [N, H, E_len]
        out_nodes_features = aggregate.view(-1, self.num_of_heads, self.num_out_features)

        # when last layer is True: get mean
        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT) if batch GAT, head_dim is 2
            self.head_dim = 2 if out_nodes_features.dim() == 4 else 1
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        return out_nodes_features.type(torch.float32), relationships, connectivity_mask
