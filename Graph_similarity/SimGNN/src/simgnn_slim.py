from .simgnn import SimGNNTrainer, SimGNN
from torch_geometric.nn import GCNConv
from .layers import AttentionModule, TenorNetworkModule
import torch


class SimGNN_slim256(SimGNN):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super().__init__(args, number_of_labels)

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_3 = GCNConv(self.number_labels, self.args.filters_3)
        self.convolution_4 = GCNConv(self.args.filters_3, self.args.filters_4)
        self.convolution_5 = GCNConv(self.args.filters_4, self.args.filters_5)
        self.convolution_6 = GCNConv(self.args.filters_5, self.args.filters_6)
        self.attention = AttentionModule_slim256(self.args)
        self.tensor_network = TenorNetworkModule_slim256(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)


    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_3(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_4(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_5(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_6(features, edge_index)
        return features


class SimGNNTrainer_slim256(SimGNNTrainer):
    def __init__(self, args, training_set, test_set, labels_path):
        super().__init__(args, training_set, test_set, labels_path)

    def setup_model(self):
        self.model = SimGNN_slim256(self.args, self.number_of_labels)


class AttentionModule_slim256(AttentionModule):
    def __init__(self, args):
        super().__init__(args)

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_6,
                                                             self.args.filters_6))


class TenorNetworkModule_slim256(TenorNetworkModule):

    def __init__(self, args):
        super().__init__(args)

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_6,
                                                             self.args.filters_6,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.args.filters_6))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.filters_6, -1))
        scoring = scoring.view(self.args.filters_6, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores