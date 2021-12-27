from Graph_similarity.Graph_Matching_Networks.evaluation import compute_similarity, auc
from Graph_similarity.Graph_Matching_Networks.loss import pairwise_loss, triplet_loss
from Graphh.Graphh import Graphh
from Parser.Make_graph import make_graph_simplex_direct
from Graph_similarity.Graph_Matching_Networks.utils import *
from Graph_similarity.Graph_Matching_Networks.configure import *
from utils import split_training_testset
import numpy as np
import torch.nn as nn
import collections
import time
import os
import pandas as pd
import pathlib
from Graph_similarity.Graph_Matching_Networks.my_utils import get_next_batch


def main():
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # Print configure
    config = get_default_config()
    for (k, v) in config.items():
        print("%s= %s" % (k, v))
    # Set random seeds
    seed = config['seed']
    random.seed(0)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Settings
    perc_train_test = 0.7
    num_retrieval_accepted = 10
    base_path = str(pathlib.Path(__file__).parent).replace("\\", "/")
    path_dataset = base_path + "/Datasets/DS_4/Models/"
    excel_path = base_path + "/Datasets/DS_4/results/wasserstein/ww_parts_score.xlsx"
    graph_saves_path = base_path + "/Graphh/graph_save/simplex_direct/"
    models_name = "models_26_12"
    save_model_path = base_path + "/Datasets/DS_4/results/gmn/" + models_name
    train = False
    load = True

    file_names = []
    for file in os.listdir(path_dataset):
        if file.endswith(".stp") or file.endswith(".step"):
            file_names.append(file)
    names = [os.path.splitext(f)[0] for f in file_names]
    print("Realizing graphs")
    list_graphs = [make_graph_simplex_direct(f, graph_saves_path, dataset_path=path_dataset) for f in file_names]
    graphs_direct = []
    for id_name, name in enumerate(names):
        g = Graphh(list_graphs[id_name], name)
        graphs_direct.append(g)
    for g in graphs_direct:
        g.print_composition()

    list_parts = []
    all_set = []
    df = pd.read_excel(excel_path)
    for i, gh_i in enumerate(graphs_direct):
        for j, part in enumerate(gh_i.parts):
            list_parts.append(part)
    for i, part_i in enumerate(list_parts):
        row = []
        for j, part_j in enumerate(list_parts):
            dist = df.iloc[i + 1, j + 1]
            if dist > 1:  # todo fix this
                dist = 1
            row.append([part_i, part_j, dist])
        min_n_elements = sorted(row, key=lambda t: t[2])
        min_n_elements = min_n_elements[:num_retrieval_accepted]
        count = 0
        for e in row:
            if e in min_n_elements:
                all_set.append([e[0], e[1], 1])
            else:
                if count < num_retrieval_accepted:
                    count += 1
                    all_set.append([e[0], e[1], -1])

    random.shuffle(all_set)
    training_set, test_set = split_training_testset(all_set, perc_train_test)
    batch_size = config['training']['batch_size']
    print("Traing set dim: " + str(len(training_set)))

    global_labels = set()
    for graphh in list_parts:
        labels = graphh.get_labels()
        global_labels = global_labels.union(set(labels))
    global_labels = sorted(global_labels)
    global_labels = {val: index for index, val in enumerate(global_labels)}

    if config['training']['mode'] == 'pair':
        training_data_iter = get_next_batch(batch_size=batch_size, training_set=training_set,
                                            one_hot_dict=global_labels)
        first_batch_graphs, _ = next(training_data_iter)
    else:
        training_data_iter = get_next_batch(batch_size=batch_size, training_set=training_set,
                                            one_hot_dict=global_labels)
        first_batch_graphs = next(training_data_iter)

    node_feature_dim = first_batch_graphs.node_features.shape[-1]
    edge_feature_dim = first_batch_graphs.edge_features.shape[-1]

    model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
    if load:
        model.load_state_dict(torch.load(save_model_path))
    model.to(device)

    accumulated_metrics = collections.defaultdict(list)

    training_n_graphs_in_batch = config['training']['batch_size']
    if config['training']['mode'] == 'pair':
        training_n_graphs_in_batch *= 2
    elif config['training']['mode'] == 'triplet':
        training_n_graphs_in_batch *= 4
    else:
        raise ValueError('Unknown training mode: %s' % config['training']['mode'])

    if train:
        t_start = time.time()
        for i_iter in range(config['training']['n_training_steps']):
            model.train(mode=True)

            batch = next(training_data_iter)
            if config['training']['mode'] == 'pair':
                node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                labels = labels.to(device)
            else:
                node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch)
            graph_vectors = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                  to_idx.to(device),
                                  graph_idx.to(device), training_n_graphs_in_batch)

            if config['training']['mode'] == 'pair':
                x, y = reshape_and_split_tensor(graph_vectors, 2)
                loss = pairwise_loss(x, y, labels,
                                     loss_type=config['training']['loss'],
                                     margin=config['training']['margin'])

                is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
                is_neg = 1 - is_pos
                n_pos = torch.sum(is_pos)
                n_neg = torch.sum(is_neg)
                sim = compute_similarity(config, x, y)
                sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
                sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)
            else:
                x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
                loss = triplet_loss(x_1, y, x_2, z,
                                    loss_type=config['training']['loss'],
                                    margin=config['training']['margin'])

                sim_pos = torch.mean(compute_similarity(config, x_1, y))
                sim_neg = torch.mean(compute_similarity(config, x_2, z))

            graph_vec_scale = torch.mean(graph_vectors ** 2)
            if config['training']['graph_vec_regularizer_weight'] > 0:
                loss += (config['training']['graph_vec_regularizer_weight'] * 0.5 * graph_vec_scale)

            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))  #
            nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
            optimizer.step()

            sim_diff = sim_pos - sim_neg
            accumulated_metrics['loss'].append(loss)
            accumulated_metrics['sim_pos'].append(sim_pos)
            accumulated_metrics['sim_neg'].append(sim_neg)
            accumulated_metrics['sim_diff'].append(sim_diff)

            # save
            if (i_iter + 1) % config['training']['save_after'] == 0:
                print("Save at iteration: " + str(i_iter + 1))
                torch.save(model.state_dict(), save_model_path)

            # evaluation
            if (i_iter + 1) % config['training']['print_after'] == 0:
                metrics_to_print = {k: torch.mean(v[0]) for k, v in accumulated_metrics.items()}
                info_str = ', '.join(['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])
                # reset the metrics
                accumulated_metrics = collections.defaultdict(list)
                print('iter %d, %s, time %.2fs' % (i_iter + 1, info_str, time.time() - t_start))
                t_start = time.time()

    # evaluation
    model.eval()
    import math
    test_data_iter = get_next_batch(batch_size=batch_size, training_set=test_set, one_hot_dict=global_labels)
    with torch.no_grad():
        accumulated_pair_auc = []
        for _ in range(math.ceil(len(test_set) / batch_size)):
            batch = next(test_data_iter)
            if config['training']['mode'] == 'pair':
                node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                labels = labels.to(device)
                graph_vectors = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                      to_idx.to(device),
                                      graph_idx.to(device), training_n_graphs_in_batch)

                x, y = reshape_and_split_tensor(graph_vectors, 2)
                sim = compute_similarity(config, x, y)
                pair_auc = auc(sim, labels)
                accumulated_pair_auc.append(pair_auc)

        eval_metrics = {'pair_auc': np.mean(accumulated_pair_auc)}
        info_str = ', ' + ', '.join(['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])
        print(info_str)


if __name__ == "__main__":
    main()
