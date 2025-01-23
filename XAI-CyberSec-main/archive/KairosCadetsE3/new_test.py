##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging
import os
import torch
import time
from kairos_utils import *
from config import *
from model import *
from tqdm import tqdm
from new_train import assoc

# Setting for logging
logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ARTIFACT_DIR + 'reconstruction.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@torch.no_grad()
def test(inference_data, memory, gnn, link_pred, neighbor_loader, nodeid2msg, path):
    logger.info("I am in test: ")
    try:
        if not os.path.exists(path):
            os.makedirs(path)

        memory.eval()
        gnn.eval()
        link_pred.eval()

        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        time_with_loss = {}  # key: time, value: the losses
        total_loss = 0
        edge_list = []

        unique_nodes = torch.tensor([]).to(device=device)
        total_edges = 0

        start_time = inference_data.t[0]
        event_count = 0
        pos_o = []

        # Record the running time to evaluate the performance
        start = time.perf_counter()

        for i in range(0, len(inference_data), 1024):
            try:
                logger.info("I am in for")
                batch = inference_data[i:i + 1024]

                src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
                total_edges += BATCH

                n_id = torch.cat([src, pos_dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                z, last_update = memory(n_id)
                z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

                pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
                logger.info("I am at pos_out")
                pos_o.append(pos_out)
                y_pred = torch.cat([pos_out], dim=0)
                y_true = []
                for m in msg:
                    l = tensor_find(m[NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1) - 1
                    y_true.append(l)
                y_true = torch.tensor(y_true).to(device=device)
                y_true = y_true.reshape(-1).to(torch.long).to(device=device)

                loss = criterion(y_pred, y_true)
                total_loss += float(loss) * batch.num_events

                # Update the edges in the batch to the memory and neighbor_loader
                memory.update_state(src, pos_dst, t, msg)
                neighbor_loader.insert(src, pos_dst)

                # Compute the loss for each edge
                each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)
                for i in range(len(pos_out)):
                    srcnode = int(src[i])
                    dstnode = int(pos_dst[i])

                    # Check if the node IDs are present in the nodeid2msg mapping
                    if srcnode in nodeid2msg and dstnode in nodeid2msg:
                        srcmsg = str(nodeid2msg[srcnode])
                        dstmsg = str(nodeid2msg[dstnode])
                    else:
                        logger.warning(f"Node ID {srcnode} or {dstnode} not found in nodeid2msg. Skipping.")
                        continue
                    t_var = int(t[i])
                    edgeindex = tensor_find(msg[i][NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1)
                    edge_type = REL2ID[edgeindex]
                    loss = each_edge_loss[i]

                    temp_dic = {
                        'loss': float(loss),
                        'srcnode': srcnode,
                        'dstnode': dstnode,
                        'srcmsg': srcmsg,
                        'dstmsg': dstmsg,
                        'edge_type': edge_type,
                        'time': t_var
                    }

                    edge_list.append(temp_dic)

                event_count += len(batch.src)
                logger.info(f"t[-1]: {t[-1]}, start_time: {start_time}, TIME_WINDOW_SIZE: {TIME_WINDOW_SIZE}")
                logger.info(f"start_time + TIME_WINDOW_SIZE: {start_time + TIME_WINDOW_SIZE}")

                if t[-1] > start_time + TIME_WINDOW_SIZE:
                    logger.info("Yes  it is")
                    # Here is a checkpoint, which records all edge losses in the current time window
                    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])

                    end = time.perf_counter()
                    time_with_loss[time_interval] = {
                        'loss': loss,
                        'nodes_count': len(unique_nodes),
                        'total_edges': total_edges,
                        'costed_time': (end - start)
                    }

                    # Replace ':' and '~' with valid characters
                    time_interval = time_interval.replace(":", "_").replace("~", "_")
                    logger.info(time_interval)
                    log_path = os.path.join(path, f"{time_interval}.txt")
                    logger.info(log_path)
                    with open(log_path, 'w') as log:
                        logger.info(" I am in txt")
                        for e in edge_list:
                            loss += e['loss']

                        loss = loss / event_count
                        logger.info(
                            f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
                        edge_list = sorted(edge_list, key=lambda x: x['loss'],
                                           reverse=True)  # Rank the results based on edge losses
                        for e in edge_list:
                            log.write(str(e))
                            log.write("\n")
                    event_count = 0
                    total_loss = 0
                    start_time = t[-1]
                    edge_list.clear()
            except Exception as e:
                logger.info(f"Error during batch processing: {e}")

        return time_with_loss

    except Exception as e:
        logger.info(f"Error during testing: {e}")
        raise


def load_data():
    try:
        # graph_4_3 - graph_4_5 will be used to initialize node IDF scores.
        graph_4_3 = torch.load(GRAPHS_DIR + "graph_4_3.TemporalData.simple").to(device=device)
        graph_4_4 = torch.load(GRAPHS_DIR + "graph_4_4.TemporalData.simple").to(device=device)
        graph_4_5 = torch.load(GRAPHS_DIR + "graph_4_5.TemporalData.simple").to(device=device)

        # Testing set
        graph_4_6 = torch.load(GRAPHS_DIR + "graph_4_6.TemporalData.simple").to(device=device)
        graph_4_7 = torch.load(GRAPHS_DIR + "graph_4_7.TemporalData.simple").to(device=device)

        return [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    logger.info("Start logging.")

    try:
        # Load the map between nodeID and node labels
        cur, _ = init_database_connection()
        nodeid2msg = gen_nodeid2msg(cur=cur)

        # Load data
        graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7 = load_data()
        logger.info("loaded data")
        # Load trained model
        try:
            memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}models.pt", map_location=device,
                                                                 weights_only=False)
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            raise

        # Reconstruct the edges in each day
        for graph, name in zip([graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7],
                               ["graph_4_3", "graph_4_4", "graph_4_5", "graph_4_6", "graph_4_7"]):
            try:
                test(
                    inference_data=graph,
                    memory=memory,
                    gnn=gnn,
                    link_pred=link_pred,
                    neighbor_loader=neighbor_loader,
                    nodeid2msg=nodeid2msg,
                    path=ARTIFACT_DIR + f"{name}/"
                )
            except Exception as e:
                logger.error(f"Error during testing for {name}: {e}")
    except Exception as e:
        logger.error(f"Error in main testing loop: {e}")
