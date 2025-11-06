##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging

from kairos_utils import *
from config import *
from model import *
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

# Setting for logging
logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ARTIFACT_DIR + 'reconstruction.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@torch.no_grad()
def test(inference_data,
          memory,
          gnn,
          link_pred,
          neighbor_loader,
          nodeid2msg,
          path
          ):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    memory.eval()
    gnn.eval()
    link_pred.eval()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    time_with_loss = {}  # key: time，  value： the losses
    total_loss = 0.0
    edge_list = []

    window_unique_nodes = torch.empty(0, dtype=torch.long, device=device)
    window_edge_count = 0


    start_time = int(inference_data.t[0].item())
    event_count = 0
    pos_o = []

    # Record the running time to evaluate the performance
    start = time.perf_counter()

    loader = TemporalDataLoader(
        inference_data,
        batch_size=BATCH,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    for batch in tqdm(loader, desc="testing windows", leave=False, mininterval=0.5):
        src = batch.src.to(device=device, non_blocking=True)
        pos_dst = batch.dst.to(device=device, non_blocking=True)
        t_cpu = batch.t
        t = t_cpu.to(device=device, non_blocking=True)
        msg_cpu = batch.msg
        msg = msg_cpu.to(device=device, non_blocking=True)
        window_unique_nodes = torch.cat([window_unique_nodes, src, pos_dst]).unique()
        window_edge_count += src.size(0)

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        e_id_cpu = e_id.cpu()
        edge_t = inference_data.t[e_id_cpu].to(device=device, non_blocking=True)
        edge_msg = inference_data.msg[e_id_cpu].to(device=device, non_blocking=True)
        z = gnn(z, last_update, edge_index, edge_t, edge_msg)

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg_cpu:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # compute the loss for each edge
        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

        for i in range(len(pos_out)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t_cpu[i].item())
            edgeindex = tensor_find(msg_cpu[i][node_embedding_dim:-node_embedding_dim], 1)
            edge_type = rel2id[edgeindex]
            edge_loss_value = each_edge_loss[i]

            temp_dic = {}
            temp_dic['loss'] = float(edge_loss_value)
            temp_dic['srcnode'] = srcnode
            temp_dic['dstnode'] = dstnode
            temp_dic['srcmsg'] = srcmsg
            temp_dic['dstmsg'] = dstmsg
            temp_dic['edge_type'] = edge_type
            temp_dic['time'] = t_var

            edge_list.append(temp_dic)

        event_count += len(batch.src)
        if int(t_cpu[-1].item()) > start_time + time_window_size:
            # Here is a checkpoint, which records all edge losses in the current time window
            time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(int(t_cpu[-1].item()))

            end = time.perf_counter()
            avg_window_loss = total_loss / event_count if event_count else 0.0
            edge_loss_sum = sum(e['loss'] for e in edge_list)
            avg_edge_loss = edge_loss_sum / len(edge_list) if edge_list else 0.0
            nodes_count = int(window_unique_nodes.numel())
            time_with_loss[time_interval] = {
                'loss': avg_window_loss,
                'nodes_count': nodes_count,
                'total_edges': window_edge_count,
                'costed_time': (end - start)
            }

            log = open(path + "/" + time_interval + ".txt", 'w')

            logger.info(
                f'Time: {time_interval}, Loss: {avg_edge_loss:.4f}, Nodes_count: {nodes_count}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)  # Rank the results based on edge losses
            for e in edge_list:
                log.write(str(e))
                log.write("\n")
            event_count = 0
            total_loss = 0.0
            window_edge_count = 0
            window_unique_nodes = torch.empty(0, dtype=torch.long, device=device)
            start_time = int(t_cpu[-1].item())
            start = time.perf_counter()
            log.close()
            edge_list.clear()

    return time_with_loss

def load_data():
    # graph_4_3 - graph_4_5 will be used to initialize node IDF scores.
    graph_4_3 = torch.load(GRAPHS_DIR + "graph_4_3.TemporalData.simple")
    graph_4_4 = torch.load(GRAPHS_DIR + "graph_4_4.TemporalData.simple")
    graph_4_5 = torch.load(GRAPHS_DIR + "graph_4_5.TemporalData.simple")

    # Testing set
    graph_4_6 = torch.load(GRAPHS_DIR + "graph_4_6.TemporalData.simple")
    graph_4_7 = torch.load(GRAPHS_DIR + "graph_4_7.TemporalData.simple")

    return [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]
    return graph_4_6


if __name__ == "__main__":
    print("[Test] Starting reconstruction runs...")
    logger.info("Start logging.")

    # load the map between nodeID and node labels
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Load data
    graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7 = load_data()
    # graph_4_6 = load_data()

    max_nodes = 0
    for g in [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]:
        candidate = getattr(g, "num_nodes", None)
        if candidate:
            max_nodes = max(max_nodes, int(candidate))
        else:
            max_nodes = max(max_nodes, int(torch.cat([g.src, g.dst]).max().item()) + 1)
    configure_node_capacity(max_nodes)

    # load trained model
    memory, gnn, link_pred, _neighbor_loader = torch.load(f"{MODELS_DIR}models.pt", map_location=device)
    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    evaluation_targets = [
        ("graph_4_3", graph_4_3),
        ("graph_4_4", graph_4_4),
        ("graph_4_5", graph_4_5),
        ("graph_4_6", graph_4_6),
        ("graph_4_7", graph_4_7),
    ]

    for label, dataset in tqdm(evaluation_targets, desc="reconstructing days", leave=False):
        test(
            inference_data=dataset,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg,
            path=ARTIFACT_DIR + label,
        )
        print(f"[Test] Completed reconstruction for {label}.")

    print("[Test] All reconstruction runs finished.")
