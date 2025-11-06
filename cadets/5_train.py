##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import copy
import os
import logging
from tqdm.auto import tqdm

from kairos_utils import *
from config import *
from model import *
from torch.optim import AdamW
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData

def concat_temporal(graphs):
    if not graphs:
        raise ValueError("Expected at least one TemporalData object.")

    # Concatenate fields and sort by timestamp so the loader sees a single continuous stream.
    src = torch.cat([g.src for g in graphs], dim=0)
    dst = torch.cat([g.dst for g in graphs], dim=0)
    t = torch.cat([g.t for g in graphs], dim=0)
    msg = torch.cat([g.msg for g in graphs], dim=0)

    # Sort by time
    order = torch.argsort(t)
    merged = TemporalData(
        src=src[order],
        dst=dst[order],
        t=t[order],
        msg=msg[order],
    )

    # Preserve any additional attributes set on the first graph.
    canonical_keys = {"src", "dst", "t", "msg"}
    for key in graphs[0].keys():
        if key in canonical_keys:
            continue
        try:
            value = getattr(graphs[0], key)
        except AttributeError:
            continue
        try:
            merged[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        except Exception:
            merged[key] = value

    return merged

# Setting for logging
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ARTIFACT_DIR + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train(train_data,
          memory,
          gnn,
          link_pred,
          optimizer,
          neighbor_loader,
          epoch_idx=None
          ):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    loader = TemporalDataLoader(
        train_data,
        batch_size=BATCH,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    for batch in tqdm(loader, desc=f"epoch {epoch_idx} • batches" if epoch_idx else "batches", leave=False, mininterval=0.5):
        optimizer.zero_grad()

        src = batch.src.to(device=device, non_blocking=True)
        pos_dst = batch.dst.to(device=device, non_blocking=True)
        t = batch.t.to(device=device, non_blocking=True)
        msg_cpu = batch.msg  # needed for tensor_find
        msg = msg_cpu.to(device=device, non_blocking=True)

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        e_id_cpu = e_id.cpu()
        edge_t = train_data.t[e_id_cpu].to(device=device, non_blocking=True)
        edge_msg = train_data.msg[e_id_cpu].to(device=device, non_blocking=True)
        z = gnn(z, last_update, edge_index, edge_t, edge_msg)
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg_cpu:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events

def load_train_data():
    g2 = torch.load(GRAPHS_DIR + "/graph_4_2.TemporalData.simple")
    g3 = torch.load(GRAPHS_DIR + "/graph_4_3.TemporalData.simple")
    g4 = torch.load(GRAPHS_DIR + "/graph_4_4.TemporalData.simple")
    merged = concat_temporal([g2, g3, g4])
    return merged

def init_models(node_feat_size):
    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        node_state_dim,
        time_dim,
        message_module=IdentityMessage(node_feat_size, node_state_dim, time_dim),
        aggregator_module=MPSSafeLastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=node_state_dim,
        out_channels=edge_dim,
        msg_dim=node_feat_size,
        time_enc=memory.time_enc,
    ).to(device)

    out_channels = len(include_edge_type)
    link_pred = LinkPredictor(in_channels=edge_dim, out_channels=out_channels).to(device)
    optimizer = AdamW(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()),
        lr=lr, eps=eps, weight_decay=weight_decay
    )

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    return memory, gnn, link_pred, optimizer, neighbor_loader

if __name__ == "__main__":
    print("[Train] Starting training run...")
    logger.info("Start logging.")

    # Echo device info at runtime, in addition to model import print
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"[Runtime] CUDA device: {gpu_name} | {total_mem_gb:.1f} GB")
        elif torch.backends.mps.is_available():
            print("[Runtime] MPS device active")
        else:
            print("[Runtime] CPU only")
    except Exception as _e:
        print(f"[Runtime] Device probe error: {_e}")

    # Load merged data for training (single continuous dataset)
    train_data = load_train_data()
    if getattr(train_data, "num_nodes", None):
        configure_node_capacity(int(train_data.num_nodes))
    else:
        max_id = int(torch.cat([train_data.src, train_data.dst]).max().item()) + 1
        configure_node_capacity(max_id)

    # Initialize the models and the optimizer
    node_feat_size = train_data.msg.size(-1)
    memory, gnn, link_pred, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)

    # Train: one pass over the merged dataset per epoch
    for epoch in tqdm(range(1, epoch_num + 1), desc="epochs"):
        loss = train(
            train_data=train_data,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            optimizer=optimizer,
            neighbor_loader=neighbor_loader,
            epoch_idx=epoch
        )
        logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # Save the trained model
    model = [memory, gnn, link_pred, neighbor_loader]

    os.system(f"mkdir -p {MODELS_DIR}")
    model_path = os.path.join(MODELS_DIR, "models.pt")
    torch.save(model, model_path)
    print(f"[Train] Training complete; model saved to {model_path}")
