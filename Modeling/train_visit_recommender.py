
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


def train_visit_recommender(model, dataset, optimizer, num_epochs=10, batch_size=1):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for data in loader:
            data = data.to(next(model.parameters()).device)
            optimizer.zero_grad()

            out = model(data.x_dict, data.edge_index_dict)  # [num_visit_area]
            label = data['visit_area'].y.to(out.device)     # [num_visit_area]

            loss = F.binary_cross_entropy_with_logits(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
