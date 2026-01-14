#!/usr/bin/env python3
"""
Smart Jal - Graph Neural Network Model
Implements GNN for spatial interpolation of groundwater levels.

Based on research from:
- "Spatial-temporal GNNs for groundwater" (Nature Scientific Reports, 2024)
- PE-GNN (Positional Encoder GNN) for geographic data
- NN-GLS for geospatial data

Key Innovation:
- Models villages + piezometers as graph nodes
- Edges connect hydrogeologically similar locations (same aquifer)
- GNN learns to propagate water level information along valid paths
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.utils import add_self_loops
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/PyTorch Geometric not available. GNN will use fallback.")


class GroundwaterGNN(nn.Module):
    """
    Graph Neural Network for groundwater level prediction.

    Architecture:
    - Input: Node features (aquifer type, rainfall, elevation, etc.)
    - Graph Attention layers to learn spatial relationships
    - Output: Predicted water level per node
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 heads: int = 4):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Graph attention layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer: hidden -> hidden with attention
                self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                         heads=heads, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                         heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Output layers
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, x, edge_index, edge_attr=None):
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Graph convolutions with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection

        # Output prediction
        prediction = self.output_mlp(x)
        uncertainty = self.uncertainty_head(x)

        return prediction, uncertainty


class SpatialGraphBuilder:
    """
    Builds graph structure from villages and piezometers.

    Graph Construction:
    - Nodes: All villages (939) + piezometers (138)
    - Edges: Connect nodes if:
      1. Same aquifer type
      2. Within distance threshold
      3. Weighted by inverse distance
    """

    def __init__(self,
                 distance_threshold_km: float = 20.0,
                 same_aquifer_bonus: float = 2.0):
        self.distance_threshold = distance_threshold_km
        self.same_aquifer_bonus = same_aquifer_bonus
        self.scaler = StandardScaler()
        self.feature_cols = []

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in km between two points."""
        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def build_graph(self,
                    villages: gpd.GeoDataFrame,
                    piezometers: gpd.GeoDataFrame,
                    water_levels: Optional[pd.DataFrame] = None,
                    target_date: Optional[pd.Timestamp] = None) -> Dict:
        """
        Build graph data structure for GNN.

        Returns:
            Dict with 'data' (PyTorch Geometric Data object),
            'village_mask', 'piezo_mask', 'train_mask'
        """
        print("Building spatial graph for GNN...")

        # Get coordinates
        village_coords = np.array([
            [g.centroid.y, g.centroid.x] for g in villages.geometry
        ])
        piezo_coords = np.array([
            [g.centroid.y, g.centroid.x] for g in piezometers.geometry
        ])

        # Combine all nodes
        all_coords = np.vstack([village_coords, piezo_coords])
        n_villages = len(villages)
        n_piezos = len(piezometers)
        n_total = n_villages + n_piezos

        print(f"  Nodes: {n_villages} villages + {n_piezos} piezometers = {n_total}")

        # Get aquifer types
        village_aquifers = villages['geo_class'].fillna('Unknown').values
        piezo_aquifers = piezometers['geo_class'].fillna('Unknown').values
        all_aquifers = np.concatenate([village_aquifers, piezo_aquifers])

        # Build edges based on distance and aquifer
        edge_list = []
        edge_weights = []

        for i in range(n_total):
            for j in range(i + 1, n_total):
                dist = self._haversine_distance(
                    all_coords[i, 0], all_coords[i, 1],
                    all_coords[j, 0], all_coords[j, 1]
                )

                # Check if within threshold
                if dist <= self.distance_threshold:
                    # Calculate edge weight
                    weight = 1.0 / (dist + 0.1)  # Inverse distance

                    # Bonus for same aquifer
                    if all_aquifers[i] == all_aquifers[j]:
                        weight *= self.same_aquifer_bonus

                    # Add bidirectional edges
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    edge_weights.append(weight)
                    edge_weights.append(weight)

        print(f"  Edges: {len(edge_list)} (within {self.distance_threshold}km)")

        # Build node features
        node_features = self._build_node_features(villages, piezometers)

        # Get target values (water levels for piezometers)
        targets = np.full(n_total, np.nan)
        train_mask = np.zeros(n_total, dtype=bool)

        if water_levels is not None and target_date is not None:
            id_col = 'piezo_id' if 'piezo_id' in water_levels.columns else 'sno'
            target_levels = water_levels[
                (water_levels['date'].dt.year == target_date.year) &
                (water_levels['date'].dt.month == target_date.month)
            ]

            piezo_id_col = 'piezo_id' if 'piezo_id' in piezometers.columns else 'sno'
            for idx, row in piezometers.iterrows():
                piezo_id = row[piezo_id_col]
                level = target_levels[target_levels[id_col] == piezo_id]['water_level'].values
                if len(level) > 0 and not np.isnan(level[0]):
                    node_idx = n_villages + list(piezometers.index).index(idx)
                    targets[node_idx] = level[0]
                    train_mask[node_idx] = True

        print(f"  Training nodes (with water level): {train_mask.sum()}")

        # Create PyTorch tensors
        if TORCH_AVAILABLE:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(targets, dtype=torch.float).unsqueeze(1)
            train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                train_mask=train_mask_tensor
            )
        else:
            data = {
                'x': node_features,
                'edge_index': np.array(edge_list).T,
                'edge_attr': np.array(edge_weights),
                'y': targets,
                'train_mask': train_mask
            }

        return {
            'data': data,
            'n_villages': n_villages,
            'n_piezos': n_piezos,
            'feature_cols': self.feature_cols,
            'village_indices': list(range(n_villages)),
            'piezo_indices': list(range(n_villages, n_total))
        }

    def _build_node_features(self,
                             villages: gpd.GeoDataFrame,
                             piezometers: gpd.GeoDataFrame) -> np.ndarray:
        """Build feature matrix for all nodes."""

        # Common features
        feature_cols = [
            'centroid_lat', 'centroid_lon', 'area_km2',
            'elevation_mean', 'slope_mean',
            'rainfall_current', 'rainfall_cumulative_3m',
            'infiltration_score', 'runoff_score',
            'n_wells', 'well_density'
        ]

        # Get available features
        available_cols = [c for c in feature_cols if c in villages.columns]
        self.feature_cols = available_cols

        # Village features
        village_features = villages[available_cols].copy()
        for col in village_features.columns:
            village_features[col] = pd.to_numeric(village_features[col], errors='coerce')
            village_features[col] = village_features[col].fillna(village_features[col].median())

        # Piezometer features (create from piezometer data)
        piezo_features = pd.DataFrame(index=piezometers.index)
        for col in available_cols:
            if col == 'centroid_lat':
                piezo_features[col] = [g.centroid.y for g in piezometers.geometry]
            elif col == 'centroid_lon':
                piezo_features[col] = [g.centroid.x for g in piezometers.geometry]
            elif col in piezometers.columns:
                piezo_features[col] = piezometers[col]
            else:
                # Use village median as default
                piezo_features[col] = village_features[col].median()

        for col in piezo_features.columns:
            piezo_features[col] = pd.to_numeric(piezo_features[col], errors='coerce')
            piezo_features[col] = piezo_features[col].fillna(piezo_features[col].median())

        # Combine and scale
        all_features = pd.concat([village_features, piezo_features], ignore_index=True)
        all_features = all_features.fillna(0)

        # Add aquifer one-hot encoding
        village_aquifers = villages['geo_class'].fillna('Unknown')
        piezo_aquifers = piezometers['geo_class'].fillna('Unknown')
        all_aquifers = pd.concat([village_aquifers, piezo_aquifers], ignore_index=True)
        aquifer_dummies = pd.get_dummies(all_aquifers, prefix='aquifer')

        all_features = pd.concat([all_features, aquifer_dummies], axis=1)

        # Scale
        scaled = self.scaler.fit_transform(all_features.values)

        print(f"  Node features: {scaled.shape[1]} dimensions")

        return scaled


class GNNPredictor:
    """
    Complete GNN prediction system for groundwater levels.
    """

    def __init__(self,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 learning_rate: float = 0.01,
                 epochs: int = 200):
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.graph_builder = SpatialGraphBuilder()
        self.model = None
        self.graph_data = None

    def fit(self,
            villages: gpd.GeoDataFrame,
            piezometers: gpd.GeoDataFrame,
            water_levels: pd.DataFrame,
            target_date: pd.Timestamp) -> Dict:
        """
        Fit GNN model.
        """
        print("=" * 60)
        print("Training Graph Neural Network")
        print("=" * 60)

        if not TORCH_AVAILABLE:
            print("PyTorch not available. Using fallback IDW.")
            return self._fit_fallback(villages, piezometers, water_levels, target_date)

        # Build graph
        self.graph_data = self.graph_builder.build_graph(
            villages, piezometers, water_levels, target_date
        )

        data = self.graph_data['data']
        n_features = data.x.shape[1]

        # Initialize model
        self.model = GroundwaterGNN(
            in_channels=n_features,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers
        )

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )

        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass
            pred, uncertainty = self.model(data.x, data.edge_index, data.edge_attr)

            # Loss only on nodes with known water levels
            mask = data.train_mask & ~torch.isnan(data.y.squeeze())

            if mask.sum() == 0:
                continue

            # MSE loss
            loss = F.mse_loss(pred[mask], data.y[mask])

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 30:
                print(f"  Early stopping at epoch {epoch}")
                break

            if epoch % 50 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            pred, uncertainty = self.model(data.x, data.edge_index)
            mask = data.train_mask & ~torch.isnan(data.y.squeeze())

            y_true = data.y[mask].numpy()
            y_pred = pred[mask].numpy()

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

        print(f"\nGNN Training Results:")
        print(f"  RMSE: {rmse:.3f}m")
        print(f"  R²: {r2:.3f}")
        print("=" * 60)

        return {
            'rmse': rmse,
            'r2': r2,
            'epochs_trained': epoch + 1
        }

    def _fit_fallback(self, villages, piezometers, water_levels, target_date):
        """Fallback to IDW when PyTorch not available."""
        self.graph_data = {
            'villages': villages,
            'piezometers': piezometers,
            'water_levels': water_levels,
            'target_date': target_date
        }
        return {'method': 'IDW_fallback'}

    def predict(self, villages: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict water levels for villages.

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._predict_fallback(villages)

        self.model.eval()
        data = self.graph_data['data']
        n_villages = self.graph_data['n_villages']

        with torch.no_grad():
            pred, uncertainty = self.model(data.x, data.edge_index)

            # Get village predictions
            village_preds = pred[:n_villages].numpy().squeeze()
            village_unc = uncertainty[:n_villages].numpy().squeeze()

        return village_preds, village_unc

    def _predict_fallback(self, villages):
        """IDW fallback prediction."""
        if self.graph_data is None:
            return np.full(len(villages), 5.0), np.full(len(villages), 2.0)

        # Simple IDW
        piezometers = self.graph_data['piezometers']
        water_levels = self.graph_data['water_levels']
        target_date = self.graph_data['target_date']

        predictions = []
        uncertainties = []

        for idx, village in villages.iterrows():
            v_lat = village.geometry.centroid.y
            v_lon = village.geometry.centroid.x

            weights = []
            values = []

            for p_idx, piezo in piezometers.iterrows():
                p_lat = piezo.geometry.centroid.y
                p_lon = piezo.geometry.centroid.x

                dist = np.sqrt((v_lat - p_lat)**2 + (v_lon - p_lon)**2) * 111  # km

                if dist < 30:
                    # Get water level
                    id_col = 'piezo_id' if 'piezo_id' in water_levels.columns else 'sno'
                    piezo_id = piezo.get('piezo_id', piezo.get('sno'))
                    level = water_levels[
                        (water_levels[id_col] == piezo_id) &
                        (water_levels['date'].dt.year == target_date.year) &
                        (water_levels['date'].dt.month == target_date.month)
                    ]['water_level'].values

                    if len(level) > 0 and not np.isnan(level[0]):
                        weights.append(1.0 / (dist + 0.1))
                        values.append(level[0])

            if weights:
                weights = np.array(weights)
                values = np.array(values)
                pred = np.average(values, weights=weights)
                unc = np.std(values) if len(values) > 1 else 2.0
            else:
                pred = 5.0
                unc = 5.0

            predictions.append(pred)
            uncertainties.append(unc)

        return np.array(predictions), np.array(uncertainties)


if __name__ == '__main__':
    print("Testing GNN Model...")

    if TORCH_AVAILABLE:
        # Test model architecture
        model = GroundwaterGNN(in_channels=20, hidden_channels=64)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        x = torch.randn(100, 20)
        edge_index = torch.randint(0, 100, (2, 500))
        pred, unc = model(x, edge_index)
        print(f"Output shape: {pred.shape}, Uncertainty shape: {unc.shape}")
    else:
        print("PyTorch not available, skipping test.")
