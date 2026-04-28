"""
Knowledge Graph Interface for structured access to KG datasets stored as Pandas DataFrames and NetworkX graphs.
"""
import traceback

import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Optional, Dict, Set, List, Tuple
from loguru import logger


# ============================================================================
# Dataset Configuration Classes
# ============================================================================

class FB15k_237_Config:
    """Configuration for FB15k-237 dataset"""
    incomplete_path = "brink_dataset/fb15k_237/knowledge_graph_complete.tsv"
    complete_path = "brink_dataset/fb15k_237/knowledge_graph_incomplete.tsv"
    # relation_path = None
    # entity_path = None 


class FamilyConfig:
    """Configuration for Family dataset"""
    incomplete_path = "brink_dataset/family/knowledge_graph_complete.tsv"
    complete_path = "brink_dataset/family/knowledge_graph_incomplete.tsv"
    # relations_path = None
    # entities_path = None


class Wikidata5mConfig:
    """Configuration for Wikidata5m dataset"""
    incomplete_path = "brink_dataset/wikidata5m/wikidata5m/kg/incomplete_facts.tsv"
    complete_path = "brink_dataset/wikidata5m/wikidata5m/kg/facts.tsv"
    relations_path = "brink_dataset/wikidata5m/wikidata5m/kg/relations.txt"
    entities_path = "brink_dataset/wikidata5m/wikidata5m/kg/entities.txt"


# ============================================================================
# Dataset Registry
# ============================================================================

DATASET_CONFIG_MAP = {
    "fb15k_237": FB15k_237_Config,
    "family": FamilyConfig,
    "wikidata5m": Wikidata5mConfig,
}


# ============================================================================
# Knowledge Graph Interface
# ============================================================================

class KGInterface:
    """
    Interface for interacting with Knowledge Graphs stored as Pandas DataFrames.
    
    The KG is represented as a DataFrame with columns: [head, relation, tail]
    Supports both 'complete' and 'incomplete' graphs via kg_type parameter (defaults to 'incomplete').
    """
    
    def __init__(self, dataset_name: str, base_path: Optional[str] = None):
        """
        Initialize the KG interface for a given dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'fb15k_237', 'family', 'wikidata5m')
            base_path: Optional base path to prepend to config file paths
            
        Raises:
            ValueError: If dataset_name is not in DATASET_CONFIG_MAP
        """
        if dataset_name.lower() not in DATASET_CONFIG_MAP:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {list(DATASET_CONFIG_MAP.keys())}"
            )
        
        self.dataset_name = dataset_name.lower()
        self.config = DATASET_CONFIG_MAP[self.dataset_name]
        self.base_path = Path(base_path) if base_path else Path(".")
        
        # Load datasets and build NetworkX graphs
        self._load_datasets()
        self._build_nx_graphs()
        
        logger.info(f"Initialized KG Interface for dataset: {self.dataset_name}")
        logger.info(f"Complete graph: {len(self.complete_kg)} triples")
        logger.info(f"Incomplete graph: {len(self.incomplete_kg)} triples")
        logger.info(f"Relations: {len(self.relations)} (optional)")
        logger.info(f"Entities: {len(self.entities)} (optional)")
        logger.info(f"NetworkX Complete Graph - Nodes: {self.complete_graph_nx.number_of_nodes()}, Edges: {self.complete_graph_nx.number_of_edges()}")
        logger.info(f"NetworkX Incomplete Graph - Nodes: {self.incomplete_graph_nx.number_of_nodes()}, Edges: {self.incomplete_graph_nx.number_of_edges()}")
    
    def _validate_kg_type(self, kg_type: str) -> str:
        """Validate and normalize kg_type parameter."""
        kg_type = kg_type.lower()
        if kg_type not in ['complete', 'incomplete']:
            raise ValueError(f"kg_type must be 'complete' or 'incomplete', got: {kg_type}")
        return kg_type
    
    def _get_kg(self, kg_type: str = 'incomplete') -> pd.DataFrame:
        """Get the appropriate KG dataframe based on kg_type."""
        kg_type = self._validate_kg_type(kg_type)
        return self.complete_kg if kg_type == 'complete' else self.incomplete_kg
    
    def _get_graph(self, kg_type: str = 'incomplete') -> nx.MultiDiGraph:
        """Get the appropriate NetworkX graph based on kg_type."""
        kg_type = self._validate_kg_type(kg_type)
        return self.complete_graph_nx if kg_type == 'complete' else self.incomplete_graph_nx
    
    def _load_datasets(self):
        """Load KG dataframes and metadata from files."""
        # Load complete KG
        complete_path = self.base_path / self.config.complete_path
        self.complete_kg = pd.read_csv(complete_path, sep='\t' if complete_path.suffix == '.tsv' else ',')
        self.complete_kg.columns = ['head', 'relation', 'tail']
        self.complete_kg = self.complete_kg.astype(str)
        
        # Load incomplete KG
        incomplete_path = self.base_path / self.config.incomplete_path
        self.incomplete_kg = pd.read_csv(incomplete_path, sep='\t' if incomplete_path.suffix == '.tsv' else ',')
        self.incomplete_kg.columns = ['head', 'relation', 'tail']
        self.incomplete_kg = self.incomplete_kg.astype(str)
        
        # Load relations and entities (optional)
        self.relations = set()
        if hasattr(self.config, 'relations_path'):
            relations_path = self.base_path / self.config.relations_path
            if relations_path.exists():
                try:
                    self.relations = set(line.strip() for line in open(relations_path) if line.strip())
                except Exception as e:
                    logger.warning(f"Failed to load relations from {relations_path}: {e}")
            else:
                logger.debug(f"Relations file not found: {relations_path}")
        else:
            self.relations = list(self.complete_kg['relation'].unique())
        
        self.entities = set()
        if hasattr(self.config, 'entities_path'):
            entities_path = self.base_path / self.config.entities_path
            if entities_path.exists():
                try:
                    self.entities = set(line.strip() for line in open(entities_path) if line.strip())
                except Exception as e:
                    logger.warning(f"Failed to load entities from {entities_path}: {e}")
            else:
                logger.debug(f"Entities file not found: {entities_path}")
        else:
            self.entities = set(pd.concat([
                self.complete_kg['head'].astype(str), 
                self.complete_kg['tail'].astype(str)
            ]).unique())
        
        # Build indices for fast lookup
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for efficient triple lookup."""
        # Index for complete KG: entity -> triples where it appears
        self.complete_head_index = self.complete_kg.groupby('head').apply(lambda x: x.values.tolist()).to_dict()
        self.complete_tail_index = self.complete_kg.groupby('tail').apply(lambda x: x.values.tolist()).to_dict()
        self.complete_relation_index = self.complete_kg.groupby('relation').apply(lambda x: x.values.tolist()).to_dict()
        
        # Index for incomplete KG
        self.incomplete_head_index = self.incomplete_kg.groupby('head').apply(lambda x: x.values.tolist()).to_dict()
        self.incomplete_tail_index = self.incomplete_kg.groupby('tail').apply(lambda x: x.values.tolist()).to_dict()
        self.incomplete_relation_index = self.incomplete_kg.groupby('relation').apply(lambda x: x.values.tolist()).to_dict()
    
    def _build_nx_graphs(self):
        """Build NetworkX MultiDiGraph representations for fast graph processing."""
        # Create MultiDiGraph (allows multiple edges between same nodes with different relations)
        self.complete_graph_nx = nx.MultiDiGraph()
        self.incomplete_graph_nx = nx.MultiDiGraph()
        
        # Build complete graph
        for _, row in self.complete_kg.iterrows():
            head, relation, tail = row['head'], row['relation'], row['tail']
            self.complete_graph_nx.add_edge(head, tail, relation=relation)
        
        # Build incomplete graph
        for _, row in self.incomplete_kg.iterrows():
            head, relation, tail = row['head'], row['relation'], row['tail']
            self.incomplete_graph_nx.add_edge(head, tail, relation=relation)
        
        logger.debug(f"Built NetworkX graphs for {self.dataset_name}")
    
    # ========================================================================
    # Query Methods
    # ========================================================================
    
    def get_1hop_triples(self, entity: str, kg_type: str = 'incomplete') -> pd.DataFrame:
        """
        Get all 1-hop triples from an entity.
        
        Args:
            entity: Entity ID/name
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            DataFrame with triples where entity is head or tail
        """
        kg = self._get_kg(kg_type)
        head_triples = kg[kg['head'] == entity]
        tail_triples = kg[kg['tail'] == entity]
        return pd.concat([head_triples, tail_triples], ignore_index=True)
    
    def get_relations_for_entity(self, entity: str, direction: str = 'both', kg_type: str = 'incomplete') -> Set[str]:
        """
        Get all relations connected to an entity.
        
        Args:
            entity: Entity ID/name
            direction: 'outgoing' (as head), 'incoming' (as tail), or 'both'
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            Set of relation names
        """
        kg = self._get_kg(kg_type)
        relations = set()
        if direction in ['outgoing', 'both']:
            relations.update(kg[kg['head'] == entity]['relation'].unique())
        if direction in ['incoming', 'both']:
            relations.update(kg[kg['tail'] == entity]['relation'].unique())
        return relations
    
    def get_triples_by_relation(self, relation: str, kg_type: str = 'incomplete') -> pd.DataFrame:
        """Get all triples with a specific relation."""
        kg = self._get_kg(kg_type)
        return kg[kg['relation'] == relation].copy()
    
    def get_tail_entities(self, head: str, relation: str, kg_type: str = 'incomplete') -> List[str]:
        """Get all tail entities for a given (head, relation) pair."""
        kg = self._get_kg(kg_type)
        triples = kg[
            (kg['head'] == head) & 
            (kg['relation'] == relation)
        ]
        return triples['tail'].tolist()
    
    # ========================================================================
    # NetworkX-based Graph Methods
    # ========================================================================
    
    def find_shortest_path(self, source: str, target: str, kg_type: str = 'incomplete') -> Optional[List[str]]:
        """
        Find shortest path between two entities using NetworkX.
        
        Args:
            source: Source entity
            target: Target entity
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            List of entities in shortest path, or None if no path exists
        """
        graph = self._get_graph(kg_type)
        try:
            return nx.shortest_path(graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def find_all_paths(self, source: str, target: str, max_length: int = 3, kg_type: str = 'incomplete') -> List[List[str]]:
        """
        Find all paths up to a maximum length between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            List of paths (each path is a list of entities)
        """
        graph = self._get_graph(kg_type)
        try:
            return list(nx.all_simple_paths(
                graph, source, target, cutoff=max_length
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_shortest_path_with_relations(self, source: str, target: str, kg_type: str = 'incomplete') -> Optional[Dict]:
        """
        Get the shortest path between two entities and return both entity path and relation sequence.
        
        Args:
            source: Source entity
            target: Target entity
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            Dictionary with keys:
                - 'path': List of entities in shortest path
                - 'relations': List of relations along the path
                - 'length': Number of hops
                or None if no path exists
        """
        graph = self._get_graph(kg_type)
        try:
            entity_path = nx.shortest_path(graph, source, target)
            
            # Extract relations from consecutive entity pairs
            relations = []
            for i in range(len(entity_path) - 1):
                head, tail = entity_path[i], entity_path[i + 1]
                edge_data = graph.get_edge_data(head, tail)
                if edge_data:
                    # Handle multiple edges (MultiDiGraph)
                    relation = edge_data[0]['relation']
                    relations.append(relation)
            
            return {
                'path': entity_path,
                'relations': relations,
                'length': len(entity_path) - 1
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_neighbors(self, entity: str, direction: str = 'both', kg_type: str = 'incomplete') -> Set[str]:
        """
        Get neighboring entities using NetworkX.
        
        Args:
            entity: Entity ID
            direction: 'outgoing' (successors), 'incoming' (predecessors), or 'both'
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            Set of neighboring entities
        """
        graph = self._get_graph(kg_type)
        neighbors = set()
        if direction in ['outgoing', 'both']:
            neighbors.update(graph.successors(entity))
        if direction in ['incoming', 'both']:
            neighbors.update(graph.predecessors(entity))
        return neighbors
    
    def get_subgraph(self, entity: str, hops: int = 1, kg_type: str = 'incomplete') -> nx.MultiDiGraph:
        """
        Extract subgraph centered at entity.
        
        Args:
            entity: Center entity
            hops: Number of hops to include
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            NetworkX subgraph
        """
        graph = self._get_graph(kg_type)
        # Get all entities within n hops
        visited = {entity}
        frontier = {entity}
        
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                neighbors = self.get_neighbors(node, kg_type=kg_type)
                next_frontier.update(neighbors - visited)
            visited.update(next_frontier)
            frontier = next_frontier
        
        return graph.subgraph(visited).copy()
    
    def get_connected_components(self, kg_type: str = 'incomplete') -> List[Set[str]]:
        """Get all weakly connected components."""
        graph = self._get_graph(kg_type)
        return [set(component) for component in nx.weakly_connected_components(graph)]
    
    def get_entity_degree(self, entity: str, kg_type: str = 'incomplete') -> Dict[str, int]:
        """
        Get in-degree and out-degree for an entity.
        
        Args:
            entity: Entity ID
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            Dict with 'in_degree' and 'out_degree'
        """
        graph = self._get_graph(kg_type)
        if entity not in graph:
            return {"in_degree": 0, "out_degree": 0}
        
        return {
            "in_degree": graph.in_degree(entity),
            "out_degree": graph.out_degree(entity)
        }
    
    def get_relation_paths(self, source: str, target: str, kg_type: str = 'incomplete') -> List[List[str]]:
        """
        Get all relations paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            kg_type: 'complete' or 'incomplete' (default: 'incomplete')
            
        Returns:
            List of relation paths (each path is a list of relations)
        """
        graph = self._get_graph(kg_type)
        all_paths = self.find_all_paths(source, target, kg_type=kg_type)
        relation_paths = []
        
        for path in all_paths:
            relation_path = []
            for i in range(len(path) - 1):
                # Get edge data between consecutive nodes
                edge_data = graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    relations = [edge_data[key]['relation'] for key in edge_data]
                    relation_path.extend(relations)
            if relation_path:
                relation_paths.append(relation_path)
        
        return relation_paths
    
    def get_pagerank(self, kg_type: str = 'incomplete') -> Dict[str, float]:
        """Calculate PageRank for all entities."""
        graph = self._get_graph(kg_type)
        return nx.pagerank(graph)
    
    def get_graph_statistics(self, kg_type: str = 'incomplete') -> Dict:
        """Get comprehensive graph statistics."""
        graph = self._get_graph(kg_type)
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "num_components": nx.number_weakly_connected_components(graph),
            "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        }
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
        """Check if an entity exists in the KG.
        
        Note: Requires entities file to be loaded. Returns False if not available.
        """
        if not self.entities:
            logger.debug("Entities file not loaded. Cannot verify entity existence.")
            return False
        return entity in self.entities
    
    def relation_exists(self, relation: str) -> bool:
        """Check if a relation exists in the KG.
        
        Note: Requires relations file to be loaded. Returns False if not available.
        """
        if not self.relations:
            logger.debug("Relations file not loaded. Cannot verify relation existence.")
            return False
        return relation in self.relations
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset including NetworkX graph metrics."""
        return {
            "dataset_name": self.dataset_name,
            "complete_triples": len(self.complete_kg),
            "incomplete_triples": len(self.incomplete_kg),
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "complete_entities": len(set(pd.concat([
                self.complete_kg['head'], 
                self.complete_kg['tail']
            ]))),
            "incomplete_entities": len(set(pd.concat([
                self.incomplete_kg['head'], 
                self.incomplete_kg['tail']
            ]))),
            "complete_graph_stats": self.get_graph_statistics_complete(),
            "incomplete_graph_stats": self.get_graph_statistics_incomplete(),
        }


if __name__ == "__main__":
    print("hello world")
