#!/usr/bin/env python3
"""
Smart Inference Router - Intelligent load balancing with cost optimization
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RequestPriority:
    """Priority levels for requests"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    
@dataclass
class BackendType:
    """Backend types"""
    MLX = "mlx"  # Apple Silicon
    CUDA = "cuda"  # NVIDIA GPUs
    CPU = "cpu"  # CPU-only
    HYBRID = "hybrid"  # Mixed precision

@dataclass
class NodeHealth:
    """Node health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class BackendNode:
    """Representation of a backend inference node"""
    
    def __init__(self, 
                node_id: str, 
                backend_type: str,
                models: List[str],
                capacity: Dict[str, Any],
                cost_per_token: float = 0.0):
        self.node_id = node_id
        self.backend_type = backend_type
        self.models = models
        self.capacity = capacity
        self.cost_per_token = cost_per_token
        self.current_load = 0
        self.queue_depth = 0
        self.health_status = NodeHealth.HEALTHY
        self.last_latency = {}
        self.error_count = 0
        self.last_error_time = 0
        
    def can_handle(self, model: str, batch_size: int = 1) -> bool:
        """Check if node can handle the requested model and batch size"""
        return (model in self.models and 
                self.health_status != NodeHealth.UNHEALTHY and
                self.current_load + batch_size <= self.capacity.get("max_batch_size", 1))
    
    def get_expected_latency(self, model: str, tokens: int) -> float:
        """Estimate expected latency based on historical data"""
        if model in self.last_latency:
            base_latency = self.last_latency[model]
            # Factor in current load
            load_factor = 1.0 + (self.current_load / self.capacity.get("max_batch_size", 1))
            # Factor in queue depth
            queue_factor = 1.0 + (0.1 * self.queue_depth)
            return base_latency * load_factor * queue_factor
        return float('inf')  # Unknown latency
        
    def update_metrics(self, 
                      model: str, 
                      latency: float, 
                      tokens: int,
                      success: bool = True):
        """Update node metrics after request completion"""
        # Update latency metrics with exponential moving average
        alpha = 0.2  # Smoothing factor
        if model in self.last_latency:
            self.last_latency[model] = (1 - alpha) * self.last_latency[model] + alpha * latency
        else:
            self.last_latency[model] = latency
            
        # Update error metrics
        if not success:
            self.error_count += 1
            self.last_error_time = time.time()
            
            # Update health status based on error rate
            if self.error_count > 5:
                self.health_status = NodeHealth.DEGRADED
            if self.error_count > 10:
                self.health_status = NodeHealth.UNHEALTHY
        else:
            # Gradually reduce error count on successful requests
            self.error_count = max(0, self.error_count - 0.2)
            
            # Recover health status
            if self.error_count < 3 and self.health_status != NodeHealth.HEALTHY:
                self.health_status = NodeHealth.HEALTHY

class AdaptiveRoutingStrategy:
    """Adaptive routing strategy with multi-objective optimization"""
    
    def __init__(self, 
                 weights: Dict[str, float] = None):
        """Initialize routing strategy
        
        Args:
            weights: Weight factors for different routing objectives
                - latency: Weight for latency optimization
                - cost: Weight for cost optimization
                - reliability: Weight for reliability
        """
        self.weights = weights or {
            "latency": 0.6,
            "cost": 0.3,
            "reliability": 0.1
        }
    
    def select_node(self, 
                   available_nodes: List[BackendNode],
                   model: str,
                   priority: int = RequestPriority.NORMAL,
                   expected_tokens: int = 1000,
                   sla_ms: Optional[int] = None) -> Optional[BackendNode]:
        """Select optimal backend node for request
        
        Args:
            available_nodes: List of available backend nodes
            model: Requested model name
            priority: Request priority level
            expected_tokens: Expected number of tokens (prompt + completion)
            sla_ms: Service level agreement in milliseconds (if any)
            
        Returns:
            Selected backend node or None if no suitable node found
        """
        # Filter nodes that can handle the model
        capable_nodes = [node for node in available_nodes if node.can_handle(model)]
        
        if not capable_nodes:
            logger.warning(f"No capable nodes for model {model}")
            return None
            
        # For critical priority, select node with lowest current latency
        if priority == RequestPriority.CRITICAL:
            return min(capable_nodes, 
                       key=lambda n: n.get_expected_latency(model, expected_tokens))
        
        # For normal requests, use weighted scoring
        scored_nodes = []
        for node in capable_nodes:
            # Calculate normalized scores (0-1, lower is better)
            latency_score = self._normalize_latency(
                node.get_expected_latency(model, expected_tokens),
                [n.get_expected_latency(model, expected_tokens) for n in capable_nodes]
            )
            
            cost_score = node.cost_per_token / max(n.cost_per_token for n in capable_nodes) if max(n.cost_per_token for n in capable_nodes) > 0 else 0
            
            reliability_score = self._calculate_reliability_score(node)
            
            # Apply weights
            total_score = (
                self.weights["latency"] * latency_score +
                self.weights["cost"] * cost_score +
                self.weights["reliability"] * reliability_score
            )
            
            scored_nodes.append((node, total_score))
        
        # Select node with best (lowest) score
        selected_node = min(scored_nodes, key=lambda x: x[1])[0]
        logger.debug(f"Selected node {selected_node.node_id} for model {model}")
        
        return selected_node
    
    def _normalize_latency(self, latency: float, all_latencies: List[float]) -> float:
        """Normalize latency score between 0 and 1"""
        min_latency = min(all_latencies)
        max_latency = max(all_latencies)
        
        if min_latency == max_latency:
            return 0.5
            
        return (latency - min_latency) / (max_latency - min_latency)
    
    def _calculate_reliability_score(self, node: BackendNode) -> float:
        """Calculate reliability score based on health status and error rate"""
        if node.health_status == NodeHealth.HEALTHY:
            return 0.0
        elif node.health_status == NodeHealth.DEGRADED:
            return 0.5
        else:
            return 1.0

class InferenceRouter:
    """Intelligent inference request router"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize router with configuration
        
        Args:
            config: Router configuration
        """
        self.config = config
        self.nodes = self._init_backend_nodes()
        self.strategy = AdaptiveRoutingStrategy(
            weights=config.get("routing_weights")
        )
        self.request_queues = {}
        self.queue_tasks = []
        self.engines = {}
        self.cost_tracker = CostTracker()
        
        # Initialize request queues for each priority level
        for priority in range(4):  # 0-3 priority levels
            self.request_queues[priority] = asyncio.Queue()
            
    async def start(self):
        """Start the router with queue processing tasks"""
        self.engines = await self._init_llm_backends()
        
        # Start queue processing tasks
        for priority in range(4):
            task = asyncio.create_task(self._process_queue(priority))
            self.queue_tasks.append(task)
            
        logger.info(f"Inference router started with {len(self.nodes)} backend nodes")
    
    async def _init_llm_backends(self):
        """Initialize available LLM backends"""
        backends = {}
        
        # Initialize MLX backend for Apple Silicon if available
        if self.config.get("enable_mlx", True):
            try:
                from llamascale.drivers.mlx.engine import MLXModelEngine, MLXConfig
                
                mlx_models = self.config.get("mlx_models", [])
                backends["mlx"] = {}
                
                for model in mlx_models:
                    model_path = f"{self.config.get('models_dir', 'models')}/{model}"
                    mlx_config = MLXConfig(**self.config.get("mlx_config", {}))
                    
                    logger.info(f"Initializing MLX engine for model: {model}")
                    backends["mlx"][model] = MLXModelEngine(
                        model_path=model_path,
                        config=mlx_config
                    )
                    
                logger.info(f"MLX backend initialized with models: {list(backends['mlx'].keys())}")
            except ImportError:
                logger.warning("MLX not available, skipping Apple Silicon backend")
            except Exception as e:
                logger.error(f"Error initializing MLX backend: {e}")
        
        # Initialize CUDA backend for NVIDIA GPUs if available
        if self.config.get("enable_cuda", False):
            try:
                # This would be implemented for NVIDIA GPU support
                # For now, just log that it's not implemented
                logger.warning("CUDA backend not implemented yet")
            except Exception as e:
                logger.error(f"Error initializing CUDA backend: {e}")
        
        return backends
    
    def _init_backend_nodes(self):
        """Initialize backend nodes from configuration"""
        nodes = []
        
        # Add MLX nodes
        for i, mlx_config in enumerate(self.config.get("mlx_nodes", [])):
            node = BackendNode(
                node_id=f"mlx-{i}",
                backend_type=BackendType.MLX,
                models=self.config.get("mlx_models", []),
                capacity=mlx_config.get("capacity", {}),
                cost_per_token=mlx_config.get("cost_per_token", 0.0)
            )
            nodes.append(node)
            
        # Add CUDA nodes (if configured)
        for i, cuda_config in enumerate(self.config.get("cuda_nodes", [])):
            node = BackendNode(
                node_id=f"cuda-{i}",
                backend_type=BackendType.CUDA,
                models=self.config.get("cuda_models", []),
                capacity=cuda_config.get("capacity", {}),
                cost_per_token=cuda_config.get("cost_per_token", 0.0)
            )
            nodes.append(node)
            
        return nodes
    
    async def dispatch(self, request):
        """Dispatch request to appropriate backend
        
        This method queues the request according to priority
        and returns when processing is complete.
        """
        priority = request.get("priority", RequestPriority.NORMAL)
        
        # Create a future to track completion
        completion_future = asyncio.Future()
        
        # Create queue item with request and completion future
        queue_item = {
            "request": request,
            "future": completion_future,
            "enqueue_time": time.time()
        }
        
        # Add to appropriate priority queue
        await self.request_queues[priority].put(queue_item)
        logger.debug(f"Request queued with priority {priority}")
        
        # Wait for completion
        result = await completion_future
        return result
    
    async def _process_queue(self, priority):
        """Process requests from queue with given priority"""
        queue = self.request_queues[priority]
        
        while True:
            # Get next request from queue
            queue_item = await queue.get()
            request = queue_item["request"]
            future = queue_item["future"]
            
            # Measure queue time
            queue_time = time.time() - queue_item["enqueue_time"]
            logger.debug(f"Request dequeued after {queue_time:.3f}s with priority {priority}")
            
            node = None
            try:
                # Select appropriate node
                node = self.strategy.select_node(
                    available_nodes=self.nodes,
                    model=request.get("model"),
                    priority=priority,
                    expected_tokens=request.get("max_tokens", 1000),
                    sla_ms=request.get("sla_ms")
                )
                
                if node is None:
                    raise ValueError(f"No suitable backend found for model {request.get('model')}")
                
                # Get the appropriate engine
                engine = self.engines[node.backend_type][request.get("model")]
                
                # Track node load
                node.current_load += 1
                
                start_time = time.time()
                
                # Generate completion
                result = await engine.generate(
                    prompt=request.get("prompt"),
                    max_tokens=request.get("max_tokens", 512),
                    temperature=request.get("temperature", 0.7),
                    top_p=request.get("top_p", 0.9),
                    seed=request.get("seed")
                )
                
                end_time = time.time()
                latency = end_time - start_time
                
                # Update node metrics
                node.update_metrics(
                    model=request.get("model"),
                    latency=latency,
                    tokens=result["usage"]["total_tokens"],
                    success=True
                )
                
                # Track cost
                self.cost_tracker.track_request(
                    model=request.get("model"),
                    tokens=result["usage"]["total_tokens"],
                    cost_per_token=node.cost_per_token
                )
                
                # Add backend info to result
                result["backend"] = {
                    "type": node.backend_type,
                    "node_id": node.node_id,
                    "latency_ms": latency * 1000
                }
                
                # Set result to future
                future.set_result(result)
                
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                future.set_exception(e)
                
                # Update node metrics if a node was selected
                if node:
                    node.update_metrics(
                        model=request.get("model"),
                        latency=0,
                        tokens=0,
                        success=False
                    )
            finally:
                # Update node load
                if node:
                    node.current_load = max(0, node.current_load - 1)
                
                # Mark task as done
                queue.task_done()
    
    async def shutdown(self):
        """Shutdown the router gracefully"""
        for task in self.queue_tasks:
            task.cancel()
            
        await asyncio.gather(*self.queue_tasks, return_exceptions=True)
        logger.info("Inference router shutdown complete")

class CostTracker:
    """Track cost of requests"""
    
    def __init__(self):
        self.total_cost = 0.0
        self.cost_by_model = {}
        self.request_count = 0
        self.token_count = 0
        
    def track_request(self, model: str, tokens: int, cost_per_token: float):
        """Track cost of a request"""
        cost = tokens * cost_per_token
        self.total_cost += cost
        self.request_count += 1
        self.token_count += tokens
        
        if model not in self.cost_by_model:
            self.cost_by_model[model] = 0
        self.cost_by_model[model] += cost
        
    def get_stats(self):
        """Get cost statistics"""
        return {
            "total_cost": self.total_cost,
            "cost_by_model": self.cost_by_model,
            "request_count": self.request_count,
            "token_count": self.token_count,
            "avg_cost_per_request": self.total_cost / self.request_count if self.request_count > 0 else 0,
            "avg_cost_per_token": self.total_cost / self.token_count if self.token_count > 0 else 0
        } 