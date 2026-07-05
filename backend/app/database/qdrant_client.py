"""
Qdrant client for video keyframes search
"""
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "video_keyframes")


class VideoQdrantClient:
    """Qdrant client for video keyframes search"""

    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = COLLECTION_NAME
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    async def init_collection(self) -> bool:
        """Initialize collection if it does not exist"""
        try:
            from qdrant_client.models import Distance, VectorParams
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            if self.collection_name not in collection_names:
                logger.info(f"Collection '{self.collection_name}' not found. Creating empty collection...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
                logger.info(f"Created empty collection '{self.collection_name}' successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            return False

    def search_by_vector(
        self,
        query_vector: List[float],
        limit: int = 500,
        object_filters: Optional[List[str]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search keyframes by vector similarity

        Args:
            query_vector: CLIP embedding vector (512-dim)
            limit: Maximum number of results
            object_filters: List of objects to filter by
            score_threshold: Minimum similarity score

        Returns:
            List of search results with payload and scores
        """
        try:
            # Build filter conditions
            filter_conditions = []

            # Filter by objects if specified
            if object_filters:
                filter_conditions.append(
                    FieldCondition(
                        key="object_labels",
                        match=MatchAny(any=object_filters)
                    )
                )

            # Create filter object
            search_filter = None
            if filter_conditions:
                search_filter = Filter(must=filter_conditions)

            # Perform vector search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )

            # Format results
            results = []
            for idx, result in enumerate(search_results):
                results.append({
                    "rank": idx + 1,
                    "score": float(result.score),
                    "payload": result.payload
                })
            logger.info(f"Vector search results: {results[:5]}")
            logger.info(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    def search_multiple_vectors(
        self,
        query_vectors: List[List[float]],
        limit: int = 500,
        object_filters: Optional[List[str]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search with multiple vectors and merge results

        Args:
            query_vectors: List of CLIP embedding vectors
            limit: Limit per query (total may be less due to deduplication)
            object_filters: List of objects to filter by
            score_threshold: Minimum similarity score

        Returns:
            Merged and deduplicated results
        """
        try:
            all_results = {}  # Use dict to deduplicate by original_id

            for i, query_vector in enumerate(query_vectors):
                logger.info(f"Processing query vector {i+1}/{len(query_vectors)}")

                results = self.search_by_vector(
                    query_vector=query_vector,
                    limit=limit,
                    object_filters=object_filters,
                    score_threshold=score_threshold
                )

                # Merge results, keeping highest score for duplicates
                for result in results:
                    original_id = result["payload"]["original_id"]

                    if (original_id not in all_results or
                            result["score"] > all_results[original_id]["score"]):
                        all_results[original_id] = result

            # Convert back to list and sort by score
            merged_results = list(all_results.values())
            merged_results.sort(key=lambda x: x["score"], reverse=True)

            # Re-rank after merging
            for idx, result in enumerate(merged_results):
                result["rank"] = idx + 1

            logger.info(f"Merged search returned {len(merged_results)} unique results")
            return merged_results

        except Exception as e:
            logger.error(f"Multiple vector search failed: {e}")
            raise

    def group_results_by_video(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group search results by video_id

        Args:
            results: List of search results

        Returns:
            Dictionary grouped by video_id
        """
        grouped = {}

        for result in results:
            video_id = result["payload"]["video_id"]

            if video_id not in grouped:
                grouped[video_id] = []

            grouped[video_id].append(result)

        # Sort each group by score
        for video_id in grouped:
            grouped[video_id].sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Results grouped into {len(grouped)} videos")
        return grouped

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise


# Global client instance
qdrant_client = VideoQdrantClient()
