"""
Global core state holding instantiated singletons of services.
Resolves circular import problems across endpoint routers and background tasks.
"""
import logging
from backend.app.services.compression_service import CompressionService
from backend.app.services.routing_service import RoutingService

logger = logging.getLogger("Core-State")

logger.info("Initializing global service singletons...")
comp_service = CompressionService()
routing_service = RoutingService(comp_service)
logger.info("Core service singletons initialized successfully.")
