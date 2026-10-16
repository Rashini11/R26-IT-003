from fastapi import APIRouter, Depends

from ...auth import require_authenticated_request
from ..controllers.detection_controller import (
    create_vessel_detection_record,
    delete_vessel_detection_record,
    edit_vessel_detection_record,
    get_demo_boat_detection,
    get_vessel_detection_history,
    get_vessel_port_country,
    predict_boat_detection,
    predict_vessel_video,
)


router = APIRouter()

router.add_api_route(
    "/demo-boat-detection",
    get_demo_boat_detection,
    methods=["GET"],
)
router.add_api_route(
    "/vessel-detection/port-country",
    get_vessel_port_country,
    methods=["GET"],
    dependencies=[Depends(require_authenticated_request)],
)
router.add_api_route(
    "/predict-boat-detection",
    predict_boat_detection,
    methods=["POST"],
    dependencies=[Depends(require_authenticated_request)],
)
router.add_api_route(
    "/predict-boat-video",
    predict_vessel_video,
    methods=["POST"],
    dependencies=[Depends(require_authenticated_request)],
)
router.add_api_route(
    "/vessel-detection-records",
    create_vessel_detection_record,
    methods=["POST"],
    dependencies=[Depends(require_authenticated_request)],
)
router.add_api_route(
    "/vessel-detection-history",
    get_vessel_detection_history,
    methods=["GET"],
    dependencies=[Depends(require_authenticated_request)],
)
router.add_api_route(
    "/vessel-detection-history/{record_id}",
    edit_vessel_detection_record,
    methods=["PUT"],
    dependencies=[Depends(require_authenticated_request)],
)
router.add_api_route(
    "/vessel-detection-history/{record_id}",
    delete_vessel_detection_record,
    methods=["DELETE"],
    dependencies=[Depends(require_authenticated_request)],
)
