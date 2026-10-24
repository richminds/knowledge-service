from fastapi import APIRouter, status

from app.core import metrics
from app.core.auth import CurrentUser
from app.core.logging import get_logger
from app.models.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(feedback: FeedbackRequest, user: CurrentUser) -> FeedbackResponse:
    """Thumbs up/down on an answer. `response_id` comes from the chat response.

    Stored in Application Insights (no database): join on response_id with the
    `rag_answer` log line to see the question's mode, confidence and scores.
    """
    mode = str(feedback.rag_mode) if feedback.rag_mode else "unknown"
    metrics.feedback.add(1, {"rag.mode": mode, "rating": feedback.rating})
    logger.info(
        "user_feedback",
        response_id=feedback.response_id,
        rating=feedback.rating,
        rag_mode=mode,
        comment=feedback.comment,
        user_id=user.user_id,
    )
    return FeedbackResponse()
