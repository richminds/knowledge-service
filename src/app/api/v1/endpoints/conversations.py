from fastapi import APIRouter

from app.api.deps import ConversationStoreDep
from app.core.audit import audit
from app.core.auth import CurrentUser

router = APIRouter()


@router.delete("/me")
async def delete_my_conversations(
    user: CurrentUser, conversations: ConversationStoreDep
) -> dict[str, int]:
    """Right to be forgotten (C-202): erase the caller's stored conversation history."""
    deleted = await conversations.delete_user(user.user_id)
    audit("delete_conversations", user_id=user.user_id, deleted=deleted)
    return {"deleted_conversations": deleted}
