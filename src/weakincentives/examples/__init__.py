# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example scripts and shared utilities for the weakincentives project."""

from .code_review_prompt import (
    ReviewGuidance,
    ReviewResponse,
    ReviewTurnParams,
    build_code_review_prompt,
)
from .code_review_session import (
    CodeReviewSession,
    SupportsReviewEvaluate,
    ToolCallLog,
)
from .code_review_tools import (
    MAX_OUTPUT_CHARS,
    REPO_ROOT,
    BranchListParams,
    BranchListResult,
    GitLogParams,
    GitLogResult,
    TagListParams,
    TagListResult,
    TimeQueryParams,
    TimeQueryResult,
    branch_list_handler,
    build_tools,
    current_time_handler,
    git_log_handler,
    tag_list_handler,
)

__all__ = [
    "MAX_OUTPUT_CHARS",
    "REPO_ROOT",
    "BranchListParams",
    "BranchListResult",
    "GitLogParams",
    "GitLogResult",
    "TagListParams",
    "TagListResult",
    "TimeQueryParams",
    "TimeQueryResult",
    "ReviewGuidance",
    "ReviewResponse",
    "ReviewTurnParams",
    "ToolCallLog",
    "SupportsReviewEvaluate",
    "CodeReviewSession",
    "build_tools",
    "build_code_review_prompt",
    "git_log_handler",
    "current_time_handler",
    "branch_list_handler",
    "tag_list_handler",
]
