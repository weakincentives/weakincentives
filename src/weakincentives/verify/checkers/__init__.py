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

"""Verification checkers.

This package contains individual verification passes organized by category:

- architecture: Module boundaries, layering, core/contrib separation
- documentation: Spec references, doc examples, markdown links/format
- security: Bandit, pip-audit
- dependencies: Deptry
- types: Type coverage, integration test validation
- tests: Pytest execution
"""

from weakincentives.verify.checkers.architecture import (
    CoreContribSeparationChecker,
    LayerViolationsChecker,
)
from weakincentives.verify.checkers.dependencies import DeptryChecker
from weakincentives.verify.checkers.documentation import (
    DocExamplesChecker,
    MarkdownFormatChecker,
    MarkdownLinksChecker,
    SpecReferencesChecker,
)
from weakincentives.verify.checkers.security import BanditChecker, PipAuditChecker
from weakincentives.verify.checkers.tests import PytestChecker
from weakincentives.verify.checkers.types import (
    IntegrationTypesChecker,
    TypeCoverageChecker,
)

__all__ = [
    # Architecture
    "LayerViolationsChecker",
    "CoreContribSeparationChecker",
    # Documentation
    "SpecReferencesChecker",
    "DocExamplesChecker",
    "MarkdownLinksChecker",
    "MarkdownFormatChecker",
    # Security
    "BanditChecker",
    "PipAuditChecker",
    # Dependencies
    "DeptryChecker",
    # Types
    "TypeCoverageChecker",
    "IntegrationTypesChecker",
    # Tests
    "PytestChecker",
]
