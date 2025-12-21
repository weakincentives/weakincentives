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

"""Progressive examples demonstrating WINK concepts incrementally.

This module contains a series of examples that build on each other,
introducing WINK concepts one at a time:

1. **01_minimal_prompt.py** - Structured output from a simple prompt
2. **02_with_tools.py** - Adding tools the LLM can call
3. **03_with_session.py** - Multi-turn state with planning tools
4. **04_with_workspace.py** - VFS file exploration
5. **05_full_agent.py** - Complete REPL with MainLoop orchestration

Start with 01 and progress through each example to understand how
WINK's components compose together.
"""
