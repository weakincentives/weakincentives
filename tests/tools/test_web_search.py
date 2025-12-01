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

from weakincentives.tools.web_search import (
    WebSearchAction,
    WebSearchCall,
    WebSearchSource,
)


def test_web_search_action_render_variants() -> None:
    assert (
        WebSearchAction(type="search", query="llama updates").render()
        == "search query: llama updates"
    )
    assert (
        WebSearchAction(type="open_page", url="https://example.com").render()
        == "opened page: https://example.com"
    )
    assert (
        WebSearchAction(type="find", pattern="hi", url="https://example.com").render()
        == "searched for 'hi' in https://example.com"
    )
    assert WebSearchAction(type="find").render() == "find"


def test_web_search_call_render() -> None:
    action = WebSearchAction(
        type="search",
        query="openai",
        sources=(WebSearchSource(url="https://openai.com"),),
    )
    call = WebSearchCall(id="search_1", status="completed", action=action)

    assert call.render() == "web search (completed): search query: openai"
