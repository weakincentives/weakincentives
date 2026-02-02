# WINK Presentation

A Reveal.js presentation introducing WINK (Weak Incentives for Agents).

## Viewing the Presentation

### Option 1: Direct File Access

Open `wink-presentation.html` directly in your browser. Most features work without a server.

### Option 2: Local Server (recommended)

For full functionality, serve the file with a local HTTP server:

```bash
# Python
python -m http.server 8000

# Then open http://localhost:8000/wink-presentation.html
```

## Navigation

- **Arrow keys**: Navigate between slides
- **Space**: Next slide
- **Escape**: Overview mode
- **S**: Speaker notes (if any)
- **F**: Fullscreen
- **?**: Show keyboard shortcuts

## Slide Structure

1. **Introduction** - What is WINK?
2. **Philosophy** - Weak incentives, definition vs harness, policies over workflows
3. **Key Principles** - The five core principles
4. **Core Abstractions** - Prompts, Tools, Sessions, Adapters
5. **Prompts** - PromptTemplate, Sections, structured output
6. **Tools** - Contracts, transactions, policies, suites
7. **Sessions** - Event-driven state, reducers, snapshots
8. **Adapters** - Provider abstraction, Claude Agent SDK
9. **Progressive Disclosure** - Dynamic context management
10. **Production Pattern** - AgentLoop, EvalLoop, LoopGroup
11. **Debugging** - Events, bundles, debug UI, SQL query
12. **Workspace Tools** - VFS, Podman, Workspace Digest
13. **Testing** - Testing pyramid and examples
14. **Getting Started** - Installation and learning path

## Customization

The presentation uses Reveal.js from CDN. To customize:

- **Theme**: Change the theme CSS link (e.g., `white.css`, `moon.css`, `dracula.css`)
- **Transitions**: Modify `transition` option in the Reveal.initialize call
- **Styling**: Edit the `<style>` block for custom CSS

## Building from Source

This is a standalone HTML file using Reveal.js from CDN. No build step required.

For offline use or to bundle dependencies locally:

```bash
npm init -y
npm install reveal.js
# Update the CDN links to local paths
```
