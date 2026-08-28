# Documentation architecture

The documentation follows the four Diátaxis modes.  The mode describes the
reader's task; the domain describes the subject.  They are independent.

| Question | Mode |
|---|---|
| Am I teaching a newcomer? | Tutorial |
| Am I showing one concrete task? | How-to |
| Am I defining an interface, parameter or invariant? | Reference |
| Am I explaining why, how, or what a result means? | Explanation |

## Navigation contract

The canonical map is `docs/diataxis_manifest.yml`.  Every Markdown or
reStructuredText page is declared there (exactly or through one glob), with a
mode, domain, status and primary navigation route.  The public portals are:

- **Tutorials** — learn by doing;
- **How-to** — complete one task;
- **Reference** — look up stable contracts;
- **Explanation** — understand the science;
- **Scientific evidence** — find claims, artefacts and permitted conclusions;
- **Maintainers** — architecture, ADRs and repository operations.

Evidence and maintainer portals route to the four modes; they are not a fifth
or sixth Diátaxis quadrant and must not become catch-all pages.

`python scripts/check_docs_structure.py` validates manifest coverage and local
toctree/`{doc}` targets.  A page with `status: historical`, `provisional`, or
`internal` must not be presented as a recommended current path.  Redirects are
kept out of navigation and use `orphan`/`nosearch` when Sphinx needs a stub.

## Adding a page

Choose the mode first, then the domain.  Add the page to the manifest and to
the appropriate index/toctree in the same change.  Keep commands in How-to,
contracts in Reference, scientific interpretation in Explanation, and guided
first runs in Tutorials.  Link across modes rather than duplicating content.
