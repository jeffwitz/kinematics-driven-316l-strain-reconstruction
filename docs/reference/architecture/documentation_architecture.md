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

## Canonical content tree

Current scientific content is organised by domain below each mode:

```text
explanation/{reconstruction,measurement,constitutive,spectral,native-srix,
             identification,evidence}
how-to/{installation,data,mechanics,crystal-plasticity,identification,
        reproduce,extend,maintenance}
reference/{scientific,data,numerics,software,evidence,architecture,operations}
```

Legacy root pages remain available for provenance while their content is
migrated. They are marked `historical` and `legacy` in the manifest and are
not primary navigation entries.

The subject-level coverage contract is
`_audit/scientific_coverage.yml`; the rendered matrix is
`_audit/scientific_coverage_matrix.md`. It separates three dimensions:

```yaml
routing_status: complete | incomplete
content_status: reviewed | partial | stub | blocked
scientific_status: verified | supported | negative | provisional | open | historical
```

`routing_status: complete` means only that every *applicable* route is current
and reachable through the canonical `toctree` menus. It does not assert that
the prose has been semantically reviewed or that the scientific claim is
positive. An inapplicable quadrant is declared with `applicable: false` and a
reason; it is never filled by a placeholder How-to.

`python scripts/check_docs_structure.py` validates manifest coverage, mode/tree
consistency, coverage targets and local navigation targets. Reachability for a
public menu uses only `toctree` edges; inline `{doc}` links are cross-references
and do not make a page a menu entry. A page with `status: historical`,
`provisional`, or `internal` must not be presented as a recommended current
path, and current pages may not link to legacy pages. Redirects are kept out
of navigation and use `orphan`/`nosearch` when Sphinx needs a stub.

## Adding a page

Choose the mode first, then the domain.  Add the page to the manifest and to
the appropriate index/toctree in the same change.  Keep commands in How-to,
contracts in Reference, scientific interpretation in Explanation, and guided
first runs in Tutorials.  Link across modes rather than duplicating content.
