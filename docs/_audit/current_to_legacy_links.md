# Audit of current-to-legacy documentation links

This audit is generated as part of the semantic Diátaxis review. A current
page may link to an internal maintainer document when that link is explicitly
architectural, but it must not use a historical/legacy page as its current
procedure, contract or explanation.

## Result

After rerouting the public entry points and current How-to wrappers, no
`status: current` page links to a manifest entry with `status: historical` or
`navigation: legacy`. The only non-public route retained is the maintainer
portal's intentional link to `adr/index.md` (`status: internal`).

The checker enforces this rule for both `{doc}` references and primary
`toctree` entries. Inline links to historical material are therefore not a
way to make a page menu-reachable, and they are rejected when they originate
from a current page.
