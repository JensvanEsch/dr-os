# 0003 - Apache-2.0 licence for the software

## Context
The repository is public but had no licence, so nobody had the right to reuse
the code. The project should become a reusable ecosystem, and it may later grow
into a business. The realistic value of such a business is in kits, tested
hardware, support and the brand, not in licence fees on the Python code. The only
commit author so far is the project owner, and the dependencies (numpy,
feetech-servo-sdk) do not constrain the choice.

## Decision
License the software under **Apache-2.0**, with the full text in `LICENSE`.
Permissive licensing is preferred for adoption, and Apache-2.0 over MIT for its
explicit patent grant. AGPL-3.0 with paid commercial licences was rejected
because it only pays off when selling the software itself, and it would put
companies off building on the project.

## Consequences
- Anyone may use, modify and sell the code, including competitors. Released
  versions stay available under Apache-2.0 for good.
- The owner can still build closed add-ons on top, or change the licence of
  future versions while holding the rights to all of the code.
- Outside contributions arrive under Apache-2.0 by default. Dual licensing later
  would need a contributor agreement in place before the first outside
  contribution.
- This covers the software only. Hardware design files and documentation/media
  have not been licensed yet and need their own decision when they are published.
- The licence does not protect the name; that takes a trademark.
