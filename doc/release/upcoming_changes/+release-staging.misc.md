The release recipe stages `CHANGELOG.md` and the fragments that towncrier removed.
It no longer runs `git add` on the fragment folder after towncrier deletes it, which failed with `fatal: pathspec`.
