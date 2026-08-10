;; Batch export org-mode files to RST for Sphinx.
;; Usage (cwd = docs/): emacs --batch --load export.el
;; Publishes into doc/source/ (existing Sphinx root) without clobbering
;; the hand-tuned landing page or the myst tutorial index.
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)

(unless (package-installed-p 'ox-rst)
  (package-refresh-contents)
  (package-install 'ox-rst))

(require 'ox-rst)
(require 'ox-publish)

(setq org-export-with-section-numbers nil)
(setq org-export-with-toc nil)
(setq org-export-with-author nil)
(setq org-export-with-timestamps nil)
(setq org-rst-headline-underline ?-)

(setq org-publish-project-alist
      '(("sphinx-rst"
         :base-directory "./orgmode/"
         :base-extension "org"
         :publishing-directory "../doc/source/"
         :publishing-function org-rst-publish-to-rst
         :recursive t
         :exclude "^index\\.org$\\|^tutorials/index\\.org$"
         :headline-levels 4
         :with-toc nil
         :section-numbers nil
         :with-author nil)))

(org-publish "sphinx-rst" t)
