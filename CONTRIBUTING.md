Contributing guidelines
=======================

Thank you for considering to contribute to PyFraME. All contributions are
welcomed and highly appreciated but please read and follow these guidelines.


Getting started
---------------

If you would like to contribute a new feature or change, please create an issue
first where it can be discussed. If fixing a bug there should also be an issue
describing it. See more [below](#feature-requestsproposals-and-bug-reports).
This is to ensure that it is agreed beforehand how the proposed feature or
bug fix should be implemented and thus avoid wasted effort.

When you are ready to start working on an open issue, please either assign
yourself to the issue if you are a member of the project, or, if you're not a
member, add a comment notifying that you're working on it. Fork the repository
or you may choose to create a branch if you're a member of the project and make
your changes there. Before creating a merge request, please go through the
[merge requests](#merge-requests) section.


Feature requests/proposals and bug reports
------------------------------------------

Check that the feature request/proposal or bug report is not already present in
the list of [issues](https://gitlab.com/FraME-projects/PyFraME/issues). If it
is not there then please create a new issue giving as much information as
possible. Generally you should include the following:

* Use a descriptive title
* Describe the feature/change or bug and include context
    * if describing a bug, explain what happens and what should happen
    * if requesting/proposing a feature or improvement, explain what you are
      trying to accomplish, how it will work, and what is the difference from
      current behavior
* Suggest a possible solution (not obligatory)
* If reporting a bug please provide the steps to reproduce it
* Describe your environment, e.g. PyFraME version, Python version, etc.

When you create a new issue there is a template where these points are
described in more detail.


Merge requests
--------------

Please go through this checklist before creating a merge request:

* Read the [contributing guidelines](https://gitlab.com/FraME-projects/PyFraME/blob/master/CONTRIBUTING.md)
* Make sure your changes conform to the [coding style](#coding-style) of this
  project
* Use nice [git commit messages](#git-commit-messages)
* Add tests to cover your changes
* Make sure all new and existing tests pass
* Update the documentation if necessary
* Update the [changelog](https://gitlab.com/FraME-projects/PyFraME/blob/master/CHANGELOG.md)
* Check that your changes can be merged using
  [fast-forward merge](https://docs.gitlab.com/ee/user/project/merge_requests/fast_forward_merge.html)

When creating the merge request, write a title that summarizes your changes.
If you intend to squash the commits then the merge-request title becomes the
commit message and should therefore not exceed 72 characters and preferably not
be longer than 50 characters (see [rules 2-4](#git-commit-messages) as
guidelines). Describe your changes and include a link to the relevant issue.
Please also indicate the types of changes that the merge request introduces,
i.e. bug fix (non-breaking change which fixes an issue), new feature
(non-breaking change which adds functionality), breaking change (fix or feature
that would cause existing functionality to change), ans/or other
(e.g. documentation). If the merge request is work in progress, remember to add
"WIP:" in the title. This will prevent it from being accidentally merged.


Git commit messages
-------------------

Use [the seven rules of a great git commit message](https://chris.beams.io/posts/git-commit/#seven-rules)
as guidelines when writing git commit messages:

1. [Separate subject from body with a blank line](https://chris.beams.io/posts/git-commit/#separate)
2. [Try to limit the subject line to 50 characters as far as possible and do not
  exceed 72 characters](https://chris.beams.io/posts/git-commit/#limit-50)
3. [Capitalize the subject line](https://chris.beams.io/posts/git-commit/#capitalize)
4. [Do not end the subject line with a period](https://chris.beams.io/posts/git-commit/#end)
5. [Use the imperative mood in the subject line](https://chris.beams.io/posts/git-commit/#imperative)
6. [Wrap the body at 72 characters](https://chris.beams.io/posts/git-commit/#wrap-72)
7. [Use the body to explain what and why vs. how](https://chris.beams.io/posts/git-commit/#why-not-how)


Coding style
------------

Changes to PyFraME should conform to the existing code style which generally
follows the [Style Guide for Python code (aka PEP 8)](https://www.python.org/dev/peps/pep-0008/).
