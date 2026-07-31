# Releasing PyCOL Optimized

The distribution name is `pycol-optimized`. Releases are published from
GitHub tags through PyPI Trusted Publishing. The workflow uses short-lived
OpenID Connect credentials; no PyPI API token is stored in GitHub.

## One-time publishing configuration

The repository is `spozi/pycol-optimized`, the release workflow is
`.github/workflows/publish.yml`, and the GitHub environment is `pypi`.

For the first release, register a pending publisher at
<https://pypi.org/manage/account/publishing/> with:

- PyPI project name: `pycol-optimized`;
- GitHub owner: `spozi`;
- GitHub repository: `pycol-optimized`;
- workflow filename: `publish.yml`;
- environment name: `pypi`.

Configure the GitHub `pypi` environment to require manual approval before a
production release when the repository plan supports protected environments.

## 1. Prepare the release

1. Update `version` in `pyproject.toml`.
2. Update `__version__` in `src/pycol_optimized/__init__.py` to the same value.
3. Record user-visible changes in `CHANGELOG.md`.
4. Confirm the copyright attribution in `LICENSE`.

## 2. Validate from a clean checkout

```bash
python -m pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
pytest -q
```

The reference tests require the `dev` or `reference` optional dependency
group because they compare against PyCOL 1.0.4 and SciPy.

## 3. Build and inspect

Remove artifacts from an older release, then validate locally:

```bash
python -m build
twine check dist/*
check-wheel-contents dist/*.whl
```

Confirm that `dist/` contains one `.whl` and one `.tar.gz` for the intended
version. Install each artifact in a fresh virtual environment and run a CPU
smoke test before committing the release.

## 4. Commit and verify CI

```bash
git add .
git commit -m "Release vX.Y.Z"
git push origin main
```

Wait for the complete CI matrix on `main` to pass.

## 5. Tag and publish

```bash
git tag -a vX.Y.Z -m "PyCOL Optimized vX.Y.Z"
git push origin vX.Y.Z
```

The tag starts `.github/workflows/publish.yml`. That workflow independently
checks the tag/version match, runs linting and tests, builds fresh artifacts,
validates them, and publishes them through the `pypi` environment.

## 6. Verify

Confirm the workflow completed, inspect
<https://pypi.org/project/pycol-optimized/>, and install the published version
in a fresh environment. PyPI files cannot be replaced for the same project
version. If an artifact is wrong, increment the version, rebuild from a clean
tree, and publish the new version.
