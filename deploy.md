# Deploy
> Build and deploy to a remote public site


See [Deploying Your Docs](https://www.mkdocs.org/user-guide/deploying-your-docs/) on the Mkdocs site for more details.


## GitHub Pages
> How to deploy your docs site to GitHub Pages.

Follow one of the approaches below:

- [Run deploy command](#run-deploy-command) - Run a MkDocs CLI command locally to deploy.
- [Deploy with GitHub Actions](#deploy-with-github-actions) - Use the project's CI workflow in the cloud to build and deploy to GH Pages on commits pushed to master.

Then go to your repo's _Settings_ and _Pages_ then enable _GitHub Pages_ on the `gh-pages` branch's root.

_Note this is for a Project Page on a subpath, you will have to make adjustments to the command below if you want an Organization or User Page on the root path._

### Run deploy command
> Run a MkDocs CLI command locally to deploy

MkDocs needs to know where to publish commits on GitHub - so make sure you are working with a repo that you cloned, or that you initialize the local repo and add a `remote` repo.

Run this command locally:

```sh
$ make d
```

That will use `Makefile` to run the following:

```sh
$ mkdocs gh-deploy --strict --force
```

That will do the following:

1. Clean and build to `site` directory.
2. Force push to `gh-pages` branch, overwriting any changes which were pushed from another build.

Then go to your repo on GitHub, look at the *Environment* tab.

When it is done building, click _View deployment_ to see your site.

e.g. [michaelcurrin.github.io/mkdocs-quickstart/](https://michaelcurrin.github.io/mkdocs-quickstart/)

See deploy options in the help:

```sh
$ mkdocs gh-deploy --help
```

### Deploy with GitHub Actions
>  Set up continuous deployment config to enable deploys on a change to files on GitHub

When you make changes to your docs config or the docs directory, especially editing on GitHub directly, it's often useful to have the docs site build and deploy automatically in a remote environment. This is provided for free by GitHub.

See the [docs.yml](https://github.com/MichaelCurrin/mkdocs-quickstart/blob/master/.github/workflows/docs.yml) workflow provided with this project. You don't have to change anything there. The token will be generated for you by GitHub Actions.
