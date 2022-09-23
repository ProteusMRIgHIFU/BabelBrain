# Usage
> Build and preview a site locally

Make sure to run all commands from the `docs` directory, as that is where `Makefile` is.

```sh
$ cd docs
```


## CLI help

```sh
$ make help
```
```
default: install
all: install build
h help:
install:
upgrade:
s serve:
b build:
d deploy:
```


## Serve docs

This will build the docs in memory (not to disk) and serve an auto-reloading server.

```sh
$ make serve
```

Then open in your browser:

- [localhost:8000](http://localhost:8000)


## Build docs

Build docs site to `site` directory. This is useful for a CI flow.

```sh
$ make build
```
