# Advanced
> Beyond the basic configuration and content

Once you've got the [Set up Project](setup-project.md) section, you can customize further using this guide. Or skip this and go to [Usage](usage.md).


## Navbar nesting

You can add an additional level to your navbar like this:

```yaml
nav:
  - Home: index.md
  - About: about.md
  - Foo:
      - Overview: foo/index.md
      - Bar: foo/bar.md
```

The value can either be a string (as in the first case) or a map (as in the last case). This seems to be a YAML limitation but see also [issue #1139](https://github.com/mkdocs/mkdocs/issues/1139).


## Add config options

See [Configuration](https://www.mkdocs.org/user-guide/configuration/) page on MkDocs site for options.


## Separate docs directory approach

You can also structure your project to have the set up above nested inside a `docs` directory. This is useful you have a few other directories and you want to keep the project root clean.


- `docs/`
    - `docs/`
        - `index.md`
    - `theme/`
        - `main.html`
        - `nav.html`
        - `toc.html`
    - `mkdocs.yml`


An example of this is the [Poetry](https://github.com/python-poetry/poetry/tree/master/docs) repo. That  project is also how I got into MkDocs in the first place.

## Embedding

To embed a gist, just copy and paste the embed script URL which is provided on a gist.

e.g.

```html
<script src="https://gist.github.com/MichaelCurrin/57caae30bd7b0991098e9804a9494c23.js"></script>
```
