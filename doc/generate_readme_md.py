import pypandoc


readme = pypandoc.convert_file(
    "https://sumu.readthedocs.io/en/latest/readme.html",
    to='markdown_github-native_divs',
    format="html",
    extra_args=["--wrap=none"]
)


# Remove permalinks (and their symbols)
readme = readme.split("\n")
for i, row in enumerate(readme):
    if "Permalink to this headline" in row:
        readme[i] = row.split("<")[0]


# Remove header and footer
start, end = [i for i in range(len(readme)) if readme[i] == '-'*72]
readme = readme[start+1:end-2]


# Better formatting for rst admonitions
for i, row in enumerate(readme):
    if row == "Note" and i < len(readme)-2 and readme[i+1] == "":
        readme[i+2] = "**" + readme[i+2] + "**"
readme = "\n".join(readme).replace("Note\n\n", "> :warning: ").split("\n")


# Remove spurious newlines between list items
_readme = list()
prev_li = False
for i, row in enumerate(readme):
    if i < len(readme) - 1 and prev_li and row == "" and readme[i+1][:4] == "-   ":
        continue
    if row[:4] == "-   ":
        prev_li = True
    _readme.append(row)
readme = _readme


# Syntax highlighting for "Getting started"
_readme = list()
in_section = False
for i, row in enumerate(readme):
    if "## Getting started" in row:
        in_section = True
        _readme.append(row)
        _readme.append("```python")
        continue
    if in_section and row[:3] == "## ":
        in_section = False
        _readme.append("```")
    if in_section:
        _readme.append(row[3:])
    else:
        _readme.append(row)
readme = _readme


# Make relative links absolute
for i, row in enumerate(readme):
    if 'class="reference internal"' in row:
        readme[i] = row.replace('a href="', 'a href="https://sumu.readthedocs.io/en/latest/')


with open("../README.md", "w") as f:
    f.write('\n'.join(readme))
