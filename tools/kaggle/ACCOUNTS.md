# Adding an account to the pool

Each account is one directory under `~/.kaggle-accounts/`, named however you like —
the name is what `queue.py` takes as its `account` argument:

```text
~/.kaggle-accounts/
  otzaria/                 <- the account whose token is already installed
  yossi/
    kaggle.json
  miriam/
    kaggle.json
```

`kaggle.json` is exactly the file Kaggle's **Settings -> API -> Create New Token**
downloads. Two fields:

```json
{ "username": "their-kaggle-username", "key": "..." }
```

Drop it in and `chmod 600` it. Nothing else changes: `queue.py` points
`KAGGLE_CONFIG_DIR` at the directory per command, so accounts never overwrite each
other and there is no login step to repeat.

The directory is outside the repository on purpose. A key is a credential — anyone
holding it can act as that account — so it must not be one `git add -A` away from
being published, and each teammate can revoke theirs from the same settings page,
which is the clean way to end their participation.

Quota is per account and does not pool. `queue.py quota` reads what is actually left
on each rather than assuming.
