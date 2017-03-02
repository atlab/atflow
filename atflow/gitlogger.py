import git
import inspect
from functools import wraps
import datajoint as dj
import os
from datajoint import DataJointError
import datetime
import numpy as np
from collections.abc import Mapping
import warnings

_FAIL_ON_ERROR = False

class PedanticError(Exception):
    pass

def get_key_for_tuple(tup, relation):
    if isinstance(tup, np.void) or isinstance(tup, Mapping):
        retval = {name: tup[name] for name in relation.heading.primary_key}
    else:  # positional insert
        retval = {name: val for name, val in zip(relation.heading, tup) if name in relation.heading.primary_key}
    return retval


def _log_git_status(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        if not args[0].connection.in_transaction:
            with args[0].connection.transaction:
                ret = func(*args, **kwargs)
                key = get_key_for_tuple(args[1], args[0])
                args[0].GitKey().log_key(key)
        else:
            ret = func(*args, **kwargs)
            key = get_key_for_tuple(args[1], args[0])
            args[0].GitKey().log_key(key)
        return ret

    return with_logging


def gitlog(cls):
    """
    Decorator that equips a datajoint class with an additional datajoint.Part table that stores the current sha1,
    the branch, the date of the head commit,and whether the code was modified since the last commit,
    for the class representing the master table. Use the instantiated version of the decorator.
    Here is an example:
    .. code-block:: python
       :linenos:
        import datajoint as dj
        from djaddon import gitlog
        schema = dj.schema('mydb',locals())
        @schema
        @gitlog
        class MyRelation(dj.Computed):
            definition = ...
    """

    cls._raise_error_on_modified = False

    class GitKey(dj.Part):
        definition = """
        ->%s
        ---
        sha1        : varchar(40)
        branch      : varchar(50)
        modified    : int   # whether there are modified files or not
        head_date   : datetime # authored date of git head
        """ % (cls.__name__,)

        def log_key(self, key):
            path = inspect.getabsfile(cls).split('/')

            for i in reversed(range(len(path))):
                tmp_path = '/'.join(path[:i])
                if os.path.exists(tmp_path + '/.git'):
                    repo = git.Repo(tmp_path)
                    break
            else:
                raise DataJointError("%s.GitKey could not find a .git directory for %s" % (cls.__name__, cls.__name__))
            sha1, branch = repo.head.commit.name_rev.split()
            modified = (repo.git.status().find("modified") > 0) * 1
            if modified:
                if not cls._raise_error_on_modified and not _FAIL_ON_ERROR:
                    warnings.warn(
                        'You have uncommited changes. Consider committing the changes before running populate.')
                else:
                    raise PedanticError('You have uncommited changes. Commit changes before running populate!')
            key['sha1'] = sha1
            key['branch'] = branch
            key['modified'] = modified
            key['head_date'] = datetime.datetime.fromtimestamp(repo.head.commit.authored_date)
            self.insert1(key, skip_duplicates=True)

    cls.GitKey = GitKey
    cls.insert1 = _log_git_status(cls.insert1)

    return cls
