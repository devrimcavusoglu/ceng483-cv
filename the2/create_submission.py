from pathlib import Path
import shutil
import tarfile


cwd = Path.cwd()
file_to_submit = "HW_notebook.ipynb"
submission_filename = "2010023_the2"

# shutil.make_archive('2010023_the2', 'gztar', cwd)


tar = tarfile.open(submission_filename + ".tar.gz", 'w:gz')
tar.add(file_to_submit)
tar.close()
