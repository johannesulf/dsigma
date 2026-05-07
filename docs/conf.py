from datetime import date

project = 'dsigma'
copyright = f'2024-{date.today().year}, Johannes U. Lange'
author = 'Johannes U. Lange'
extensions = ['numpydoc', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc',
              'sphinx.ext.intersphinx']
autodoc_mock_imports = ['h5py']
html_theme = 'furo'
html_logo = 'dsigma.png'
html_theme_options = {'sidebar_hide_name': True}
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'astropy': ('https://docs.astropy.org/en/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}
numpydoc_xref_param_type = True
