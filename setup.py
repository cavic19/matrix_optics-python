from setuptools import setup
setup(
  name = 'matrixopt',         
  packages = ['matrixopt'],   
  version = '0.1',      
  license='MIT',        
  description = 'Matrix optics simplified',   
  author = 'David Tomecek',                  
  author_email = 'david.tomecek1@seznam.cz',     
  url = 'https://github.com/cavic19/matrixopt',   
  download_url = 'https://github.com/cavic19/matrixopt/archive/refs/tags/v_01.tar.gz',  
  keywords = ['optics', 'geometric optics', 'matrix optics'],   
  install_requires=[           
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Science/Research',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3.9',
  ],
)